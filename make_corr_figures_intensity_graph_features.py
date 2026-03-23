import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVENT_COLOR = {"sell": "#0072B2", "repair": "#D55E00"}  # Okabe–Ito

PRETTY = {
    # ---- Intensity / outcomes (rows) ----
    "log10_dp_nei_repair": r"Repair: $\log_{10}\,\overline{\Delta p}_{nei}$",
    "p_total_repair":      r"Repair: $\overline{p}$",
    "mean_nei_logit_repair":   r"Repair: $\overline{\lambda}^{nei}$",
    "mean_total_logit_repair": r"Repair: $\overline{\lambda}$",

    "log10_dp_nei_sell": r"Sell: $\log_{10}\,\overline{\Delta p}_{nei}$",
    "p_total_sell":      r"Sell: $\overline{p}$",
    "mean_nei_logit_sell":   r"Sell: $\overline{\lambda}^{nei}$",
    "mean_total_logit_sell": r"Sell: $\overline{\lambda}$",

    # ---- Exposure ----
    "mean_ian_flood_depth": "Flood depth (mean)",
    "mean_ian_est_loss":    "Est. loss (mean)",
    "damage_expected_severity": "Damage severity (exp.)",

    # ---- Community / socio ----
    "community_size_households": "Community size",
    "mean_population_density":   "Pop. density",
    "occupancy_prop_single_family": "Single-family share",
    "occupancy_prop_multi_family":  "Multi-family share",
    "mean_income":               "Income (median)",
    "mean_housing_affordability_index": "Affordability",
    "mean_educational_level":    "Education",
    "mean_age":                  "Age (mean)",

    # ---- Building ----
    "mean_yearbuilt":            "Year built (mean)",
    "mean_house_age":            "Building age",
    "mean_house_market_or_assessed_value": "Home value (mean)",

    # ---- Graph ----
    "w_density_meanK":        "Edge density (wtd.)",
    "bin_density_meanK":      "Edge density (bin.)",
    "gini_out_strength_meanK":"Out-strength Gini",
}


# -------------------------
# Correlation + p-values
# -------------------------
def _pairwise_clean(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]

def spearman_corr(a, b):
    a, b = _pairwise_clean(a, b)
    if a.size < 5:
        return np.nan
    ra = pd.Series(a).rank(method="average").values
    rb = pd.Series(b).rank(method="average").values
    # pearson on ranks
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (np.sqrt((ra**2).sum()) * np.sqrt((rb**2).sum()))
    if denom <= 0:
        return np.nan
    return float((ra * rb).sum() / denom)

def perm_pvalue_corr(a, b, method="spearman", n_perm=2000, seed=0):
    """
    Two-sided permutation test: shuffle b relative to a.
    Returns p-value for observed correlation.
    """
    a, b = _pairwise_clean(a, b)
    if a.size < 8:
        return np.nan
    if pd.Series(a).nunique(dropna=True) < 3 or pd.Series(b).nunique(dropna=True) < 3:
        return np.nan

    rng = np.random.RandomState(seed)
    if method == "pearson":
        obs = pd.Series(a).corr(pd.Series(b), method="pearson")
        if not np.isfinite(obs):
            return np.nan
        cnt = 0
        for _ in range(n_perm):
            bp = rng.permutation(b)
            rp = pd.Series(a).corr(pd.Series(bp), method="pearson")
            if np.isfinite(rp) and (abs(rp) >= abs(obs)):
                cnt += 1
    else:
        obs = spearman_corr(a, b)
        if not np.isfinite(obs):
            return np.nan
        cnt = 0
        for _ in range(n_perm):
            bp = rng.permutation(b)
            rp = spearman_corr(a, bp)
            if np.isfinite(rp) and (abs(rp) >= abs(obs)):
                cnt += 1

    # +1 smoothing
    return float((cnt + 1) / (n_perm + 1))

def fdr_bh(pvals_1d):
    """
    Benjamini–Hochberg FDR correction.
    Input: 1D array of p-values (may contain NaN).
    Output: q-values same shape (NaN preserved).
    """
    p = np.asarray(pvals_1d, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    msk = np.isfinite(p)
    pv = p[msk]
    if pv.size == 0:
        return q
    order = np.argsort(pv)
    pv_sorted = pv[order]
    m = pv_sorted.size
    ranks = np.arange(1, m + 1)
    q_sorted = pv_sorted * m / ranks
    # enforce monotone non-increasing when traversed backwards
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_out = np.empty_like(pv_sorted)
    q_out[order] = np.clip(q_sorted, 0, 1)
    q[msk] = q_out
    return q

def stars_from_q(q):
    if not np.isfinite(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""

def compute_or_load_crosscorr(df, row_vars, col_vars, method, n_perm, seed, cache_npz=None, recompute=False):
    if cache_npz is not None and (not recompute) and os.path.exists(cache_npz):
        z = np.load(cache_npz, allow_pickle=True)
        return {
            "R": z["R"], "P": z["P"],
            "row_vars": list(z["row_vars"]),
            "col_vars": list(z["col_vars"]),
            "method": str(z["method"]),
            "n_perm": int(z["n_perm"]),
            "seed": int(z["seed"]),
        }

    R = np.full((len(row_vars), len(col_vars)), np.nan, dtype=float)
    P = np.full((len(row_vars), len(col_vars)), np.nan, dtype=float)

    base_seed = int(seed)
    for i, rv in enumerate(row_vars):
        a = df[rv].values
        if pd.Series(a).nunique(dropna=True) < 3:
            continue
        for j, cv in enumerate(col_vars):
            b = df[cv].values
            if pd.Series(b).nunique(dropna=True) < 3:
                continue

            r = spearman_corr(a, b) if method == "spearman" else pd.Series(a).corr(pd.Series(b), method="pearson")
            R[i, j] = r

            if np.isfinite(r):
                P[i, j] = perm_pvalue_corr(
                    a, b, method=method, n_perm=n_perm,
                    seed=base_seed + 9973*i + 101*j
                )

    if cache_npz is not None:
        np.savez_compressed(
            cache_npz,
            R=R, P=P,
            row_vars=np.array(row_vars, dtype=object),
            col_vars=np.array(col_vars, dtype=object),
            method=np.array(method, dtype=object),
            n_perm=np.array(int(n_perm)),
            seed=np.array(int(seed)),
        )
        print(f"[OK] Saved correlation cache: {cache_npz}")

    return {"R": R, "P": P, "row_vars": row_vars, "col_vars": col_vars,
            "method": method, "n_perm": n_perm, "seed": seed}

# -------------------------
# Grouping helpers
# -------------------------
def build_grouped_columns(groups, available_cols):
    """
    groups: list of (group_name, [col1, col2, ...])
    returns: ordered_cols, group_spans (name, start_idx, end_idx) in the ORDERED list
    """
    ordered = []
    spans = []
    start = 0
    for gname, cols in groups:
        cols_kept = [c for c in cols if c in available_cols]
        if len(cols_kept) == 0:
            continue
        ordered.extend(cols_kept)
        end = start + len(cols_kept) - 1
        spans.append((gname, start, end))
        start = end + 1
    return ordered, spans

def insert_gaps_for_groups_rows(mat, rows, spans, gap_rows=1):
    """
    Insert NaN 'gap rows' between row groups for visual separation.
    Inputs:
      mat:   [R, C]
      rows:  list of row names (len R)
      spans: list of (group_name, start_idx, end_idx) in the rows list
    Returns:
      new_mat, new_rows, new_spans
    """
    mat = np.asarray(mat)
    R, C = mat.shape
    new_rows = []
    new_blocks = []
    new_spans = []

    cur = 0
    for gi, (gname, s, e) in enumerate(spans):
        block = mat[s:e+1, :]
        block_rows = rows[s:e+1]

        new_blocks.append(block)
        new_rows.extend(block_rows)

        block_start = cur
        block_end = cur + (e - s)
        new_spans.append((gname, block_start, block_end))
        cur = block_end + 1

        if gi < len(spans) - 1 and gap_rows and gap_rows > 0:
            gap = np.full((gap_rows, C), np.nan, dtype=float)
            new_blocks.append(gap)
            new_rows.extend([""] * gap_rows)
            cur += gap_rows

    new_mat = np.concatenate(new_blocks, axis=0) if len(new_blocks) else mat
    return new_mat, new_rows, new_spans


def insert_gaps_for_groups(mat, cols, spans, gap_cols=1):
    """
    Insert NaN "gap columns" between groups for visual separation.
    Returns: new_mat, new_cols, new_spans, real_x_positions
    - new_spans indices refer to new_cols
    - real_x_positions maps original col indices -> new col index
    """
    n_rows, n_cols = mat.shape
    new_cols = []
    new_mat_list = []
    new_spans = []

    real_x_positions = []
    cur = 0

    for gi, (gname, s, e) in enumerate(spans):
        # group block
        block = mat[:, s:e+1]
        block_cols = cols[s:e+1]

        new_mat_list.append(block)
        new_cols.extend(block_cols)

        block_start = cur
        block_end = cur + (e - s)
        new_spans.append((gname, block_start, block_end))

        # record mapping for tick locations
        for k in range(e - s + 1):
            real_x_positions.append(cur + k)

        cur = block_end + 1

        # add gaps (except after last group)
        if gi < len(spans) - 1 and gap_cols > 0:
            gap = np.full((n_rows, gap_cols), np.nan, dtype=float)
            new_mat_list.append(gap)
            new_cols.extend([""] * gap_cols)
            cur += gap_cols

    new_mat = np.concatenate(new_mat_list, axis=1) if len(new_mat_list) else mat
    return new_mat, new_cols, new_spans, real_x_positions

# -------------------------
# Style (publication-ish)
# -------------------------
def apply_pub_style(fig_w=10.5, fig_h=6.5, base_font=13, save_dpi=800):
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.figsize": (fig_w, fig_h),
        "figure.dpi": 200,
        "savefig.dpi": save_dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "DejaVu Sans",
        "font.size": base_font,
        "axes.labelsize": base_font + 1,
        "xtick.labelsize": base_font,
        "ytick.labelsize": base_font,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })


# -------------------------
# Utilities
# -------------------------
def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out

def safe_save(fig, out_png):
    fig.tight_layout()
    fig.savefig(out_png)
    if out_png.lower().endswith(".png"):
        fig.savefig(out_png[:-4] + ".pdf")
    plt.close(fig)

def mean_over_eligible(values_TN, elig_TN):
    m = elig_TN.astype(bool)
    denom = m.sum()
    if denom == 0:
        return np.nan
    return float((values_TN * m).sum() / denom)

def log10_eps(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    return np.log10(x + eps)

def spearman_corr(a, b):
    # Spearman = Pearson on ranks
    a = pd.Series(a).rank(method="average").values.astype(np.float64)
    b = pd.Series(b).rank(method="average").values.astype(np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return np.nan
    a = a[m]; b = b[m]
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.sqrt((a*a).sum()) * np.sqrt((b*b).sum()))
    if denom <= 0:
        return np.nan
    return float((a*b).sum() / denom)

def residualize(y, X):
    """
    Residualize y on X via least squares. Adds intercept.
    y: (n,)
    X: (n,p)
    returns residuals (n,)
    """
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    n = y.shape[0]
    X_ = np.concatenate([np.ones((n, 1)), X], axis=1)
    # handle NaNs by masking rows
    m = np.isfinite(y) & np.all(np.isfinite(X_), axis=1)
    if m.sum() < 10:
        return np.full_like(y, np.nan)
    y_m = y[m]
    X_m = X_[m]
    beta, *_ = np.linalg.lstsq(X_m, y_m, rcond=None)
    y_hat = X_ @ beta
    res = y - y_hat
    res[~m] = np.nan
    return res


# -------------------------
# Loaders
# -------------------------
def load_processed(processed_npz: str):
    d = np.load(processed_npz, allow_pickle=True)
    obj = d["processed"]
    return obj.item() if isinstance(obj, np.ndarray) else obj

def load_graphs(graphs_npz: str):
    """
    Robust loader: graphs["graphs"] can contain dicts directly OR 0-d wrappers.
    Returns dict gid -> graph_entry.
    """
    d = np.load(graphs_npz, allow_pickle=True)
    graphs = d["graphs"]
    if isinstance(graphs, dict):
        graphs_iter = [graphs]
    else:
        graphs_iter = list(graphs)

    out = {}
    for g in graphs_iter:
        if isinstance(g, dict):
            gg = g
        elif isinstance(g, np.ndarray) and g.shape == ():
            gg = g.item()
        else:
            try:
                gg = g.item()
            except Exception:
                gg = g
        if not isinstance(gg, dict):
            continue
        gid = int(gg["community_global_index"])
        out[gid] = gg
    return out


# -------------------------
# Build community-level table
# -------------------------
def compute_influence_metrics(processed, start_month=1, min_points=500, eps=1e-12):
    """
    Returns df: gid, dp_nei_{sell,repair}, log10_dp_nei_{sell,repair},
               p_total_{sell,repair}, mean_nei_logit_{sell,repair}, mean_total_logit_{sell,repair}
    """
    comms = processed.get("communities", {})
    rows = []
    for gid_key, gd in comms.items():
        try:
            gid = int(gid_key)
        except Exception:
            continue

        row = {"gid": gid}
        ok = False

        for event in ["sell", "repair"]:
            if event not in gd:
                continue
            out = gd[event]
            total = out["total"]      # [T,N] logit
            nei   = out["nei"]        # [T,N] logit
            elig  = out["eligible"]   # [T,N] bool

            T = total.shape[0]
            t0 = max(0, int(start_month))
            total = total[t0:T]
            nei   = nei[t0:T]
            elig  = elig[t0:T]

            if elig.sum() < min_points:
                continue

            # neighbor influence in probability space
            dp = sigmoid(total) - sigmoid(total - nei)

            row[f"dp_nei_{event}"] = mean_over_eligible(dp, elig)
            row[f"log10_dp_nei_{event}"] = log10_eps(row[f"dp_nei_{event}"], eps=eps)

            # total intensity in probability space
            row[f"p_total_{event}"] = mean_over_eligible(sigmoid(total), elig)

            # logit-space means (often cleaner signal)
            row[f"mean_nei_logit_{event}"] = mean_over_eligible(nei, elig)
            row[f"mean_total_logit_{event}"] = mean_over_eligible(total, elig)

            ok = True

        if ok:
            rows.append(row)

    return pd.DataFrame(rows)


def gini(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if np.allclose(x, 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def network_metrics_from_A(A_knn, threshold=0.0, topq=None):
    K, N, _ = A_knn.shape
    if N <= 1:
        return {}

    denom = N * (N - 1)
    wdens, bdens, gouts = [], [], []
    mean_out_strengths = []
    spec_rads = []
    top10_mass = []

    for k in range(K):
        A = A_knn[k].astype(np.float64).copy()
        np.fill_diagonal(A, 0.0)

        wdens.append(A.sum() / denom)

        if topq is not None:
            flat = A.reshape(-1)
            flat = flat[flat > 0]
            if flat.size == 0:
                bdens.append(0.0)
            else:
                tau = np.quantile(flat, 1.0 - float(topq))
                bdens.append(((A >= tau).sum()) / denom)
        else:
            bdens.append(((A > float(threshold)).sum()) / denom)

        out_strength = A.sum(axis=1)
        mean_out_strengths.append(float(np.mean(out_strength)))
        gouts.append(gini(out_strength))

        # spectral radius (largest eigenvalue magnitude)
        # (use eigvals; N per community should be manageable)
        try:
            ev = np.linalg.eigvals(A)
            spec_rads.append(float(np.max(np.abs(ev))))
        except Exception:
            spec_rads.append(np.nan)

        # concentration: fraction of total weight in top 10% edges
        flat = A.reshape(-1)
        flat = flat[flat > 0]
        if flat.size == 0:
            top10_mass.append(0.0)
        else:
            q = np.quantile(flat, 0.90)
            top10_mass.append(float(flat[flat >= q].sum() / (flat.sum() + 1e-12)))

    return {
        "w_density_meanK": float(np.nanmean(wdens)),
        "bin_density_meanK": float(np.nanmean(bdens)),
        "mean_out_strength_meanK": float(np.nanmean(mean_out_strengths)),
        "gini_out_strength_meanK": float(np.nanmean(gouts)),
        "spectral_radius_meanK": float(np.nanmean(spec_rads)),
        "top10_edge_mass_meanK": float(np.nanmean(top10_mass)),
        "N_graph": int(N),
        "K_graph": int(K),
    }



def compute_network_metrics(graphs_dict, threshold=0.0, topq=None):
    rows = []
    for gid, g in graphs_dict.items():
        A = g.get("A", None)
        if A is None:
            continue
        m = network_metrics_from_A(A, threshold=threshold, topq=topq)
        if not m:
            continue
        m["gid"] = int(gid)
        # if you saved geoid
        if "geoid" in g:
            m["cbg"] = str(g["geoid"])
        rows.append(m)
    return pd.DataFrame(rows)


def merge_all(cbg_csv, infl_df, net_df):
    feat = pd.read_csv(cbg_csv)

    # Robust merge strategy:
    # 1) If net_df has cbg and feat has cbg, merge by cbg.
    # 2) Else fallback: assume row index corresponds to gid (your earlier pipeline).
    if "cbg" in feat.columns and "cbg" in net_df.columns:
        feat["cbg"] = feat["cbg"].astype(str)
        net_df["cbg"] = net_df["cbg"].astype(str)
        df = net_df.merge(infl_df, on="gid", how="inner")
        df = df.merge(feat, on="cbg", how="left")
    else:
        feat = feat.reset_index().rename(columns={"index": "gid"})
        df = infl_df.merge(net_df, on="gid", how="inner").merge(feat, on="gid", how="left")

    return df


# -------------------------
# Figure: cross-correlation heatmap (outcomes vs predictors)
# -------------------------
def plot_graph_feature_heatmap(df, graph_cols, feature_cols, method, out_path):
    graph_cols = [c for c in graph_cols if c in df.columns]
    feature_cols = [c for c in feature_cols if c in df.columns]

    mat = np.full((len(graph_cols), len(feature_cols)), np.nan, dtype=np.float64)
    for i, gc in enumerate(graph_cols):
        for j, fc in enumerate(feature_cols):
            a = df[gc].values
            b = df[fc].values
            if pd.Series(a).nunique(dropna=True) < 3 or pd.Series(b).nunique(dropna=True) < 3:
                continue
            mat[i, j] = spearman_corr(a, b) if method == "spearman" else pd.Series(a).corr(pd.Series(b), method="pearson")

    # drop empty columns
    keep_c = np.isfinite(mat).sum(axis=0) > 0
    mat = mat[:, keep_c]
    feature_cols = [c for c, k in zip(feature_cols, keep_c) if k]

    keep_r = np.isfinite(mat).sum(axis=1) > 0
    mat = mat[keep_r, :]
    graph_cols = [c for c, k in zip(graph_cols, keep_r) if k]

    fig_w = max(10.0, 0.55 * len(feature_cols) + 3.0)
    fig_h = max(4.2, 0.55 * len(graph_cols) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_yticks(np.arange(len(graph_cols)))
    ax.set_yticklabels(graph_cols)

    ax.set_xticks(np.arange(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha="right")

    ax.set_xticks(np.arange(-.5, len(feature_cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(graph_cols), 1), minor=True)
    ax.grid(which="minor", color="0.92", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(f"{method} correlation")

    safe_save(fig, out_path)

def draw_row_group_labels(ax, row_spans_g, x_axes=-0.32, fontsize=14):
    trans = ax.get_yaxis_transform()  # x in axes, y in data
    for gname, s, e in row_spans_g:
        y_mid = 0.5 * (s + e)
        ax.text(
            x_axes, y_mid, gname,
            transform=trans,
            rotation=90,
            va="center", ha="right",
            fontsize=fontsize, fontweight="bold",
            clip_on=False,   # IMPORTANT
        )






def plot_crosscorr_grouped_rowscols_with_stars(
    df,
    outcome_groups,    # list of (row_group_name, [row_vars...])
    predictor_groups,  # list of (col_group_name, [col_vars...])
    method="spearman",
    out_path="crosscorr.png",
    n_perm=2000,
    seed=42,
    use_fdr=True,
    gap_cols=1,
    gap_rows=1,
    drop_empty=True,
    recompute=False,
    cache_npz=None
):
    available = set(df.columns)

    # Build ordered rows + row spans
    row_vars, row_spans = build_grouped_columns(outcome_groups, available)
    col_vars, col_spans = build_grouped_columns(predictor_groups, available)

    if len(row_vars) == 0 or len(col_vars) == 0:
        print("[WARN] No rows or columns available for heatmap.")
        return

    # Compute R and P (rows x cols)
    # R = np.full((len(row_vars), len(col_vars)), np.nan, dtype=float)
    # P = np.full((len(row_vars), len(col_vars)), np.nan, dtype=float)

    # base_seed = int(seed)

    # for i, rv in enumerate(row_vars):
    #     a = df[rv].values
    #     if pd.Series(a).nunique(dropna=True) < 3:
    #         continue
    #     for j, cv in enumerate(col_vars):
    #         b = df[cv].values
    #         if pd.Series(b).nunique(dropna=True) < 3:
    #             continue

    #         if method == "pearson":
    #             r = pd.Series(a).corr(pd.Series(b), method="pearson")
    #         else:
    #             r = spearman_corr(a, b)
    #         R[i, j] = r

    #         # permutation p-value
    #         if np.isfinite(r):
    #             P[i, j] = perm_pvalue_corr(a, b, method=method, n_perm=n_perm,
    #                                        seed=base_seed + 9973*i + 101*j)
    cache = compute_or_load_crosscorr(
        df, row_vars, col_vars,
        method=method, n_perm=n_perm, seed=seed,
        cache_npz=cache_npz, recompute=recompute
    )
    R, P = cache["R"], cache["P"]
    finite_p = P[np.isfinite(P)]
    print("min p:", np.nanmin(finite_p))
    print("p<0.05 count:", np.sum(finite_p < 0.05), " / ", finite_p.size)

    Q = fdr_bh(P.reshape(-1)).reshape(P.shape)
    finite_q = Q[np.isfinite(Q)]
    print("min q:", np.nanmin(finite_q))
    print("q<0.05 count:", np.sum(finite_q < 0.05), "/", finite_q.size)


    # Drop empty columns/rows (all-NaN)
    if drop_empty:
        keep_c = np.isfinite(R).sum(axis=0) > 0
        keep_r = np.isfinite(R).sum(axis=1) > 0
        R = R[keep_r][:, keep_c]
        P = P[keep_r][:, keep_c]
        row_vars = [v for v, k in zip(row_vars, keep_r) if k]
        col_vars = [v for v, k in zip(col_vars, keep_c) if k]

        # rebuild spans after filtering
        idx_c = {v: i for i, v in enumerate(col_vars)}
        col_spans2 = []
        for gname, cols in predictor_groups:
            cols_kept = [c for c in cols if c in idx_c]
            if not cols_kept:
                continue
            inds = sorted([idx_c[c] for c in cols_kept])
            col_spans2.append((gname, inds[0], inds[-1]))
        col_spans = col_spans2

        idx_r = {v: i for i, v in enumerate(row_vars)}
        row_spans2 = []
        for gname, rows in outcome_groups:
            rows_kept = [c for c in rows if c in idx_r]
            if not rows_kept:
                continue
            inds = sorted([idx_r[c] for c in rows_kept])
            row_spans2.append((gname, inds[0], inds[-1]))
        row_spans = row_spans2

    # FDR
    Q = fdr_bh(P.reshape(-1)).reshape(P.shape) if use_fdr else P

    # Insert gaps for columns
    Rg, cols_g, col_spans_g, _ = insert_gaps_for_groups(R, col_vars, col_spans, gap_cols=gap_cols)
    Qg, _, _, _ = insert_gaps_for_groups(Q, col_vars, col_spans, gap_cols=gap_cols)

    # Insert gaps for rows
    Rg2, rows_g, row_spans_g = insert_gaps_for_groups_rows(Rg, row_vars, row_spans, gap_rows=gap_rows)
    Qg2, _, _ = insert_gaps_for_groups_rows(Qg, row_vars, row_spans, gap_rows=gap_rows)

   
    # Plot
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("white")

    fig_w = max(12.0, 0.55 * len(cols_g) + 3.0)
    fig_h = max(6.0, 0.55 * len(rows_g) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

    im = ax.imshow(Rg2, vmin=-1, vmax=1, cmap=cmap, aspect="auto")

    # ax.set_xticks(np.arange(len(cols_g)))
    # # ax.set_xticklabels(cols_g, rotation=45, ha="right", fontsize=12)
    # ax.set_yticks(np.arange(len(rows_g)))
    # # ax.set_yticklabels(rows_g, fontsize=13)

    def pretty_list(names):
        out = []
        for n in names:
            if n == "" or n is None:
                out.append("")
            else:
                out.append(PRETTY.get(n, n))
        return out

     # call it AFTER you compute row_spans_g and AFTER you inserted gaps (rows_g)
    # draw_row_group_labels(ax, row_spans_g, x_axes=-0.24, fontsize=14)

    ax.set_xticks(np.arange(len(cols_g)))
    ax.set_xticklabels(pretty_list(cols_g), rotation=45, ha="right", fontsize=11)

    ax.set_yticks(np.arange(len(rows_g)))
    ax.set_yticklabels(pretty_list(rows_g), fontsize=12)     # <-- RIGHT

    # light grid
    ax.set_xticks(np.arange(-.5, len(cols_g), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows_g), 1), minor=True)
    ax.grid(which="minor", color="0.92", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    # group separators + labels (top for columns)
    for gname, s, e in col_spans_g:
        ax.axvline(e + 0.5, color="0.75", lw=1.4)
        xc = (s + e) / 2.0
        ax.text(xc, 1.02, gname, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=14, fontweight="bold")

    # group separators + labels (left for rows)
    for gname, s, e in row_spans_g:
        ax.axhline(e + 0.5, color="0.75", lw=1.4)
        yc = (s + e) / 2.0
        # ax.text(-0.02, yc, gname, transform=ax.get_yaxis_transform(),
        #         ha="right", va="center", fontsize=14, fontweight="bold", rotation=90)

    draw_row_group_labels(ax, row_spans_g, x_axes=-0.32, fontsize=14)

    
    # stars
    for i in range(Rg2.shape[0]):
        for j in range(Rg2.shape[1]):
            if not np.isfinite(Rg2[i, j]):
                continue
            st = stars_from_q(Qg2[i, j])
            if st:
                ax.text(j, i, st, ha="center", va="center", fontsize=11, color="black")

    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(f"{method} correlation", fontsize=14)
    cb.ax.tick_params(labelsize=12)

    ax.tick_params(axis="y", pad=10)          # more space between ticks and labels
    fig.subplots_adjust(left=0.28)            # increase if still tight (e.g., 0.30–0.35)

    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path[:-4] + ".pdf")
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")




# -------------------------
# Figure: partial correlation bars (graph vs influence controlling for confounders)
# -------------------------
def plot_partialcorr_bars(df, event, ycol, xcols, controls, out_path, use_spearman=True):
    # residualize y and each x on controls, then correlate residuals
    Xc = df[controls].values.astype(np.float64)

    y = df[ycol].values.astype(np.float64)
    y_res = residualize(y, Xc)

    vals = []
    for xc in xcols:
        x = df[xc].values.astype(np.float64)
        x_res = residualize(x, Xc)
        if use_spearman:
            r = spearman_corr(x_res, y_res)
        else:
            r = pd.Series(x_res).corr(pd.Series(y_res), method="pearson")
        vals.append(r)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.bar(np.arange(len(xcols)), vals, color=EVENT_COLOR[event], alpha=0.85)
    ax.axhline(0, color="0.85", linewidth=1.0)
    ax.set_xticks(np.arange(len(xcols)))
    ax.set_xticklabels(xcols, rotation=25, ha="right")
    ax.set_ylabel("Partial correlation (residualized)")
    safe_save(fig, out_path)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_npz", type=str, required=True)
    ap.add_argument("--graphs_npz", type=str, required=True)
    ap.add_argument("--cbg_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="corr_figs")

    ap.add_argument("--start_month", type=int, default=1)
    ap.add_argument("--min_points", type=int, default=500)
    ap.add_argument("--eps", type=float, default=1e-12)

    ap.add_argument("--method", type=str, default="spearman", choices=["spearman", "pearson"])
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--topq", type=float, default=None, help="e.g., 0.05 for top 5% edges")
    ap.add_argument("--n_perm", type=int, default=2000, help="Permutation tests per cell")
    ap.add_argument("--gap_cols", type=int, default=1, help="Number of blank columns between groups")
    ap.add_argument("--gap_rows", type=int, default=1, help="Number of blank columns between groups")
    ap.add_argument("--no_fdr", action="store_true", help="Use raw permutation p-values instead of BH-FDR")
    ap.add_argument("--cache_npz", type=str, default=None,
               help="If set, save/load computed correlation matrices to avoid recomputation.")
    ap.add_argument("--recompute", action="store_true",
               help="Ignore cache and recompute correlations.")
    ap.add_argument("--community_table_csv", type=str, default=None,
               help="Precomputed community-level table CSV. If provided and exists, load it and skip building.")
    ap.add_argument("--recompute_table", action="store_true",
               help="Ignore --community_table_csv and rebuild the table from NPZ/graphs/cbg.")


    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    apply_pub_style()

    if args.community_table_csv is not None and os.path.exists(args.community_table_csv):
        df = pd.read_csv(args.community_table_csv)
        print(f"[OK] Loaded cached community table: {args.community_table_csv} | rows={len(df)}")
    else:
        processed = load_processed(args.processed_npz)
        graphs = load_graphs(args.graphs_npz)


        infl = compute_influence_metrics(
            processed, start_month=args.start_month, min_points=args.min_points, eps=args.eps
        )
        net = compute_network_metrics(graphs, threshold=args.threshold, topq=args.topq)

        df = merge_all(args.cbg_csv, infl, net)
        out_csv = os.path.join(args.out_dir, "community_table_intensity_graph_features.csv")
        df.to_csv(out_csv, index=False)
        print(f"[OK] Wrote merged table: {out_csv} (rows={len(df)})")
    
    graph_cols = [
        "w_density_meanK",
        "bin_density_meanK",
        "mean_out_strength_meanK",
        "gini_out_strength_meanK",
        "spectral_radius_meanK",
        "top10_edge_mass_meanK",
        "N_graph",
    ]

    # features only (exclude any neural outcomes + exclude graph cols themselves)
    feature_cols = [
        # exposure
        "mean_ian_flood_depth", "mean_ian_est_loss", "damage_expected_severity",
        # community structure
        "community_size_households", "mean_population_density",
        "occupancy_prop_single_family", "occupancy_prop_multi_family",
        # socio-demographic
        "mean_income", "mean_housing_affordability_index", "mean_educational_level", "mean_age",
        # building
        "mean_yearbuilt", "mean_house_age", "mean_house_market_or_assessed_value",
    ]

    # out_path = os.path.join(args.out_dir, "FigX_graph_vs_features_corr.png")
    # plot_graph_feature_heatmap(df, graph_cols, feature_cols, args.method, out_path)
    # print(f"[OK] Saved: {out_path}")

    # 8-row combined outcomes (repair then sell)
    outcome_cols = [
        ("Intensity", [
            "log10_dp_nei_repair", "p_total_repair", "mean_nei_logit_repair", "mean_total_logit_repair",
            "log10_dp_nei_sell",   "p_total_sell",   "mean_nei_logit_sell",   "mean_total_logit_sell",
        ]),
        ("Graph", graph_cols)]


    # Define predictor groups in the order you want in the figure
    predictor_groups = [
        ("Exposure", [
            "mean_ian_flood_depth",
            "mean_ian_est_loss",
            "damage_expected_severity",
        ]),
        ("Socio", [
            "mean_income",
            "mean_housing_affordability_index",
            "mean_educational_level",
            "mean_age",
            "occupancy_prop_single_family",
            "occupancy_prop_multi_family",
        ]),
        ("Building", [
            "mean_yearbuilt",
            "mean_house_age",
            "mean_house_market_or_assessed_value",
        ]),
        ("Graph", [
            "w_density_meanK",
            "bin_density_meanK",
            "mean_out_strength_meanK",
            "gini_out_strength_meanK",
            "spectral_radius_meanK",
            "top10_edge_mass_meanK",
            "N_graph",
        ]),
    ]

    out_path = os.path.join(args.out_dir, "Fig2_crosscorr_rowsIntensityPlusGraph_colsGrouped_stars.png")
    plot_crosscorr_grouped_rowscols_with_stars(
        df=df,
        outcome_groups=outcome_cols,
        predictor_groups=predictor_groups,
        method=args.method,
        out_path=out_path,
        n_perm=args.n_perm,
        use_fdr=not args.no_fdr,
        gap_cols=args.gap_cols,
        gap_rows=args.gap_rows,
        recompute=args.recompute,
        cache_npz=args.cache_npz,
    )


    # Partial correlation: graph -> neighbor influence, controlling for exposure + size + density (+income)
    graph_cols = [c for c in ["w_density_meanK", "bin_density_meanK", "gini_out_strength_meanK"] if c in df.columns]
    controls = [c for c in ["mean_ian_flood_depth", "mean_ian_est_loss",
                            "community_size_households", "mean_population_density",
                            "mean_income"] if c in df.columns]

    for event in ["sell", "repair"]:
        ycol = f"log10_dp_nei_{event}"
        if ycol not in df.columns or len(graph_cols) == 0 or len(controls) == 0:
            continue
        out_path = os.path.join(args.out_dir, f"Fig3_partialcorr_graph_to_dpnei_{event}.png")
        plot_partialcorr_bars(df, event, ycol, graph_cols, controls, out_path,
                              use_spearman=(args.method == "spearman"))
        print(f"[OK] Saved: {out_path}")

    print(f"[DONE] Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
