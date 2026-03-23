import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVENT_COLOR = {
    "sell":  "#0072B2",  # blue (Okabe–Ito)
    "repair":"#D55E00",  # vermillion (Okabe–Ito)
}


def apply_nature_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.figsize": (7.5, 4.8),
        "figure.dpi": 200,
        "savefig.dpi": 800,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        "font.family": "DejaVu Sans",
        "font.size": 12,          # base
        "axes.titlesize": 12,
        "axes.labelsize": 13,     # axis labels
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,

        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,

        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,

        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "legend.frameon": False,
    })

apply_nature_style()


def _load_processed(processed_npz: str):
    d = np.load(processed_npz, allow_pickle=True)
    if "processed" not in d:
        raise KeyError(f"'processed' not found. Keys={list(d.keys())}")
    obj = d["processed"]
    return obj.item() if isinstance(obj, np.ndarray) else obj


def _get_window_indices(processed: dict, window: str):
    T_train = int(processed.get("T_train", 0))
    T_meta = int(processed.get("T_meta", 0))
    if T_meta <= 0:
        meta = processed.get("meta", {})
        T_meta = int(meta.get("T", 0))

    if window == "train":
        return 0, T_train
    if window == "test":
        return T_train, T_meta
    return 0, T_meta


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def mean_over_nodes_per_t(values_TN: np.ndarray, eligible_TN: np.ndarray):
    mask = eligible_TN.astype(bool)
    denom = mask.sum(axis=1).astype(np.float64)  # [T]
    num = (values_TN * mask).sum(axis=1).astype(np.float64)
    mean_t = np.full(values_TN.shape[0], np.nan, dtype=np.float64)
    ok = denom > 0
    mean_t[ok] = num[ok] / denom[ok]
    return mean_t


def plot_mean_nei_timeseries(processed: dict, event: str, window: str, out_path: str,
                             alpha_faded: float = 0.08, lw_faded: float = 1.0, lw_mean: float = 3.0,
                             max_comms: int = None, seed: int = 0):
    t0, t1 = _get_window_indices(processed, window)
    comms = processed.get("communities", {})
    if not comms:
        raise ValueError("processed['communities'] is empty.")

    gids = [gid for gid in comms.keys() if event in comms[gid]]
    if len(gids) == 0:
        raise ValueError(f"No communities contain event='{event}'.")

    # optional subsample for readability
    rng = np.random.RandomState(seed)
    if max_comms is not None and len(gids) > max_comms:
        gids = list(rng.choice(gids, size=max_comms, replace=False))

    series_list = []
    max_T = 0

    for gid in gids:
        out = comms[gid][event]
        nei = out["nei"]          # [T,N] (logit neighbor term)
        elig = out["eligible"]    # [T,N] bool

        T_use = nei.shape[0]
        tt0 = max(0, min(t0, T_use))
        tt0 = max(tt0, 1)
        tt1 = max(0, min(t1, T_use))
        if tt1 <= tt0:
            continue

        mean_t = mean_over_nodes_per_t(nei[tt0:tt1], elig[tt0:tt1])
        series_list.append(mean_t)
        max_T = max(max_T, mean_t.shape[0])

    if len(series_list) == 0:
        raise ValueError(f"No valid time series for event='{event}' in window='{window}'.")

    S = np.full((len(series_list), max_T), np.nan, dtype=np.float64)
    for i, s in enumerate(series_list):
        S[i, : s.shape[0]] = s

    mean_across_comm = np.nanmean(S, axis=0)

    fig, ax = plt.subplots()

    # start from month 1 (skip t=0)
    start_month = 1
    x = np.arange(start_month, start_month + max_T)

    main = EVENT_COLOR.get(event, "#0072B2")  # default blue
    faint_alpha = 0.10        # community lines transparency
    faint_lw = 0.8
    mean_lw = 3.0

    # community curves: same hue, very faint
    for i in range(S.shape[0]):
        ax.plot(
            x, S[i],
            color=main,
            alpha=faint_alpha,
            linewidth=faint_lw,
            zorder=1
        )

    # mean curve: saturated, thicker
    ax.plot(
        x, mean_across_comm,
        color=main,
        linewidth=mean_lw,
        zorder=3
    )

    ax.set_xlabel("Month")
    ax.set_ylabel(r"Mean neighbor excitation (logit)")

    # remove title (as requested)
    # ax.set_title(...)

    # subtle zero line (optional)
    ax.axhline(0, color="0.85", linewidth=1.0, zorder=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=800)
    if out_path.lower().endswith(".png"):
        fig.savefig(out_path[:-4] + ".pdf")
    plt.close(fig)


def plot_overlay_distributions_total_prob(processed: dict, event: str, window: str, out_path: str,
                                         bins: int = 50, alpha_faded: float = 0.06, lw_faded: float = 1.0,
                                         lw_mean: float = 3.0, max_comms: int = None, seed: int = 0,
                                         min_points_per_comm: int = 50):
    """
    One distribution per community, overlayed:
      For each community:
        samples = sigmoid(total_logit[t,i]) over eligible (t,i) in window
        hist density on common bins -> line (faded)
      Then plot mean density across communities -> solid
    """
    t0, t1 = _get_window_indices(processed, window)
    comms = processed.get("communities", {})
    if not comms:
        raise ValueError("processed['communities'] is empty.")

    gids = [gid for gid in comms.keys() if event in comms[gid]]
    if len(gids) == 0:
        raise ValueError(f"No communities contain event='{event}'.")

    rng = np.random.RandomState(seed)
    if max_comms is not None and len(gids) > max_comms:
        gids = list(rng.choice(gids, size=max_comms, replace=False))

    edges = np.linspace(0.0, 1.0, bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])

    H_list = []
    kept = 0

    for gid in gids:
        out = comms[gid][event]
        total_logit = out["total"]      # [T,N] logits
        elig = out["eligible"]          # [T,N] bool

        T_use = total_logit.shape[0]
        tt0 = max(0, min(t0, T_use))
        tt1 = max(0, min(t1, T_use))
        if tt1 <= tt0:
            continue

        mask = elig[tt0:tt1].astype(bool)
        if mask.sum() < min_points_per_comm:
            continue

        samples = total_logit[tt0:tt1][mask]  # 1D
        if samples.size < min_points_per_comm:
            continue

        hist, _ = np.histogram(samples, bins=edges, density=True)
        H_list.append(hist)
        kept += 1

    if kept == 0:
        raise ValueError(f"No communities had enough eligible points for distribution (event={event}, window={window}).")

    H = np.stack(H_list, axis=0)  # [C, bins]
    mean_hist = H.mean(axis=0)

    plt.figure(figsize=(8, 5))
    for i in range(H.shape[0]):
        plt.plot(mids, H[i], alpha=alpha_faded, linewidth=lw_faded)

    plt.plot(mids, mean_hist, linewidth=lw_mean)
    plt.xlabel("total_logit")
    plt.ylabel("Density")
    plt.title(f"Overlayed community distributions of logit \n(event={event}, window={window}, communities={kept})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_npz", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="viz_from_processed")
    ap.add_argument("--window", type=str, default="test", choices=["train", "test", "full"])
    ap.add_argument("--bins", type=int, default=50)
    ap.add_argument("--max_comms", type=int, default=None)  # useful if you have hundreds/thousands
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_points_per_comm", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    processed = _load_processed(args.processed_npz)

    for event in ["sell", "repair"]:
        out1 = os.path.join(args.out_dir, f"mean_nei_timeseries_{event}_{args.window}.png")
        out2 = os.path.join(args.out_dir, f"overlay_dist_sigmoid_total_{event}_{args.window}.png")

        plot_mean_nei_timeseries(
            processed, event=event, window=args.window, out_path=out1,
            max_comms=args.max_comms, seed=args.seed
        )
        print(f"[OK] Saved: {out1}")

        plot_overlay_distributions_total_prob(
            processed, event=event, window=args.window, out_path=out2,
            bins=args.bins, max_comms=args.max_comms, seed=args.seed,
            min_points_per_comm=args.min_points_per_comm
        )
        print(f"[OK] Saved: {out2}")


if __name__ == "__main__":
    main()
