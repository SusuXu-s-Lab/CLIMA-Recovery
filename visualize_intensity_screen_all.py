import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_multi_real import MultiScaleCoupledHawkesReal


# -----------------------------
# Loading
# -----------------------------
def load_hdd_model_from_ckpt(ckpt_path: str, npz_path: str, device: str = "cpu"):
    """
    Load HDD model from checkpoint. Construct a dummy model with correct shapes,
    then load state_dict. Buffers like mu/sigma will be overwritten by state_dict.
    """
    data = np.load(npz_path, allow_pickle=True)
    communities = list(data["communities"])
    meta = data["meta"].item()

    d_in = communities[0]["X"].shape[1]
    K = communities[0]["Y"].shape[2]

    dummy_mu = torch.zeros(d_in, dtype=torch.float32)
    dummy_sigma = torch.ones(d_in, dtype=torch.float32)

    model = MultiScaleCoupledHawkesReal(
        d_in=d_in,
        K=K,
        mu=dummy_mu.to(device),
        sigma=dummy_sigma.to(device),
        d_hid=128,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and ("model_state" in ckpt):
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and ("state_dict" in ckpt):
        state = ckpt["state_dict"]
    else:
        state = ckpt

    model.load_state_dict(state, strict=False)
    model.eval()
    return model, meta


# -----------------------------
# Core computation
# -----------------------------
def compute_components_over_time(
    model,
    comm: dict,
    gamma_vec: torch.Tensor,
    horizon_T: int = None,
    event_type: int = 0,
    device: str = "cpu",
    eligibility_mode: str = "sell_not_sold",
):
    """
    Compute decomposition consistent with step_intensity for ONE community.

    IMPORTANT (your requirement 1, updated):
      self := baseline + self_raw
      where self_raw = diag_term + cross_self
      NO CLAMP.

    Returns dict with:
      total:    [T, N]
      baseline: [T, N]
      self_raw: [T, N]  (diag + cross_self)
      self:     [T, N]  (baseline + self_raw)   <-- merged baseline into self
      nei:      [T, N]
      eligible: [T, N] bool mask (sell: not sold yet)
      Y:        [T, N] labels for event_type
    """
    X = torch.from_numpy(comm["X"]).float().to(device)             # [N, d]
    coords = torch.from_numpy(comm["coords"]).float().to(device)   # [N, 2]
    Y = torch.from_numpy(comm["Y"]).float().to(device)             # [T, N, K]

    T, N, K = Y.shape
    T_use = min(T, int(horizon_T)) if horizon_T is not None else T

    # Build structures once (matches training)
    H_g, w_self_node, A_list, baseline_node = model.build_structures(X, coords)
    w_self_node = torch.clamp(w_self_node, min=0.0)

    # histories
    R_self = torch.zeros(N, K, dtype=torch.float32, device=device)

    total = torch.zeros(T_use, N, dtype=torch.float32, device=device)
    base = torch.zeros(T_use, N, dtype=torch.float32, device=device)
    self_raw = torch.zeros(T_use, N, dtype=torch.float32, device=device)
    self_merged = torch.zeros(T_use, N, dtype=torch.float32, device=device)
    nei_term = torch.zeros(T_use, N, dtype=torch.float32, device=device)

    eligible = torch.ones(T_use, N, dtype=torch.bool, device=device)
    sold_cum = torch.zeros(N, dtype=torch.float32, device=device)

    for t in range(T_use):
        if t > 0:
            # sold status update uses SELL channel (k=0)
            sold_cum = torch.clamp(sold_cum + Y[t - 1, :, 0], 0.0, 1.0)
            # self history updates using all K channels
            R_self = gamma_vec.view(1, K) * R_self + Y[t - 1]

        # neighbor history
        R_nei_cols = []
        for k in range(K):
            Rk = R_self[:, k:k+1]        # [N, 1]
            R_nei_cols.append(A_list[k] @ Rk)
        R_nei = torch.cat(R_nei_cols, dim=1)  # [N, K]

        # official total logits
        logits = model.step_intensity(R_self, R_nei, w_self_node, baseline_node)  # [N, K]

        # components for this event_type
        baseline_k = model.b[event_type].view(1) + baseline_node[:, event_type]  # [N]
        diag_term_k = w_self_node[:, event_type] * R_self[:, event_type]         # [N]
        cross_self_k = (R_self @ model.W_self.T)[:, event_type]                  # [N]
        cross_nei_k = (R_nei @ model.W_nei.T)[:, event_type]                     # [N]
        cross_nei_k = cross_nei_k * model.alpha_nei[event_type].view(1)

        self_raw_k = diag_term_k + cross_self_k
        self_k = baseline_k + self_raw_k  # merged baseline into self (NO CLAMP)

        total[t] = logits[:, event_type]
        base[t] = baseline_k
        self_raw[t] = self_raw_k
        self_merged[t] = self_k
        nei_term[t] = cross_nei_k

        # eligibility mask
        if eligibility_mode == "sell_not_sold" and event_type == 0:
            eligible[t] = (sold_cum < 0.5)
        else:
            eligible[t] = torch.ones(N, dtype=torch.bool, device=device)

    return {
        "total": total.detach().cpu().numpy(),
        "baseline": base.detach().cpu().numpy(),
        "self_raw": self_raw.detach().cpu().numpy(),
        "self": self_merged.detach().cpu().numpy(),
        "nei": nei_term.detach().cpu().numpy(),
        "eligible": eligible.detach().cpu().numpy(),
        "Y": Y[:T_use, :, event_type].detach().cpu().numpy(),
    }


# -----------------------------
# Plotting utilities
# -----------------------------
def _sample_nodes(eligible_TN: np.ndarray, max_nodes: int, rng: np.random.RandomState):
    """Sample nodes that are eligible at least once."""
    if max_nodes is None:
        max_nodes = 0
    ever_ok = eligible_TN.any(axis=0)
    idx = np.where(ever_ok)[0]
    if idx.size == 0:
        return idx
    if max_nodes > 0 and idx.size > max_nodes:
        idx = rng.choice(idx, size=max_nodes, replace=False)
    return idx


def _accumulate_global_mean(sum_t: np.ndarray, cnt_t: np.ndarray, values_TN: np.ndarray, eligible_TN: np.ndarray):
    """
    Accumulate sum and counts across ALL eligible nodes (vectorized) for global mean line.
    """
    # values_TN: [T, N], eligible_TN: [T, N]
    mask = eligible_TN.astype(bool)
    # sum over nodes per t
    sum_t += (values_TN * mask).sum(axis=1)
    cnt_t += mask.sum(axis=1)


def _finalize_mean_line(sum_t: np.ndarray, cnt_t: np.ndarray):
    mean_t = np.full_like(sum_t, np.nan, dtype=np.float64)
    ok = cnt_t > 0
    mean_t[ok] = sum_t[ok] / cnt_t[ok]
    return mean_t


def setup_fig(title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(title)
    ax.set_xlabel("Time (months)")
    ax.set_ylabel(ylabel)
    return fig, ax


def save_fig(fig, path: str):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# -----------------------------
# Scatter helper (weighted mean over eligible points)
# -----------------------------
def mean_over_window_weighted(values_TN: np.ndarray, eligible_TN: np.ndarray, t0: int, t1: int):
    """
    Weighted mean over eligible points in [t0,t1):
      numerator = sum_{t,i} values[t,i] where eligible
      denom     = count eligible points
    """
    T, N = values_TN.shape
    t0 = max(0, int(t0))
    t1 = min(T, int(t1))
    if t1 <= t0:
        return np.nan

    vals = values_TN[t0:t1]
    elig = eligible_TN[t0:t1].astype(bool)
    denom = elig.sum()
    if denom == 0:
        return np.nan
    num = (vals * elig).sum()
    return float(num / denom)


# -----------------------------
# Main
# -----------------------------
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device

    model, meta = load_hdd_model_from_ckpt(args.ckpt_path, args.npz_path, device=device)

    data = np.load(args.npz_path, allow_pickle=True)
    communities = list(data["communities"])
    T_train = int(meta.get("T_train", 0))
    T_meta = int(meta.get("T", len(communities[0]["Y"])))

    # gamma vector
    gamma_vals = [float(x) for x in args.gamma.split(",")]
    gamma_vec = torch.tensor(gamma_vals, dtype=torch.float32, device=device)

    # scatter window (absolute indices)
    if args.scatter_window == "test":
        t0, t1 = T_train, T_meta
    elif args.scatter_window == "train":
        t0, t1 = 0, T_train
    else:
        t0, t1 = 0, T_meta

    # global plotting horizon
    T_global = min(T_meta, args.horizon_T) if args.horizon_T is not None else T_meta
    ts_global = np.arange(T_global)

    rng = np.random.RandomState(args.seed)

    # Requirement 2: save all processed data in one file
    processed = {
        "meta": meta,
        "T_train": T_train,
        "T_meta": T_meta,
        "T_global": T_global,
        "scatter_window": (t0, t1),
        "gamma": gamma_vec.detach().cpu().numpy(),
        "communities": {},  # gid -> {"sell": out, "repair": out}
    }

    # Prepare 8 global node-line plots (Requirement 3)
    figs = {}
    axs = {}
    sum_cnt = {}  # for global mean lines: sum_cnt[(event,comp)] = (sum_t, cnt_t)

    for event_name in ["sell", "repair"]:
        for comp, ylabel in [
            ("total", "logit"),
            ("baseline", "logit contribution"),
            ("self", "baseline + self-excitation (logit contribution)"),
            ("nei", "neighbor-excitation (logit contribution)"),
        ]:
            key = (event_name, comp)
            fig, ax = setup_fig(
                title=f"GLOBAL node trajectories: {event_name.upper()} {comp}",
                ylabel=ylabel,
            )
            figs[key] = fig
            axs[key] = ax
            sum_t = np.zeros(T_global, dtype=np.float64)
            cnt_t = np.zeros(T_global, dtype=np.float64)
            sum_cnt[key] = (sum_t, cnt_t)

    # scatter storage
    scatter_self_sell = []
    scatter_nei_sell = []
    scatter_ids_sell = []

    scatter_self_repair = []
    scatter_nei_repair = []
    scatter_ids_repair = []

    kept = 0
    for gid, comm in enumerate(communities):
        Yg = comm["Y"]
        T, Ng, K = Yg.shape

        if Ng < args.min_nodes:
            continue

        # filter by events in window (rough)
        win = slice(max(0, t0), min(T, t1))
        sell_cnt = int(Yg[win, :, 0].sum())
        rep_cnt = int(Yg[win, :, 1].sum())
        if (sell_cnt < args.min_sell_events) and (rep_cnt < args.min_repair_events):
            continue

        processed["communities"][gid] = {}

        for k, event_name in [(0, "sell"), (1, "repair")]:
            out = compute_components_over_time(
                model=model,
                comm=comm,
                gamma_vec=gamma_vec,
                horizon_T=args.horizon_T,
                event_type=k,
                device=device,
                eligibility_mode="sell_not_sold",
            )

            elig = out["eligible"]
            if elig.sum() < args.min_eligible_points:
                continue

            # truncate to global horizon
            T_use = min(out["total"].shape[0], T_global)
            for kk in out.keys():
                if isinstance(out[kk], np.ndarray) and out[kk].ndim == 2 and out[kk].shape[0] >= T_use:
                    out[kk] = out[kk][:T_use]

            # store (Req 2)
            processed["communities"][gid][event_name] = out

            # Global plots (Req 3): plot node-lines sampled per community
            idx_nodes = _sample_nodes(out["eligible"], args.nodes_per_comm_plot, rng)

            for comp in ["total", "baseline", "self", "nei"]:
                key = (event_name, comp)
                ax = axs[key]

                vals = out[comp]        # [T_use, N]
                el = out["eligible"]    # [T_use, N]

                # accumulate global mean over ALL eligible nodes (not just sampled)
                sum_t, cnt_t = sum_cnt[key]
                _accumulate_global_mean(sum_t[:T_use], cnt_t[:T_use], vals, el)

                # plot sampled node lines
                for i in idx_nodes:
                    y = vals[:, i].copy()
                    e = el[:, i].astype(bool)
                    y[~e] = np.nan
                    ax.plot(np.arange(T_use), y, alpha=args.global_alpha, linewidth=args.global_lw)

            # Scatter: community-level mean self vs nei over chosen window
            i# Scatter: community-level mean self vs nei over chosen window (do BOTH)
            t0_use = max(0, min(t0, T_use))
            t1_use = max(0, min(t1, T_use))

            mean_self = mean_over_window_weighted(out["self"], out["eligible"], t0_use, t1_use)
            mean_nei  = mean_over_window_weighted(out["nei"],  out["eligible"], t0_use, t1_use)

            if np.isfinite(mean_self) and np.isfinite(mean_nei):
                if event_name == "sell":
                    scatter_self_sell.append(mean_self)
                    scatter_nei_sell.append(mean_nei)
                    scatter_ids_sell.append(gid)
                elif event_name == "repair":
                    scatter_self_repair.append(mean_self)
                    scatter_nei_repair.append(mean_nei)
                    scatter_ids_repair.append(gid)


        # if community has neither event, drop it from processed
        if len(processed["communities"][gid]) == 0:
            processed["communities"].pop(gid, None)
            continue

        kept += 1
        if args.max_communities is not None and kept >= args.max_communities:
            break

    # Overlay global mean lines and save global plots
    for (event_name, comp), (sum_t, cnt_t) in sum_cnt.items():
        ax = axs[(event_name, comp)]
        mean_t = _finalize_mean_line(sum_t, cnt_t)
        ax.plot(ts_global, mean_t, linewidth=2.8)  # solid mean line

        if args.ylim_lo is not None and args.ylim_hi is not None:
            ax.set_ylim(args.ylim_lo, args.ylim_hi)

        out_png = os.path.join(args.out_dir, f"GLOBAL_nodes_{event_name}_{comp}.png")
        save_fig(figs[(event_name, comp)], out_png)
        print(f"[INFO] Saved global plot: {out_png}")

    # Scatter plot
    # Scatter plot: SELL
    if len(scatter_self_sell) > 0:
        plt.figure(figsize=(6, 5))
        plt.scatter(scatter_self_sell, scatter_nei_sell, alpha=0.7)
        plt.xlabel("Mean self (baseline + self_raw) [sell]")
        plt.ylabel("Mean neighbor-excitation [sell]")
        plt.title(f"Community-level excitation scatter (SELL, {args.scatter_window} window)")
        plt.tight_layout()
        out_scatter = os.path.join(args.out_dir, f"scatter_self_vs_nei_sell_{args.scatter_window}.png")
        plt.savefig(out_scatter, dpi=200)
        plt.close()
        print(f"[INFO] Saved scatter: {out_scatter}")
    else:
        print("[WARN] No valid communities for SELL scatter plot.")

    # Scatter plot: REPAIR
    if len(scatter_self_repair) > 0:
        plt.figure(figsize=(6, 5))
        plt.scatter(scatter_self_repair, scatter_nei_repair, alpha=0.7)
        plt.xlabel("Mean self (baseline + self_raw) [repair]")
        plt.ylabel("Mean neighbor-excitation [repair]")
        plt.title(f"Community-level excitation scatter (REPAIR, {args.scatter_window} window)")
        plt.tight_layout()
        out_scatter = os.path.join(args.out_dir, f"scatter_self_vs_nei_repair_{args.scatter_window}.png")
        plt.savefig(out_scatter, dpi=200)
        plt.close()
        print(f"[INFO] Saved scatter: {out_scatter}")
    else:
        print("[WARN] No valid communities for REPAIR scatter plot.")


    # Save processed data (Req 2)
    out_npz = os.path.join(args.out_dir, args.processed_out)
    np.savez_compressed(out_npz, processed=processed)
    print(f"[INFO] Saved processed data: {out_npz}")

    print(f"[DONE] Outputs saved under: {args.out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npz_path", type=str, required=True)
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="intensity_viz_all")

    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--horizon_T", type=int, default=None)

    # gamma vector (comma-separated)
    p.add_argument("--gamma", type=str, default="0.90,0.80")

    # community filtering
    p.add_argument("--min_nodes", type=int, default=100)
    p.add_argument("--min_sell_events", type=int, default=1)
    p.add_argument("--min_repair_events", type=int, default=1)
    p.add_argument("--min_eligible_points", type=int, default=500)
    p.add_argument("--max_communities", type=int, default=None)

    # global plotting (Req 3)
    p.add_argument("--nodes_per_comm_plot", type=int, default=50)   # sample node-lines per community
    p.add_argument("--global_alpha", type=float, default=0.05)
    p.add_argument("--global_lw", type=float, default=0.9)
    p.add_argument("--ylim_lo", type=float, default=None)
    p.add_argument("--ylim_hi", type=float, default=None)

    # scatter controls
    p.add_argument("--scatter_event", type=str, default="sell", choices=["sell", "repair"])
    p.add_argument("--scatter_window", type=str, default="test", choices=["train", "test", "full"])

    # processed output (Req 2)
    p.add_argument("--processed_out", type=str, default="processed_all_communities.npz")

    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    main(args)
