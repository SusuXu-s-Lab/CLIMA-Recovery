from __future__ import annotations

import argparse
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_eval_multi_real import (
    filter_communities_with_events,
    select_communities_by_distance,
    evaluate_tail_metrics,
)


# -------------------------------------------------------
# Xu-style multivariate Hawkes (exponential kernel)
# -------------------------------------------------------

class XuHawkesTorch(nn.Module):
    """
    Continuous-time multivariate Hawkes with exponential kernel:

        lambda_d(t) = mu_d
                      + sum_j alpha_{d,j} sum_{m: mark_m = j, t_m < t}
                            exp(-beta (t - t_m))

    D = N_nodes * K_types
    """

    def __init__(self, D: int, beta: float = 1.0):
        super().__init__()
        self.D = D
        self.beta = float(beta)
        # log-params => enforce positivity with softplus
        self.log_mu = nn.Parameter(torch.full((D,), -3.0))
        self.log_alpha = nn.Parameter(torch.full((D, D), -4.0))

    def intensity_loglik(
        self,
        t_events: torch.Tensor,
        marks: torch.Tensor,
        T_max: float,
    ) -> torch.Tensor:
        """
        Hawkes log-likelihood for a single sequence.
        t_events: [M], marks: [M] in {0,...,D-1}
        """
        device = t_events.device
        mu = F.softplus(self.log_mu) + 1e-6       # [D]
        alpha = F.softplus(self.log_alpha)        # [D, D]
        beta = self.beta
        M = t_events.shape[0]

        if M == 0:
            # no events -> only integral term
            return -(mu * T_max).sum()

        S = torch.zeros(mu.shape[0], device=device)  # kernel state
        loglik = torch.zeros((), device=device)
        prev_t = t_events[0]

        for n in range(M):
            t = t_events[n]
            if n > 0:
                dt = t - prev_t
                S = torch.exp(-beta * dt) * S

            d = marks[n].item()
            lam_d = mu[d] + torch.dot(alpha[d], S)
            loglik = loglik + torch.log(lam_d + 1e-8)

            S = S.clone()
            S[d] = S[d] + 1.0
            prev_t = t

        # Integrated intensity ∫ lambda_d(s) ds from 0..T_max
        T_max_t = torch.tensor(float(T_max), device=device)
        one_minus_exp = 1.0 - torch.exp(-beta * (T_max_t - t_events))  # [M]
        contrib = torch.zeros(mu.shape[0], device=device)
        contrib.scatter_add_(0, marks, one_minus_exp)                  # [D]
        integral = mu * T_max_t + alpha @ (contrib / beta)             # [D]
        loglik = loglik - integral.sum()

        return loglik

    def l1_penalty(self) -> torch.Tensor:
        alpha = F.softplus(self.log_alpha)
        return alpha.abs().sum()

    @torch.no_grad()
    def get_alpha_mu(self) -> Tuple[np.ndarray, np.ndarray]:
        alpha = F.softplus(self.log_alpha).cpu().numpy()
        mu = F.softplus(self.log_mu).cpu().numpy()
        return alpha, mu


# -------------------------------------------------------
# Discrete Y[t,i,k] <-> continuous events
# -------------------------------------------------------

def build_events_from_Y(
    Y: np.ndarray,
    jitter: bool = False,
    seed: int = 0,
):
    """
    Convert discrete-time indicators Y[t,i,k] to continuous events.

    For each unit in Y[t,i,k], create an event at time:
        t_event = t + 0.5  (or t + U[0,1) if jitter=True)

    Flatten node i and type k into one Hawkes dimension:
        d = i*K + k
    """
    T, N, K = Y.shape
    rng = np.random.RandomState(seed) if jitter else None

    times = []
    marks = []
    for t in range(T):
        for i in range(N):
            for k in range(K):
                cnt = int(Y[t, i, k])
                if cnt <= 0:
                    continue
                for _ in range(cnt):
                    tau = t + (rng.rand() if jitter else 0.5)
                    d = i * K + k
                    times.append(float(tau))
                    marks.append(d)

    if not times:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    t_events = np.asarray(times, dtype=np.float32)
    marks_arr = np.asarray(marks, dtype=np.int64)
    return t_events, marks_arr


def alpha_to_node_adjacency(alpha: np.ndarray, N: int, K: int) -> np.ndarray:
    """
    Convert alpha[D,D] (D=N*K) to node-level adjacency A[k, i, j]:

        A[k, i, j] = sum_{source_type l} alpha[u, v]
    where
        u = i*K + k (target dimension),
        v runs over {j*K + l : l=0..K-1}.
    """
    D = alpha.shape[0]
    assert alpha.shape == (D, D)
    assert D == N * K

    A = np.zeros((K, N, N), dtype=np.float32)
    for k_tgt in range(K):
        for i in range(N):
            u = i * K + k_tgt
            for j in range(N):
                v_start = j * K
                v_end = (j + 1) * K
                A[k_tgt, i, j] = alpha[u, v_start:v_end].sum()
    return A


# -------------------------------------------------------
# Train Hawkes per community
# -------------------------------------------------------

def train_hawkes_for_community(
    Y: np.ndarray,
    T_train: int,
    beta: float,
    lr: float,
    n_epochs: int,
    lambda_l1: float,
    device: torch.device,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit Hawkes to one community using events in [0, ..., T_train-1].

    Y: [T_full, N, K]
    Returns:
      alpha_hat: [D, D]
      mu_hat   : [D]
    """
    T_full, N, K = Y.shape
    if T_train <= 0 or T_train > T_full:
        T_train = T_full

    Y_train = Y[:T_train]

    t_np, marks_np = build_events_from_Y(Y_train)
    D = N * K

    t_events = torch.from_numpy(t_np).float().to(device)
    marks = torch.from_numpy(marks_np).long().to(device)

    model = XuHawkesTorch(D=D, beta=beta).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    T_max = float(T_train)

    for epoch in range(n_epochs):
        opt.zero_grad()
        loglik = model.intensity_loglik(t_events, marks, T_max)
        loss = -loglik + lambda_l1 * model.l1_penalty()
        loss.backward()
        opt.step()

        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f"    epoch {epoch:03d}  neg-loglik+L1 = {float(loss):.4f}")

    alpha_hat, mu_hat = model.get_alpha_mu()
    return alpha_hat, mu_hat


# -------------------------------------------------------
# Discrete-time intensity grid -> logits for prediction
# -------------------------------------------------------

def compute_lambda_grid(
    mu: np.ndarray,
    alpha: np.ndarray,
    beta: float,
    Y_train: np.ndarray,
    T_full: int,
) -> np.ndarray:
    """
    Given Hawkes parameters (mu, alpha, beta) and training events Y_train
    (shape [T_train, N, K]), compute a discrete-time grid:

        lambda_grid[t, d]  for t=0..T_full-1, d=0..D-1

    using only past events up to T_train (no peeking into eval period).
    """
    T_train, N, K = Y_train.shape
    D = N * K

    mu = mu.reshape(D)
    alpha = alpha.reshape(D, D)

    t_events, marks = build_events_from_Y(Y_train)
    M = len(t_events)

    lam_grid = np.zeros((T_full, D), dtype=np.float32)

    if M == 0:
        lam_grid[:] = mu[None, :]
        return lam_grid

    beta = float(beta)
    S = np.zeros(D, dtype=np.float64)
    current_time = 0.0
    idx = 0

    for t in range(T_full):
        # propagate kernel state
        dt = t - current_time
        if dt > 0:
            S *= np.exp(-beta * dt)
            current_time = float(t)

        # intensity at bin start
        lam_grid[t, :] = mu + alpha @ S

        # process any training events in [t, t+1)
        t_next = t + 1.0
        while idx < M and t_events[idx] < t_next:
            dt2 = float(t_events[idx] - current_time)
            if dt2 > 0:
                S *= np.exp(-beta * dt2)
                current_time = float(t_events[idx])
            S[marks[idx]] += 1.0
            idx += 1

    return lam_grid


def lambda_to_logits(
    lam_grid: np.ndarray,
    horizon: int,
    N: int,
    K: int,
):
    """
    Convert lambda_grid[t, d] to:

      - logits_exact[t, i, k]  (prob of event in [t, t+1))
      - logits_window[t, i, k] (prob of event in [t, ..., t+H-1])

    via Poisson approx:
      p_exact  = 1 - exp(-lambda[t] * 1)
      p_window = 1 - exp(-sum_{tau=t}^{t+H-1} lambda[tau])
    """
    T_full, D = lam_grid.shape
    assert D == N * K

    lam_clipped = np.clip(lam_grid, 0.0, 50.0)

    # exact per-bin probability
    p_exact = 1.0 - np.exp(-lam_clipped)            # [T_full, D]

    # window probability
    p_window = np.zeros_like(p_exact)
    for t in range(T_full):
        t_end = min(T_full, t + horizon)
        lam_sum = lam_clipped[t:t_end, :].sum(axis=0)
        p_window[t, :] = 1.0 - np.exp(-lam_sum)

    p_exact = np.clip(p_exact, 1e-8, 1.0 - 1e-8)
    p_window = np.clip(p_window, 1e-8, 1.0 - 1e-8)

    logits_exact_flat = np.log(p_exact) - np.log(1.0 - p_exact)
    logits_window_flat = np.log(p_window) - np.log(1.0 - p_window)

    logits_exact = np.zeros((T_full, N, K), dtype=np.float32)
    logits_window = np.zeros((T_full, N, K), dtype=np.float32)

    for d in range(D):
        i = d // K
        k = d % K
        logits_exact[:, i, k] = logits_exact_flat[:, d]
        logits_window[:, i, k] = logits_window_flat[:, d]

    return logits_exact, logits_window


# -------------------------------------------------------
# Main driver: REAL data, same inputs as Graph Hawkes
# -------------------------------------------------------

def run_xu_hawkes_real_and_eval(
    real_npz: str,
    graphs_out: str,
    ref_lon: float,
    ref_lat: float,
    max_communities: int,
    min_nodes_real: int,
    max_nodes_real: int,
    beta: float,
    lr: float,
    n_epochs: int,
    lambda_l1: float,
    device_str: str,
    horizon_months: int,
    verbose: bool = True,
):
    """
    REAL-data Xu-style Hawkes baseline:
      * loads lee_ian_by_cbg_*.npz
      * selects communities with select_communities_by_distance
      * fits Hawkes per community
      * exports learned graphs in same format as export_learned_graphs
      * evaluates future state prediction via evaluate_tail_metrics
    """
    device = torch.device(device_str)

    print(f"[INFO] Loading real data from {real_npz}")
    data = np.load(real_npz, allow_pickle=True)
    communities_np = data["communities"]
    communities_all = list(communities_np)

    meta = data["meta"]
    if not isinstance(meta, dict):
        meta = meta.item()
    T_train_meta = int(meta["T_train"])

    # Select subset by distance to reference point
    # First: spatial / size filtering
    spatial_indices = select_communities_by_distance(
        communities_all,
        ref_lon,
        ref_lat,
        max_communities=10_000,  # large number, we’ll cap later
        min_nodes=min_nodes_real,
        max_nodes=max_nodes_real,
    )

    # Second: event-based filtering (TRAIN-ONLY, no leakage)
    event_indices = filter_communities_with_events(
        [communities_all[i] for i in spatial_indices],
        T_train=T_train_meta,
        min_sell=1,        # at least one sell in training
        min_repair=1,      # or tune this
        mode="test",
    )

    # Map back to original indices
    selected_indices = [spatial_indices[i] for i in event_indices]

    # Finally cap the number of communities
    selected_indices = selected_indices[:max_communities]


    print(f"#communities after spatial filter = {len(spatial_indices)}")
    print(f"#communities after event filter   = {len(selected_indices)}")


    communities = [communities_all[i] for i in selected_indices]
    print(f"[INFO] Selected {len(communities)} communities for Xu-Hawkes baseline.")

    graphs = []
    logits_dict: Dict[int, np.ndarray] = {}
    window_logits_dict: Dict[int, np.ndarray] = {}

    for local_idx, global_idx in enumerate(selected_indices):
        comm = communities_all[global_idx]
        Y_g = np.asarray(comm["Y"], dtype=np.float32)  # [T_g, N_g, K_g]
        T_g, N_g, K_g = Y_g.shape
        T_train = min(T_g, T_train_meta)

        if verbose:
            print(f"\n[COMM {local_idx} (global {global_idx})] "
                  f"N={N_g}, K={K_g}, T={T_g}, T_train={T_train}")

        # 1) Fit Hawkes on training horizon
        alpha_hat, mu_hat = train_hawkes_for_community(
            Y=Y_g,
            T_train=T_train,
            beta=beta,
            lr=lr,
            n_epochs=n_epochs,
            lambda_l1=lambda_l1,
            device=device,
            verbose=verbose,
        )

        # 2) Node-level adjacency A[k, i, j] for graph reconstruction / interpretability
        A_stack = alpha_to_node_adjacency(alpha_hat, N=N_g, K=K_g)  # [K_g, N_g, N_g]

        graph_entry = {
            "community_global_index": global_idx,
            "node_ids": comm["node_ids"],
            "coords": comm["coords"],
            "A": A_stack,
        }
        if "cbg" in comm:
            graph_entry["cbg"] = comm["cbg"]
        graphs.append(graph_entry)

        # 3) Discrete-time intensities and logits for forecasting
        lam_grid = compute_lambda_grid(
            mu=mu_hat,
            alpha=alpha_hat,
            beta=beta,
            Y_train=Y_g[:T_train],
            T_full=T_g,
        )
        logits_exact, logits_window = lambda_to_logits(
            lam_grid,
            horizon=horizon_months,
            N=N_g,
            K=K_g,
        )
        logits_dict[local_idx] = logits_exact
        window_logits_dict[local_idx] = logits_window

    # Save graphs exactly like export_learned_graphs
    graphs_arr = np.array(graphs, dtype=object)
    np.savez_compressed(graphs_out, graphs=graphs_arr)
    print(f"\n[INFO] Saved Xu-Hawkes learned graphs to {graphs_out}")

    # Future state prediction metrics (same interface as your method)
    metrics = evaluate_tail_metrics(
        communities=communities,
        logits_dict=logits_dict,
        window_logits_dict=window_logits_dict,
        T_train=T_train_meta,
        horizon=horizon_months,
    )

    return metrics


# -------------------------------------------------------
# CLI
# -------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Xu-style Hawkes baseline on REAL Ian-by-CBG data."
    )
    p.add_argument("--real_npz", type=str, required=True,
                   help="Path to real Ian-by-CBG NPZ "
                        "(e.g. lee_ian_by_cbg_2state_v2_dr0.005_tr0.6.npz)")
    p.add_argument("--graphs_out", type=str, required=True,
                   help="Output NPZ path for learned graphs.")
    p.add_argument("--ref_lon", type=float, required=True,
                   help="Reference longitude for community selection.")
    p.add_argument("--ref_lat", type=float, required=True,
                   help="Reference latitude for community selection.")
    p.add_argument("--max_communities", type=int, default=200,
                   help="Max number of communities to use.")
    p.add_argument("--min_nodes_real", type=int, default=100,
                   help="Min nodes per community.")
    p.add_argument("--max_nodes_real", type=int, default=900,
                   help="Max nodes per community.")
    p.add_argument("--beta", type=float, default=1.0,
                   help="Exponential kernel decay rate.")
    p.add_argument("--lr", type=float, default=5e-2,
                   help="Learning rate for Adam.")
    p.add_argument("--n_epochs", type=int, default=100,
                   help="Training epochs per community.")
    p.add_argument("--lambda_l1", type=float, default=1e-3,
                   help="L1 penalty weight on alpha.")
    p.add_argument("--device", type=str, default="cpu",
                   help="'cpu' or 'cuda'.")
    p.add_argument("--horizon_months", type=int, default=6,
                   help="Future window horizon H for window labels.")
    p.add_argument("--metrics_out", type=str,
                   default="xu_hawkes_real_metrics.npz",
                   help="Where to save metrics npz.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-community progress.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; falling back to CPU.")
        device_str = "cpu"
    else:
        device_str = args.device

    metrics, debug_dump = run_xu_hawkes_real_and_eval(
        real_npz=args.real_npz,
        graphs_out=args.graphs_out,
        ref_lon=args.ref_lon,
        ref_lat=args.ref_lat,
        max_communities=args.max_communities,
        min_nodes_real=args.min_nodes_real,
        max_nodes_real=args.max_nodes_real,
        beta=args.beta,
        lr=args.lr,
        n_epochs=args.n_epochs,
        lambda_l1=args.lambda_l1,
        device_str=device_str,
        horizon_months=args.horizon_months,
        verbose=not args.quiet,
    )

    # Save metrics in the same style you used for other baselines
    np.savez_compressed(args.metrics_out, metrics=metrics)
    print(f"\n[INFO] Saved Xu-Hawkes REAL baseline metrics to {args.metrics_out}")

    print("\n[Xu-Hawkes REAL baseline metrics]")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
