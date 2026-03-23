"""
baseline_xu_hawkes_sim.py

Classic multivariate Hawkes with exponential kernels + L1 sparsity
(Xu et al. 2016 style), trained community-by-community on the
simulated Ian-by-CBG data.

Outputs:
  - Graphs NPZ compatible with eval_graph_recovery.py
  - (Optionally) future state prediction metrics via evaluate_tail_metrics
    using the same [T, N_g, K] logits format as other baselines.
"""

from __future__ import annotations

import argparse
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# For forecasting metrics (same as your Graph Hawkes)
from train_eval_multi_real import evaluate_tail_metrics


# --------------------------------------------------------------------
# Xu-style Hawkes model (exponential kernels + L1 sparsity)
# --------------------------------------------------------------------


class XuHawkesTorch(nn.Module):
    """
    Continuous-time multivariate Hawkes with exponential kernel:

        lambda_d(t) = mu_d
                      + sum_j alpha_{d,j} sum_{m: mark_m = j, t_m < t}
                            exp(-beta (t - t_m))

    Parameters:
        D    : number of dimensions (= N_g * K)
        beta : fixed decay rate of exponential kernel
    """

    def __init__(self, D: int, beta: float = 1.0):
        super().__init__()
        self.D = D
        self.beta = float(beta)
        # Log-parameters so we can enforce positivity with softplus
        self.log_mu = nn.Parameter(torch.full((D,), -3.0))
        self.log_alpha = nn.Parameter(torch.full((D, D), -4.0))

    def intensity_loglik(
        self,
        t_events: torch.Tensor,
        marks: torch.Tensor,
        T_max: float,
    ) -> torch.Tensor:
        """
        Hawkes log-likelihood for a single community sequence.

        Args
        ----
        t_events : [M] sorted event times
        marks    : [M] event dimensions in {0,...,D-1}
        T_max    : observation horizon
        """
        device = t_events.device
        mu = F.softplus(self.log_mu) + 1e-6      # [D]
        alpha = F.softplus(self.log_alpha)       # [D, D]
        D = self.D
        beta = self.beta
        M = t_events.shape[0]

        if M == 0:
            # No events: loglik is just integral term - sum_d mu_d T_max
            return -(mu * T_max).sum()

        S = torch.zeros(D, device=device)  # kernel state
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

        # Integrated intensity term
        T_max_t = torch.tensor(float(T_max), device=device)
        one_minus_exp = 1.0 - torch.exp(-beta * (T_max_t - t_events))  # [M]
        contrib = torch.zeros(D, device=device)
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


# --------------------------------------------------------------------
# Discrete Y[t,i,k] -> continuous events, and back
# --------------------------------------------------------------------


def build_events_from_Y(
    Y: np.ndarray,
    jitter: bool = False,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert discrete-time indicators/counts Y[t,i,k] to continuous events.

    We create one event for each unit in Y[t,i,k] at time:
        t_event = t + 0.5  (or t + U[0,1) if jitter=True)

    We flatten node i and type k into one Hawkes dimension:
        d = i*K + k  (0 <= d < D=N*K)

    Returns:
        t_events: [M] times
        marks   : [M] dims in {0,...,D-1}
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
                    if jitter:
                        tau = t + rng.rand()
                    else:
                        tau = t + 0.5
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
    Turn Hawkes alpha[D,D] (D=N*K) into node-level adjacency A[k,i,j]:

        A[k, i, j] = sum_{source_type l} alpha[u, v]
    where
        u = i*K + k  (target dimension),
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


# --------------------------------------------------------------------
# Training per community
# --------------------------------------------------------------------


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

    t_np, marks_np = build_events_from_Y(Y_train, jitter=False)
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


# --------------------------------------------------------------------
# Discrete intensity and probability grid for prediction
# --------------------------------------------------------------------


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

    where D = N*K, using only past events up to T_train (no peeking).
    """
    T_train, N, K = Y_train.shape
    D = N * K
    mu = mu.reshape(D)
    alpha = alpha.reshape(D, D)

    t_events, marks = build_events_from_Y(Y_train, jitter=False)
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
        # propagate kernel state from current_time to integer t
        dt = t - current_time
        if dt > 0:
            S *= np.exp(-beta * dt)
            current_time = float(t)

        # intensity at bin start
        lam_grid[t, :] = mu + alpha @ S

        # now process any events in [t, t+1) -- all training events have t< T_train
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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert lambda_grid[t, d] to:
      - logits_exact[t, i, k]  (prob event in [t, t+1))
      - logits_window[t, i, k] (prob event in [t, ..., t+H-1])

    using the Poisson approximation:
      p_exact = 1 - exp(-lambda * 1)
      p_window = 1 - exp(-sum_{tau=t}^{t+H-1} lambda[tau])

    lam_grid: [T_full, D] with D = N*K
    """
    T_full, D = lam_grid.shape
    assert D == N * K

    lam_clipped = np.clip(lam_grid, 0.0, 50.0)
    # exact per-bin probability
    p_exact = 1.0 - np.exp(-lam_clipped)           # [T_full, D]

    # window probability
    p_window = np.zeros_like(p_exact)
    for t in range(T_full):
        t_end = min(T_full, t + horizon)
        # sum lam over [t, t+H-1]
        lam_sum = lam_clipped[t:t_end, :].sum(axis=0)
        p_window[t, :] = 1.0 - np.exp(-lam_sum)

    # clamp to avoid inf logits
    p_exact = np.clip(p_exact, 1e-8, 1.0 - 1e-8)
    p_window = np.clip(p_window, 1e-8, 1.0 - 1e-8)

    logits_exact_flat = np.log(p_exact) - np.log(1.0 - p_exact)     # [T_full, D]
    logits_window_flat = np.log(p_window) - np.log(1.0 - p_window)  # [T_full, D]

    # reshape to [T, N, K]
    logits_exact = np.zeros((T_full, N, K), dtype=np.float32)
    logits_window = np.zeros((T_full, N, K), dtype=np.float32)

    for d in range(D):
        i = d // K
        k = d % K
        logits_exact[:, i, k] = logits_exact_flat[:, d]
        logits_window[:, i, k] = logits_window_flat[:, d]

    return logits_exact, logits_window


# --------------------------------------------------------------------
# Main driver: train per community, export graphs, build logits_dict
# --------------------------------------------------------------------


def run_xu_hawkes_sim_and_eval(
    sim_npz: str,
    graphs_out: str,
    max_communities: int = 200,
    min_nodes: int = 20,
    max_nodes: int = 1000,
    beta: float = 1.0,
    lr: float = 5e-2,
    n_epochs: int = 100,
    lambda_l1: float = 1e-3,
    device_str: str = "cpu",
    horizon_months: int = 6,
    verbose: bool = True,
):
    """
    Load simulated NPZ, fit Xu-style Hawkes per community, export learned
    graphs, and compute future-state prediction metrics via
    `evaluate_tail_metrics`.

    Returns:
        metrics: dict with AUC/AP (exact + window) per event type.
    """
    device = torch.device(device_str)

    print(f"[INFO] Loading simulated data from {sim_npz}")
    data = np.load(sim_npz, allow_pickle=True)
    communities_arr = data["communities"]
    communities = list(communities_arr)

    meta = data.get("meta", None)
    if meta is not None and not isinstance(meta, dict):
        meta = meta.item()

    if meta is None:
        # Fallback if meta missing
        print("[WARN] meta missing; inferring T_train from Y.")
        T_train_default = None
    else:
        T_train_default = int(meta.get("T_train", 0)) or None

    G = len(communities)
    print(f"[INFO] #communities in sim_npz = {G}")

    selected_idxs: List[int] = []
    for g_idx, comm in enumerate(communities):
        Y_g = comm["Y"]   # [T, N_g, K]
        T_g, N_g, K_g = Y_g.shape
        if N_g < min_nodes or N_g > max_nodes:
            continue
        selected_idxs.append(g_idx)
        if len(selected_idxs) >= max_communities:
            break

    if not selected_idxs:
        raise ValueError("No communities selected; adjust min_nodes/max_nodes.")

    print(f"[INFO] Selected {len(selected_idxs)} communities (global indices: {selected_idxs[:8]}...)")

    graphs = []
    logits_dict: Dict[int, np.ndarray] = {}
    window_logits_dict: Dict[int, np.ndarray] = {}

    for local_idx, g_idx in enumerate(selected_idxs):
        comm = communities[g_idx]
        Y_g = np.asarray(comm["Y"], dtype=np.float32)  # [T, N_g, K]
        T_g, N_g, K_g = Y_g.shape

        if T_train_default is None:
            T_train = T_g
        else:
            T_train = min(T_g, T_train_default)

        if verbose:
            print(f"\n[COMMUNITY {local_idx} / global {g_idx}] N={N_g}, K={K_g}, T={T_g}, T_train={T_train}")

        # 1) Fit Hawkes for this community
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

        # 2) Node-level adjacency A[k, i, j] for graph recovery
        A_stack = alpha_to_node_adjacency(alpha_hat, N=N_g, K=K_g)

        graph_entry = {
            "community_global_index": g_idx,
            "node_ids": np.asarray(comm["node_ids"], dtype=int),
            "A": A_stack,
            "beta": float(beta),
        }
        if "coords" in comm:
            graph_entry["coords"] = np.asarray(comm["coords"], dtype=float)
        graphs.append(graph_entry)

        # 3) Build discrete-time intensities and logits for forecasting
        lam_grid = compute_lambda_grid(
            mu=mu_hat,
            alpha=alpha_hat,
            beta=beta,
            Y_train=Y_g[:T_train],
            T_full=T_g,
        )
        logits_exact, logits_window = lambda_to_logits(
            lam_grid, horizon=horizon_months, N=N_g, K=K_g
        )

        logits_dict[local_idx] = logits_exact
        window_logits_dict[local_idx] = logits_window

    # Save graphs in format expected by eval_graph_recovery.py
    graphs_arr = np.array(graphs, dtype=object)
    np.savez_compressed(graphs_out, graphs=graphs_arr)
    print(f"\n[INFO] Saved Xu-Hawkes learned graphs for {len(graphs)} communities to {graphs_out}")

    # If T_train in meta, evaluate tail metrics using same interface
    if meta is not None and "T_train" in meta:
        T_train_eval = int(meta["T_train"])
    else:
        # fallback: use 80% of horizon as train
        T_train_eval = int(0.8 * communities[0]["Y"].shape[0])
        print(f"[WARN] meta['T_train'] missing; using T_train_eval={T_train_eval} for metrics.")

    # Note: we pass the subset of communities in the same order as logits_dict keys.
    selected_comms = [communities[g_idx] for g_idx in selected_idxs]

    metrics = evaluate_tail_metrics(
        communities=selected_comms,
        logits_dict=logits_dict,
        window_logits_dict=window_logits_dict,
        T_train=T_train_eval,
        horizon=horizon_months,
    )

    return metrics


# --------------------------------------------------------------------
# CLI wrapper
# --------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Xu-style Hawkes baseline (sim) with graph export + forecasting metrics."
    )
    p.add_argument("--sim_npz", type=str, required=True, help="Path to simulated NPZ.")
    p.add_argument("--graphs_out", type=str, required=True, help="Output NPZ for learned graphs.")
    p.add_argument("--max_communities", type=int, default=200)
    p.add_argument("--min_nodes", type=int, default=20)
    p.add_argument("--max_nodes", type=int, default=1000)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--n_epochs", type=int, default=100)
    p.add_argument("--lambda_l1", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cpu", help="'cpu' or 'cuda'")
    p.add_argument("--horizon_months", type=int, default=6, help="Window horizon for window labels.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device_str = args.device
    verbose = not args.quiet

    metrics = run_xu_hawkes_sim_and_eval(
        sim_npz=args.sim_npz,
        graphs_out=args.graphs_out,
        max_communities=args.max_communities,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        beta=args.beta,
        lr=args.lr,
        n_epochs=args.n_epochs,
        lambda_l1=args.lambda_l1,
        device_str=device_str,
        horizon_months=args.horizon_months,
        verbose=verbose,
    )

    print("\n[Xu-Hawkes baseline metrics]")
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for kk, vv in v.items():
                print(f"  {kk}: {vv}")
        else:
            print(k, ":", v)


if __name__ == "__main__":
    main()
