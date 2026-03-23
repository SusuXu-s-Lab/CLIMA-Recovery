# train_eval_multi_real.py
from __future__ import annotations
import math
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_multi_real import MultiScaleCoupledHawkesReal


# -------------------------------------------------------
# Utility: haversine + community selection
# -------------------------------------------------------

def haversine_km(lon1, lat1, lon2, lat2):
    """
    Great-circle distance between (lon1, lat1) and (lon2, lat2) in km.
    Inputs in degrees.
    """
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat / 2.0) ** 2 +
         np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2)
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def select_communities_by_distance(
    communities: List[dict],
    ref_lon: float,
    ref_lat: float,
    max_communities: Optional[int] = None,
    min_nodes: int = 1,
    max_nodes: int = 10**9,
):
    """
    communities: list of dicts, each with 'coords': [N_g,2], 'X': [N_g,d], 'Y':[T,N_g,K]
    Returns: list of selected indices, sorted by distance of centroid.
    """
    stats = []
    for idx, comm in enumerate(communities):
        coords = comm["coords"]  # [N_g,2]
        if coords.shape[0] < min_nodes or coords.shape[0] > max_nodes:
            continue
        lon_c = coords[:, 0].mean()
        lat_c = coords[:, 1].mean()
        dist = haversine_km(ref_lon, ref_lat, lon_c, lat_c)
        stats.append((idx, dist, coords.shape[0]))

    stats.sort(key=lambda x: x[1])  # sort by distance

    if (max_communities is not None) and (len(stats) > max_communities):
        stats = stats[:max_communities]

    selected_indices = [s[0] for s in stats]

    print("Selected communities (index, N_g, dist_km):")
    for idx, dist, N_g in stats:
        print(f"  idx={idx:3d}, N={N_g:4d}, dist={dist:5.3f}")

    return selected_indices

def export_learned_graphs(
    model,
    communities,
    comm_tensors,
    selected_indices,
    meta,
    graphs_out,
    device,
):
    """
    Export learned adjacency matrices and node metadata for each community.

    Output: an .npz file with a single array `graphs` of dtype=object.
    Each element `g = graphs[i].item()` is a dict with:

      - "community_global_index": index in the FULL communities list
      - "geoid": optional CBG GEOID (if meta["g2geoid"] exists)
      - "node_ids": array of parcel-level node_ids (indices into the
                    original fl_lee.csv; see notes below)
      - "coords": array of shape [N_g, 2] with (lat, lon)
      - "A": array of shape [K, N_g, N_g] with learned adjacency per kernel

    You can later load it via:

        data = np.load(graphs_out, allow_pickle=True)
        graphs = data["graphs"]
        g0 = graphs[0].item()
        A = g0["A"]              # shape [K, N_g, N_g]
        coords = g0["coords"]    # shape [N_g, 2]
        node_ids = g0["node_ids"]

    """
    model.eval()
    graphs = []

    # Meta usually contains g2geoid, geoid2g, etc.
    g2geoid = None
    if isinstance(meta, dict) and "g2geoid" in meta:
        g2geoid = meta["g2geoid"]

    with torch.no_grad():
        for local_idx, comm in enumerate(communities):
            # Map to original community index (in communities_all)
            if selected_indices is not None:
                global_idx = int(selected_indices[local_idx])
            else:
                global_idx = local_idx

            tensors = comm_tensors[local_idx]
            X_g = tensors["X"].to(device)
            coords_g = tensors["coords"].to(device)

            # We only need A_list here
            _, _, A_list, baseline_node = model.build_structures(X_g, coords_g)

            # A_list is a Python list of length K; each entry is [N_g, N_g]
            A_stack = torch.stack(A_list, dim=0).cpu().numpy()  # [K, N_g, N_g]

            graph_entry = {
                "community_global_index": global_idx,
                # These node_ids are exactly what you saved in prepare_real_ian_by_cbg.py
                # They index into the "node_id" column of the original parcel dataframe.
                "node_ids": comm["node_ids"],
                # coords is numpy [N_g, 2] (lat, lon) already stored in the npz
                "coords": comm["coords"],
                "A": A_stack,
            }

            if g2geoid is not None:
                graph_entry["geoid"] = g2geoid[global_idx]

            graphs.append(graph_entry)

    np.savez_compressed(graphs_out, graphs=np.array(graphs, dtype=object))
    print(f"[INFO] Saved learned graphs for {len(graphs)} communities to {graphs_out}")


# -------------------------------------------------------
# Label utilities
# -------------------------------------------------------

def build_window_labels(Y: torch.Tensor, horizon: int) -> torch.Tensor:
    """
    Y: [T, N, K] binary events (0/1).
    horizon: integer H (months).

    For each t and k, Yw[t, i, k] = 1 if there exists any event of type k
    in (t, t+H] for node i.

    Returned Yw has shape [T, N, K] (same as Y).
    """
    T, N, K = Y.shape
    Y_bool = (Y > 0)
    Yw = torch.zeros_like(Y, dtype=torch.float32)

    H_vec = [horizon] * K
    for t in range(T):
        for k in range(K):
            h = H_vec[k]
            t_end = min(T, t + h + 1)  # inclusive horizon: t+1 .. t+H
            if t + 1 >= t_end:
                # no future window
                continue
            window = Y_bool[t + 1:t_end, :, k].any(dim=0)  # [N]
            Yw[t, :, k] = window.float()
    return Yw


# -------------------------------------------------------
# Metrics
# -------------------------------------------------------

def _flatten_tail(
    Y: np.ndarray,
    P: np.ndarray,
    T_train: int,
    k: int,
    window: bool = False,
) -> (np.ndarray, np.ndarray):
    """
    Extract tail labels & scores for event type k in [T_train, T).
    """
    T, N = Y.shape[:2]
    y = Y[T_train:, :, k].reshape(-1)
    p = P[T_train:, :, k].reshape(-1)
    # Keep only positions with at least one positive somewhere to avoid trivial.
    return y, p


def _auc_ap(y_true: np.ndarray, y_score: np.ndarray):
    """
    Compute AUC and AP with simple fallback if degenerate.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    # If all labels are same, AUC is undefined; return 0.5 & AP = prevalence
    if np.all(y_true == 0) or np.all(y_true == 1):
        prevalence = float(y_true.mean())
        return 0.5, prevalence

    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = 0.5
    try:
        ap = average_precision_score(y_true, y_score)
    except Exception:
        ap = float(y_true.mean())
    return auc, ap


def evaluate_tail_metrics(
    communities: List[dict],
    logits_dict: Dict[int, np.ndarray],
    window_logits_dict: Dict[int, np.ndarray],
    T_train: int,
    horizon: int,
):
    """
    communities: list of community dicts (with Y)
    logits_dict[g]: np array [T, N_g, K] logits for exact events
    window_logits_dict[g]: logits for window labels (same shape)
    Return dict of AUC/AP per type & per label mode.
    """
    K = 2
    names = ["sell", "repair"]

    results = {}

    # Sanity check: tail event counts
    print("\n[Sanity check] Tail event counts over selected communities:")
    for k in range(K):
        exact_tail = 0
        window_tail = 0
        for g, comm in enumerate(communities):
            Yg = comm["Y"]  # [T,N,K]
            T, Ng, _ = Yg.shape
            exact_tail += Yg[T_train:, :, k].sum()
            Yw_g = build_window_labels(torch.tensor(Yg), horizon).numpy()
            window_tail += Yw_g[T_train:, :, k].sum()
        print(
            f"    {names[k]:<6}- exact_tail_count = {int(exact_tail)}, "
            f"window_tail_count (H={horizon}) = {int(window_tail)}"
        )

    # Now compute AUC/AP
    for k, name in enumerate(names):
        # exact labels
        ys_exact = []
        ps_exact = []
        # window labels
        ys_win = []
        ps_win = []

        for g, comm in enumerate(communities):
            Yg = comm["Y"].astype(np.float32)  # [T,N,K]
            T, Ng, _ = Yg.shape

            logits = logits_dict[g]  # [T,N,K]
            probs = 1.0 / (1.0 + np.exp(-logits))

            w_logits = window_logits_dict[g]
            w_probs = 1.0 / (1.0 + np.exp(-w_logits))

            y_e, p_e = _flatten_tail(Yg, probs, T_train, k, window=False)
            ys_exact.append(y_e)
            ps_exact.append(p_e)

            # build window labels on the fly
            Yw_g = build_window_labels(torch.tensor(Yg), horizon).numpy()
            y_w, p_w = _flatten_tail(Yw_g, w_probs, T_train, k, window=True)
            ys_win.append(y_w)
            ps_win.append(p_w)

        y_exact = np.concatenate(ys_exact)
        p_exact = np.concatenate(ps_exact)
        y_win = np.concatenate(ys_win)
        p_win = np.concatenate(ps_win)

        auc_e, ap_e = _auc_ap(y_exact, p_exact)
        auc_w, ap_w = _auc_ap(y_win, p_win)

        results[f"{name}_auc_tail"] = auc_e
        results[f"{name}_ap_tail"] = ap_e
        results[f"{name}_auc_window_tail"] = auc_w
        results[f"{name}_ap_window_tail"] = ap_w

    return results


# -------------------------------------------------------
# Training
# -------------------------------------------------------

def train_model_real(
    npz_path: str,
    num_epochs: int,
    lr: float,
    lambda_edge: float,
    label_mode: str,
    horizon_months: int,
    alpha_window: float,
    device,
    selected_indices=None,
    gamma_k=None,
    verbose: bool = False,
    save_graphs: bool = False,
    graphs_out: str = None,
):

    """
    Main training loop for real data with coupled Hawkes model.

    npz_path: path to lee_ian_by_cbg*.npz from prepare_real_ian_by_cbg.py
    selected_indices: optional list of community indices to train on.
    gamma_k: per-type decays [gamma_sell, gamma_repair, gamma_vacate]
    label_mode: "exact", "window", or "both".
    horizon_months: horizon H for window labels.
    alpha_window: weight of window loss if label_mode == "both".
    """
    if gamma_k is None:
        gamma_k = [0.90, 0.80, 0.85]

    data = np.load(npz_path, allow_pickle=True)
    communities_all = list(data["communities"])
    meta = data["meta"].item()
    time_index = data["time_index"]

    T = int(meta["T"])
    T_train = int(meta["T_train"])
    T_val = int(meta["T_val"])
    K = 2
    
    if verbose:
        print(f"[INFO] Real-data meta: T={T}, T_train={T_train}, T_val={T_val}")

    # select subset of communities if requested
    if selected_indices is not None:
        communities = [communities_all[i] for i in selected_indices]
    else:
        communities = communities_all

    G = len(communities)
    if G == 0:
        raise ValueError("No communities selected for training.")

    # --------------------------------------------------------------
    # Compute per-type class imbalance over training horizon
    # --------------------------------------------------------------
    Ys = []
    for comm in communities:
        Y = comm["Y"]  # [T, N_g, K]
        Ys.append(Y[:T_train])  # only training segment
    Y_all = np.concatenate(Ys, axis=1)  # [T_train, sum_g N_g, K]

    flat = Y_all.reshape(-1, K)  # [T_train * N_total, K]
    pos = flat.sum(axis=0).astype(float)  # [K]
    neg = flat.shape[0] - pos

    # Avoid zero-division
    pos[pos <= 0.0] = 1.0
    neg[neg <= 0.0] = 1.0

    ratio = neg / pos
    pos_weight_np = np.sqrt(ratio)

    print("[INFO] pos_weight per type (sell, repair) (capped):", pos_weight_np.tolist())

    pos_weight = torch.from_numpy(pos_weight_np).float().to(device)
    bce = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight)

    # ------------------------------------------------------------------
    # Build global feature stats (mu, sigma) over selected communities
    # ------------------------------------------------------------------
    X_concat = np.concatenate([comm["X"] for comm in communities], axis=0).astype(
        np.float32
    )
    mu_np = X_concat.mean(axis=0)
    sigma_np = X_concat.std(axis=0)
    sigma_np[sigma_np < 1e-6] = 1.0

    mu = torch.from_numpy(mu_np)
    sigma = torch.from_numpy(sigma_np)

    d_in = X_concat.shape[1]

    # ------------------------------------------------------------------
    # Instantiate model
    # ------------------------------------------------------------------
    model = MultiScaleCoupledHawkesReal(
        d_in=d_in,
        K=K,
        mu=mu.to(device),
        sigma=sigma.to(device),
        d_hid=128,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight)


    gamma_vec = torch.tensor(gamma_k, dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # Precompute per-community tensors on device
    # ------------------------------------------------------------------
    comm_tensors = []
    for g, comm in enumerate(communities):
        X_g = torch.from_numpy(comm["X"]).float().to(device)           # [N_g,d]
        coords_g = torch.from_numpy(comm["coords"]).float().to(device) # [N_g,2]
        Y_g = torch.from_numpy(comm["Y"]).float().to(device)           # [T,N_g,K]
        comm_tensors.append({"X": X_g, "coords": coords_g, "Y": Y_g})

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_nll = 0.0
        total_edge = 0.0
        total_count = 0

        for g in range(G):
            X_g = comm_tensors[g]["X"]        # [N_g,d]
            coords_g = comm_tensors[g]["coords"]
            Y_g = comm_tensors[g]["Y"]        # [T,N_g,K]
            Tg, N_g, K_g = Y_g.shape
            assert Tg == T and K_g == K

            # window labels for this community
            Yw_g = build_window_labels(Y_g, horizon_months)  # [T,N_g,K]

            # structures
            H_g, w_self_node, A_list, baseline_node = model.build_structures(X_g, coords_g)

            # A_list is python list length K, each [N_g,N_g]

            # Hawkes histories
            R_self = torch.zeros(N_g, K, dtype=torch.float32, device=device)
            R_nei = torch.zeros(N_g, K, dtype=torch.float32, device=device)

            loss_g = 0.0
            count_g = 0

            for t in range(T_train):
                if t > 0:
                    # update R_self with per-type decay gamma_k
                    # R_self = gamma * R_self + Y[t-1]
                    R_self = gamma_vec.view(1, K) * R_self + Y_g[t - 1]

                # neighbor history for each type k
                # R_nei[:,k] = A_k @ R_self[:,k]
                R_nei_t = []
                for k in range(K):
                    A_k = A_list[k]
                    Rk = R_self[:, k:k+1]          # [N_g,1]
                    R_nei_k = A_k @ Rk            # [N_g,1]
                    R_nei_t.append(R_nei_k)
                R_nei = torch.cat(R_nei_t, dim=1)  # [N_g,K]

                logits_t = model.step_intensity(R_self, R_nei, w_self_node, baseline_node)  # [N_g,K]
                y_t = Y_g[t]   # [N_g,K]
                yw_t = Yw_g[t] # [N_g,K]

                # exact and/or window losses
                loss_exact = bce(logits_t, y_t)
                loss_window = bce(logits_t, yw_t)  # we reuse logits_t for now

                if label_mode == "exact":
                    loss_t = loss_exact
                elif label_mode == "window":
                    loss_t = loss_window
                else:  # "both"
                    loss_t = (1.0 - alpha_window) * loss_exact + alpha_window * loss_window

                loss_g = loss_g + loss_t
                total_nll += float(loss_exact.detach().cpu().item())
                count_g += 1

            loss_g = loss_g / max(count_g, 1)

            # simple edge regularization: L1 on adjacency magnitudes
            if lambda_edge > 0.0:
                edge_reg = 0.0
                for A_k in A_list:
                    edge_reg = edge_reg + A_k.abs().mean()
                edge_reg = lambda_edge * edge_reg          # tensor
            else:
                edge_reg = torch.tensor(0.0, device=device)  # tensor, not float

            loss = loss_g + edge_reg

            # accumulate diagnostics
            total_edge += float(edge_reg.detach().cpu().item())
            total_loss += float(loss.detach().cpu().item())
            total_count += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        avg_loss = total_loss / max(total_count, 1)
        avg_nll = total_nll / max(total_count, 1)
        avg_edge = total_edge / max(total_count, 1)

        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == num_epochs):
            print(
                f"[REAL-NEURAL epoch {epoch:03d}] "
                f"loss={avg_loss:.3f} (nll={avg_nll:.3f}, edge={avg_edge:.3f})"
            )

    # ------------------------------------------------------------------
    # Evaluation on tail (T_train: T_train+T_val)
    # ------------------------------------------------------------------
    model.eval()
    logits_per_comm = {}
    window_logits_per_comm = {}
    A_sell_list = []
    A_repair_list = []
    # A_vacate_list = []
    comm_indices_for_adj = []

    with torch.no_grad():
        for g in range(G):
            X_g = comm_tensors[g]["X"]
            coords_g = comm_tensors[g]["coords"]
            Y_g = comm_tensors[g]["Y"]  # [T,N,K]
            Tg, N_g, _ = Y_g.shape

            H_g, w_self_node, A_list, baseline_node = model.build_structures(X_g, coords_g)

            A_sell_list.append(A_list[0].detach().cpu().numpy().astype("float32"))
            A_repair_list.append(A_list[1].detach().cpu().numpy().astype("float32"))
            # A_vacate_list.append(A_list[2].detach().cpu().numpy().astype("float32"))
            comm_indices_for_adj.append(selected_indices[g])

            R_self = torch.zeros(N_g, K, dtype=torch.float32, device=device)
            R_nei = torch.zeros(N_g, K, dtype=torch.float32, device=device)

            logits_TNK = []
            w_logits_TNK = []

            # precompute window labels
            Yw_g = build_window_labels(Y_g, horizon_months)

            for t in range(Tg):
                if t > 0:
                    R_self = gamma_vec.view(1, K) * R_self + Y_g[t - 1]

                R_nei_t = []
                for k in range(K):
                    A_k = A_list[k]
                    Rk = R_self[:, k:k+1]
                    R_nei_k = A_k @ Rk
                    R_nei_t.append(R_nei_k)
                R_nei = torch.cat(R_nei_t, dim=1)

                logits_t = model.step_intensity(R_self, R_nei, w_self_node, baseline_node)

                logits_TNK.append(logits_t.unsqueeze(0))

                # we reuse same logits for window; you could alternatively build
                # a separate head, but this keeps it simple
                w_logits_TNK.append(logits_t.unsqueeze(0))

            logits_TNK = torch.cat(logits_TNK, dim=0)       # [T,N_g,K]
            w_logits_TNK = torch.cat(w_logits_TNK, dim=0)   # [T,N_g,K]

            logits_per_comm[g] = logits_TNK.cpu().numpy()
            window_logits_per_comm[g] = w_logits_TNK.cpu().numpy()

    metrics = evaluate_tail_metrics(
        communities=communities,
        logits_dict=logits_per_comm,
        window_logits_dict=window_logits_per_comm,
        T_train=T_train,
        horizon=horizon_months,
    )

    # Add meta info to results
    metrics["T"] = T
    metrics["T_train"] = T_train
    metrics["T_val"] = T_val

    if save_graphs:
        if graphs_out is None:
            base, ext = os.path.splitext(npz_path)
            graphs_out = f"{base}_learned_graphs.npz"

        export_learned_graphs(
            model=model,
            communities=communities,
            comm_tensors=comm_tensors,
            selected_indices=selected_indices,
            meta=meta,
            graphs_out=graphs_out,
            device=device,
        )
        metrics["graphs_out"] = graphs_out

    metrics["A_hats"] = None


    return metrics
