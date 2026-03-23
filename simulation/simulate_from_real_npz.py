#!/usr/bin/env python3
"""
simulate_from_real_npz.py

Generate synthetic multi-type (coupled) discrete-time Hawkes events on a
synthetic graph built from a real NPZ dataset (e.g. lee_ian_by_cbg_2state_...npz),
and save the result in a NPZ with (almost) the same structure plus:

  - 'graphs': object array of length G_sim (num simulated communities),
              each entry is a float32 array of shape (K, N_g, N_g)
              with per-type adjacency weights.
  - 'events_per_node': object array of length G_sim,
              each entry is int8 array of shape (T_sim, K, N_g)
              with simulated binary events.
  - communities[g]['Y']: replaced with simulated per-node labels
              of shape (T_sim, N_g, K) so that it matches meta['T'], meta['K'].

IMPORTANT
---------
This script DOES NOT use the real Y (neither global Y nor community['Y'])
to define intensities or the time horizon. Real data is used only for:

  - community structure: which node_ids belong to which community,
  - X_all, coords, etc. (unchanged),
  - meta['K'] (number of event types),
  - the start date of time_index.

Discrete-time coupled Hawkes model
----------------------------------
For each community (with N_g nodes), time t=0..T-1, type k=0..K-1:

    p_{t,i,k} = sigmoid( base_logit_k + h_{t,i,k} )

where

    h_{t+1} = decay * h_t
              + alpha_self  * (# events of ANY TYPE at node i at step t)
              + alpha_neigh * (# events of ANY TYPE on neighbors of i at step t)

So events of any type at a node or its neighbors increase the intensity
of ALL types at that node in future steps -> cross-type coupling.

Adjacency (true graph)
----------------------
We first build a structural adjacency S_g (N_g x N_g) with:
  - a self-loop S_g[i, i] = 1
  - up to k_in additional incoming neighbors per node

Then we create a typed adjacency A_g (K, N_g, N_g) by:

    A_g[k, :, :] = alpha_neigh * S_g
    A_g[k, i, i] += alpha_self    (so diagonals carry extra self influence)

This A_g is stored in 'graphs'.

Global labels Y_sim
-------------------
We aggregate simulated events across all communities and nodes to build
a global label tensor Y_sim of shape (T_sim, K, 2):

  Y_sim[t, k, 0] = exact count at (t,k)
  Y_sim[t, k, 1] = windowed sum over horizon H steps (default H=6 or
                   taken from meta['H'] if available)

Usage example
-------------
    python simulate_from_real_npz.py \
        --in_real_npz lee_ian_by_cbg_2state_v2_dr0.005_tr0.6.npz \
        --out_sim_npz lee_ian_sim_small.npz \
        --T 36 \
        --seed 123 \
        --k_in 3 \
        --alpha_self 2.0 \
        --alpha_neigh 1.5 \
        --decay 0.7 \
        --base_rate 3e-4 \
        --min_nodes 100 \
        --max_nodes 400 \
        --max_communities 200
"""

import argparse
import os
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate coupled multi-type Hawkes events on a synthetic "
                    "graph built from a real NPZ (communities/meta/etc)."
    )
    parser.add_argument(
        "--in_real_npz", type=str, required=True,
        help="Input real NPZ file (e.g. lee_ian_by_cbg_2state_*.npz)."
    )
    parser.add_argument(
        "--out_sim_npz", type=str, required=True,
        help="Base name of output NPZ (suffix with params will be added)."
    )
    parser.add_argument(
        "--T", type=int, default=None,
        help="Number of time steps to simulate. "
             "If None, use len(time_index) from real NPZ."
    )
    parser.add_argument(
        "--seed", type=int, default=123,
        help="Random seed."
    )
    parser.add_argument(
        "--k_in", type=int, default=5,
        help="Approx. number of incoming neighbors per node in the synthetic graph."
    )
    parser.add_argument(
        "--alpha_self", type=float, default=1.5,
        help="Self-excitation weight (per event at a node) in the Hawkes process."
    )
    parser.add_argument(
        "--alpha_neigh", type=float, default=0.75,
        help="Neighbor-excitation weight (per event at neighbors) in the Hawkes process."
    )
    parser.add_argument(
        "--decay", type=float, default=0.7,
        help="Decay factor for influence h_{t+1} = decay * h_t + ... (0 < decay < 1)."
    )

    # control community sizes and count
    parser.add_argument(
        "--min_nodes", type=int, default=1,
        help="Minimum number of nodes in a community to be simulated."
    )
    parser.add_argument(
        "--max_nodes", type=int, default=10**9,
        help="Maximum number of nodes in a community to be simulated."
    )
    parser.add_argument(
        "--max_communities", type=int, default=0,
        help="Maximum number of communities to simulate. "
             "If 0 or negative, use all that pass size filters."
    )

    parser.add_argument(
        "--base_rate", type=float, default=3e-4,
        help="Per-node per-step base event rate per type (Bernoulli). "
             "This is NOT inferred from real Y."
    )

    
    return parser.parse_args()


def fmt_scalar(x):
    """
    Sanitize floats for filenames: 1.5 -> '1p5', -0.7 -> 'm0p7'.
    """
    s = str(x)
    s = s.replace("-", "m")
    s = s.replace(".", "p")
    return s


def build_struct_adj(N, k_in, rng):
    """
    Build a simple directed structural adjacency matrix S (N x N),
    where S[i, j] = 1 means j -> i (j influences i).
    Each node i has:
      - a self-loop (S[i, i] = 1)
      - up to k_in additional incoming neighbors chosen uniformly among j != i.
    """
    S = np.zeros((N, N), dtype=np.float32)
    if N == 0:
        return S
    all_idx = np.arange(N, dtype=int)
    for i in range(N):
        # self-loop
        S[i, i] = 1.0
        # choose incoming neighbors
        if N > 1 and k_in > 0:
            candidates = all_idx[all_idx != i]
            k_eff = min(k_in, candidates.size)
            nbrs = rng.choice(candidates, size=k_eff, replace=False)
            S[i, nbrs] = 1.0
    return S


def simulate_hawkes_on_graph(S, T, K, base_logits, alpha_self, alpha_neigh, decay, rng):
    """
    Simulate a discrete-time, multi-type Hawkes process on a given structural adjacency S.

    Inputs
    ------
    S: (N, N) float32
        Structural adjacency: S[i, j] = 1 if j -> i (including self-loops).
    T: int
        Number of time steps.
    K: int
        Number of event types.
    base_logits: (K,) float
        Baseline logit per type.
    alpha_self: float
        Self-excitation weight (per event at node).
    alpha_neigh: float
        Neighbor-excitation weight (per event at neighbors).
    decay: float
        Influence decay factor (0 < decay < 1).
    rng: np.random.RandomState
        Random generator.

    Returns
    -------
    events: (T, K, N) int8
        Binary events[t, k, i] in {0,1}.
    """
    N = S.shape[0]
    events = np.zeros((T, K, N), dtype=np.int8)
    if N == 0 or T <= 0:
        return events

    # influence h[k, i] evolves over time; cross-type coupling via total events
    h = np.zeros((K, N), dtype=np.float32)

    for t in range(T):
        # compute probabilities for events at this step
        # shape: (K, N)
        logits = base_logits[:, None] + h
        # clip to avoid overflow
        logits = np.clip(logits, -20.0, 20.0)
        p = 1.0 / (1.0 + np.exp(-logits))
        # sample Bernoulli
        u = rng.rand(K, N)
        y = (u < p).astype(np.int8)
        events[t] = y

        # update influence for next step
        # total events at each node (sum across types) -> cross-type coupling
        total_events_node = y.sum(axis=0)  # shape (N,)
        # total neighbor events per node via adjacency
        # S[i, j] = 1 if j -> i; we want sum_j S[i, j] * total_events_node[j]
        total_neigh_events = S @ total_events_node  # shape (N,)

        # decay previous influence
        h = decay * h
        # self-excitation: each node's total events boosts all types at that node
        h += alpha_self * total_events_node[None, :]
        # neighbor-excitation: neighbor events boost all types
        h += alpha_neigh * total_neigh_events[None, :]

    return events


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    print(f"[INFO] Loading real NPZ from: {args.in_real_npz}")
    data = np.load(args.in_real_npz, allow_pickle=True)
    print("NPZ keys:", list(data.files))

    # --- basic inputs ---
    communities_all = data["communities"]     # object array of dicts
    time_index_real = data["time_index"]      # shape (T_real,)
    X_all = data["X_all"]                     # (N_total, D)
    node_ids = data["node_ids"]               # (N_total,)

    G_all = len(communities_all)
    T_real = len(time_index_real)

    print(f"[INFO] X_all shape = {X_all.shape}")
    print(f"[INFO] #communities in input NPZ = {G_all}")
    print(f"[INFO] time_index length in real NPZ = {T_real}")

    # --- meta & number of event types ---
    if "meta" in data.files:
        meta_real = data["meta"].item()
        print("[INFO] Loaded meta from real NPZ.")
    else:
        meta_real = {}
        print("[WARN] No 'meta' in NPZ; starting with empty dict.")

    # number of event types from meta (e.g., 2 for sell/repair)
    K = int(meta_real.get("K", 2))
    print(f"[INFO] #event types (K) = {K}")

    # --- decide simulation horizon T ---
    if args.T is None:
        # If user doesn't specify T, fall back to real horizon
        T_sim = T_real
    else:
        # ALWAYS respect user's requested T, even if > T_real
        T_sim = args.T
    print(f"[INFO] Will simulate T = {T_sim} steps (independent of real Y).")

    # --- base rates & logits (NOT from Y_real) ---
    base_rate = np.full(K, args.base_rate, dtype=np.float64)
    print(f"[INFO] Using user-specified base_rate per type: {base_rate}")
    base_logits = np.log(base_rate / (1.0 - base_rate))
    print(f"[INFO] base_logits: {base_logits}")

    # --- horizon for window labels ---
    H = meta_real.get("H", 6)

    # decide T_train/T_val based on train_ratio (NOT real T)
    train_ratio = float(meta_real.get("train_ratio", 0.6))
    T_train = int(round(train_ratio * T_sim))
    T_train = max(1, min(T_train, T_sim - 1))
    T_val = T_sim - T_train

    meta_out = dict(meta_real)
    meta_out["T"] = T_sim
    meta_out["T_train"] = int(T_train)
    meta_out["T_val"] = int(T_val)
    meta_out["K"] = int(K)
    meta_out["train_ratio"] = train_ratio

    # store as 0-d object array to match your original style
    meta_out_obj = np.array(meta_out, dtype=object)

    # Build a simulated time_index of length T_sim:
    # start from first real timestamp, extend monthly if needed.
    if len(time_index_real) > 0:
        start_ts = pd.to_datetime(time_index_real[0])
    else:
        start_ts = pd.to_datetime("2000-01-01")
    time_index_sim = pd.date_range(start=start_ts, periods=T_sim, freq="MS")

    print(f"[INFO] meta_out: T={meta_out['T']}, "
          f"T_train={meta_out['T_train']}, T_val={meta_out['T_val']}")
    print(f"[INFO] time_index_sim length = {len(time_index_sim)}")

    # --- filter communities by node count ---
    sizes = []
    for g in range(G_all):
        comm = communities_all[g]
        comm_nodes = np.asarray(comm["node_ids"], dtype=int)
        sizes.append(len(comm_nodes))
    sizes = np.asarray(sizes, dtype=int)

    mask_size = (sizes >= args.min_nodes) & (sizes <= args.max_nodes)
    idx_candidates = np.nonzero(mask_size)[0]
    print(f"[INFO] Communities passing size filter "
          f"[{args.min_nodes}, {args.max_nodes}]: {idx_candidates.size} / {G_all}")

    if idx_candidates.size == 0:
        raise RuntimeError("No communities pass the size filter; "
                           "please relax min_nodes/max_nodes.")

    # optionally subsample to max_communities
    if args.max_communities > 0 and idx_candidates.size > args.max_communities:
        idx_selected = rng.choice(
            idx_candidates, size=args.max_communities, replace=False
        )
        idx_selected = np.sort(idx_selected)
        print(f"[INFO] Subsampled to max_communities={args.max_communities}, "
              f"selected {idx_selected.size} communities.")
    else:
        idx_selected = idx_candidates
        print(f"[INFO] Using all {idx_selected.size} communities that pass size filter.")

    G_sim = idx_selected.size
    print(f"[INFO] #simulated communities (G_sim) = {G_sim}")

    # --- simulate per-community graphs & events, and overwrite community['Y'] ---
    graphs = []
    events_per_node = []
    communities_out = []

    for local_g, g_idx in enumerate(idx_selected):
        comm_orig = communities_all[g_idx]
        # comm_orig is a dict-like object
        comm_nodes = np.asarray(comm_orig["node_ids"], dtype=int)
        N_g = int(comm_nodes.size)

        # structural adjacency
        S_g = build_struct_adj(N_g, args.k_in, rng)

        # typed adjacency tensor A_g (K, N_g, N_g)
        A_g = np.zeros((K, N_g, N_g), dtype=np.float32)
        for k in range(K):
            # neighbor weights
            A_g[k, :, :] = args.alpha_neigh * S_g
            # additional self-excitation weight on diagonal
            diag_idx = np.arange(N_g, dtype=int)
            A_g[k, diag_idx, diag_idx] += args.alpha_self

        graphs.append(A_g)

        # simulate events on this graph: (T_sim, K, N_g)
        ev_g = simulate_hawkes_on_graph(
            S=S_g,
            T=T_sim,
            K=K,
            base_logits=base_logits,
            alpha_self=args.alpha_self,
            alpha_neigh=args.alpha_neigh,
            decay=args.decay,
            rng=rng,
        )
        events_per_node.append(ev_g)

        # Overwrite Y for this community with simulated per-node events:
        # expected shape for training: (T_sim, N_g, K)
        Y_g_sim = np.transpose(ev_g, (0, 2, 1))  # (T_sim, N_g, K)

        new_comm = dict(comm_orig)
        new_comm["Y"] = Y_g_sim
        communities_out.append(new_comm)

        if local_g < 3:
            print(f"[DEBUG] Community local_idx={local_g}, "
                  f"global_idx={g_idx}, N_g={N_g}, "
                  f"A_g.shape={A_g.shape}, ev_g.shape={ev_g.shape}, "
                  f"Y_g_sim.shape={Y_g_sim.shape}")

    # convert to object arrays to avoid broadcasting issues
    graphs_arr = np.empty(G_sim, dtype=object)
    events_per_node_arr = np.empty(G_sim, dtype=object)
    communities_arr = np.empty(G_sim, dtype=object)
    for i in range(G_sim):
        graphs_arr[i] = graphs[i]
        events_per_node_arr[i] = events_per_node[i]
        communities_arr[i] = communities_out[i]

    # --- build global labels Y_sim (T_sim, K, 2) from simulated events ---
    exact = np.zeros((T_sim, K), dtype=np.int64)
    for g in range(G_sim):
        # events_per_node_arr[g]: (T_sim, K, N_g)
        ev_g = events_per_node_arr[g]
        exact += ev_g.sum(axis=2)  # sum over nodes

    window = np.zeros((T_sim, K), dtype=np.int64)
    for t in range(T_sim):
        t_end = min(T_sim, t + H)
        window[t, :] = exact[t:t_end, :].sum(axis=0)

    Y_sim = np.zeros((T_sim, K, 2), dtype=np.int64)
    Y_sim[:, :, 0] = exact
    Y_sim[:, :, 1] = window

    print("[INFO] Built global Y_sim with exact/window labels.")
    print(f"[INFO] Exact counts per type over all time: {exact.sum(axis=0)}")
    print(f"[INFO] Window horizon H = {H}")

    # --- decide output filename with parameter + filter suffix ---
    base, ext = os.path.splitext(args.out_sim_npz)
    if ext == "":
        ext = ".npz"
    suffix = (
        f"_k{args.k_in}"
        f"_as{fmt_scalar(args.alpha_self)}"
        f"_an{fmt_scalar(args.alpha_neigh)}"
        f"_d{fmt_scalar(args.decay)}"
        f"_br{fmt_scalar(args.base_rate)}"
        f"_T{T_sim}"
        f"_seed{args.seed}"
        f"_mn{args.min_nodes}"
        f"_mx{args.max_nodes}"
        f"_Gc{G_sim}"
    )
    out_path = base + suffix + ext

    # --- save NPZ ---
    np.savez_compressed(
        out_path,
        communities=communities_arr,          # filtered + Y overwritten
        meta=meta_out_obj,
        time_index=time_index_sim.values,     # store as numpy array
        X_all=X_all,
        Y=Y_sim,
        node_ids=node_ids,
        graphs=graphs_arr,
        events_per_node=events_per_node_arr,
    )

    print(f"[INFO] Saved simulation NPZ to: {out_path}")


if __name__ == "__main__":
    main()
