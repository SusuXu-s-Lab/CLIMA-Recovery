#!/usr/bin/env python3
"""
eval_graph_recovery.py

Evaluate how well learned graphs recover the *true* graphs stored in a
simulation NPZ created by simulate_from_real_npz.py (or similar).

Key features
------------
- Aligns communities via:
    * community_global_index (if present in learned graphs), OR
    * fallback: same index (learn_idx -> sim_idx).

- Aligns *nodes* via node_ids intersection:
    * Uses community['node_ids'] in sim_npz
    * Uses graph_learned['node_ids'] in learned npz (if present)
    * Evaluates only on overlapping nodes.

- Supports "top-k incoming neighbors" evaluation to match the generative
  process (k_in + self-loop per node).

- Optionally transposes the learned adjacency before evaluation
  (to be robust against orientation mismatches).

Assumed adjacency convention
----------------------------
We treat adjacency as:

    A[k, i, j] > 0  means edge j -> i  (j influences i)

This matches what we used in simulate_from_real_npz.py:
S[i, j] = 1 if j -> i.

If your training code uses the opposite convention (i -> j),
you can either:
    - call this script with --transpose_pred
or
    - hard-code a transpose of A_pred inside this script.
"""

import argparse
import numpy as np


# ----------------------------------------------------------------------
# Args
# ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate learned graphs against true graphs in a sim NPZ."
    )
    parser.add_argument(
        "--sim_npz", type=str, required=True,
        help="Simulation NPZ with true graphs (e.g., from simulate_from_real_npz.py)."
    )
    parser.add_argument(
        "--learned_graphs_npz", type=str, required=True,
        help="NPZ produced by main_multi_real.py with 'graphs' array."
    )
    parser.add_argument(
        "--max_communities", type=int, default=200,
        help="Max number of learned communities to evaluate."
    )
    parser.add_argument(
        "--min_overlap", type=int, default=10,
        help="Minimum overlapping nodes between true and learned community "
             "to include in evaluation."
    )
    parser.add_argument(
        "--top_k_pred", type=int, default=None,
        help=(
            "If set, sparsify the *learned* adjacency by keeping only the "
            "top-k incoming edges per node (per type), including self-loop. "
            "Recommended: top_k_pred = k_in + 1 used in the simulator "
            "(neighbors + self)."
        ),
    )
    parser.add_argument(
        "--transpose_pred", action="store_true",
        help=(
            "If set, transpose the learned adjacency A_pred -> A_pred^T "
            "before evaluation (per type). Use this if your training code "
            "uses the opposite adjacency orientation."
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-community alignment info."
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def sparsify_topk_incoming(A, top_k, keep_diag=True):
    """
    Given learned adjacency A of shape (K, N, N), keep only the top_k
    incoming edges per node (row) for each type, zeroing out the rest.

    Convention assumed:
        A[k, i, j] = weight of edge (j -> i).

    We:
      - for each row i, compute importance = abs(A[k, i, :])
      - optionally force-keep the diagonal (i -> i) by setting importance[i] = inf
      - pick indices of top_k entries
      - copy those entries into the output; others set to 0.
    """
    if A is None:
        return None

    A = np.asarray(A)
    if A.ndim != 3:
        raise ValueError(f"sparsify_topk_incoming expects A with shape (K, N, N), got {A.shape}")
    K, N, N2 = A.shape
    if N != N2:
        raise ValueError(f"sparsify_topk_incoming expects square matrices, got {A.shape}")

    top_k_eff = min(top_k, N)
    if top_k_eff <= 0:
        return np.zeros_like(A)

    A_out = np.zeros_like(A)
    for k in range(K):
        M = A[k]  # (N, N)
        for i in range(N):
            row = np.abs(M[i]).astype(float)
            if keep_diag:
                row[i] = np.inf  # ensure self-loop is kept

            # indices of top_k_eff entries in row i
            idx = np.argpartition(-row, top_k_eff - 1)[:top_k_eff]
            A_out[k, i, idx] = M[i, idx]

    return A_out


def extract_graph_and_nodes_from_sim(communities_sim, graphs_true_arr, sim_idx):
    """
    From the sim NPZ, get (node_ids_true, A_true) for community sim_idx.

    communities_sim[sim_idx] must be a dict with 'node_ids'.
    graphs_true_arr[sim_idx] must be a (K, N, N) array matching those nodes.
    """
    comm = communities_sim[sim_idx]
    node_ids_true = np.asarray(comm["node_ids"], dtype=int)
    A_true = np.asarray(graphs_true_arr[sim_idx])
    return node_ids_true, A_true


def extract_graph_and_nodes_from_learned(graphs_learned_arr, learn_idx):
    """
    From the learned graphs NPZ, get (node_ids_pred, A_pred, global_index).

    Cases:
      - entry is a dict with keys:
          'community_global_index', 'node_ids', 'A', ...
      - entry is just an ndarray (K, N, N); then:
          node_ids_pred = np.arange(N), global_index = learn_idx
    """
    g_obj = graphs_learned_arr[learn_idx]

    if isinstance(g_obj, dict):
        if "A" not in g_obj:
            raise ValueError(f"Learned graph dict at index {learn_idx} has no 'A' key.")
        A_pred = np.asarray(g_obj["A"])
        if "node_ids" in g_obj:
            node_ids_pred = np.asarray(g_obj["node_ids"], dtype=int)
        else:
            N = A_pred.shape[-1]
            node_ids_pred = np.arange(N, dtype=int)

        global_idx = g_obj.get("community_global_index", learn_idx)
    else:
        # assume it's already an adjacency array
        A_pred = np.asarray(g_obj)
        N = A_pred.shape[-1]
        node_ids_pred = np.arange(N, dtype=int)
        global_idx = learn_idx

    return node_ids_pred, A_pred, int(global_idx)


def align_by_node_ids(node_ids_true, A_true, node_ids_pred, A_pred, min_overlap):
    """
    Align true and predicted adjacency by node_ids intersection.

    Returns:
        A_true_al, A_pred_al, node_ids_overlap
    Or (None, None, None) if |overlap| < min_overlap.
    """
    # mapping from node_id to position
    id2pos_true = {nid: i for i, nid in enumerate(node_ids_true)}
    id2pos_pred = {nid: i for i, nid in enumerate(node_ids_pred)}

    overlap_ids = sorted(set(id2pos_true.keys()) & set(id2pos_pred.keys()))
    if len(overlap_ids) < min_overlap:
        return None, None, None

    idx_true = np.array([id2pos_true[n] for n in overlap_ids], dtype=int)
    idx_pred = np.array([id2pos_pred[n] for n in overlap_ids], dtype=int)

    # subset / reorder rows & cols to the same node order
    A_true_al = A_true[:, idx_true, :][:, :, idx_true]
    A_pred_al = A_pred[:, idx_pred, :][:, :, idx_pred]

    return A_true_al, A_pred_al, np.array(overlap_ids, dtype=int)


def binarize_adjacency(A, eps=1e-8):
    """
    Turn weighted adjacency into boolean edges.
    An edge exists if |A| > eps.
    """
    return (np.abs(A) > eps)


def compute_edge_metrics(A_true_bin, A_pred_bin):
    """
    Compute TP, FP, FN over all types / nodes / edges.
    """
    assert A_true_bin.shape == A_pred_bin.shape
    y_true = A_true_bin.reshape(-1)
    y_pred = A_pred_bin.reshape(-1)

    tp = np.logical_and(y_true, y_pred).sum()
    fp = np.logical_and(~y_true, y_pred).sum()
    fn = np.logical_and(y_true, ~y_pred).sum()
    return int(tp), int(fp), int(fn)


def summarize_results(tp_all, fp_all, fn_all, label="Overall"):
    tp = tp_all
    fp = fp_all
    fn = fn_all
    denom = tp + fp
    precision = tp / denom if denom > 0 else 0.0
    denom = tp + fn
    recall = tp / denom if denom > 0 else 0.0
    denom_f1 = precision + recall
    f1 = 2 * precision * recall / denom_f1 if denom_f1 > 0 else 0.0
    denom_j = tp + fp + fn
    jacc = tp / denom_j if denom_j > 0 else 0.0

    print(f"\n=== {label} ===")
    print(f"  TP={tp}, FP={fp}, FN={fn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print(f"  Jaccard:   {jacc:.4f}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    args = parse_args()

    # 1) Load NPZ files
    print(f"[INFO] sim_npz: {args.sim_npz}")
    sim = np.load(args.sim_npz, allow_pickle=True)
    print("[INFO] sim_npz keys:", list(sim.files))

    print(f"[INFO] learned_graphs_npz: {args.learned_graphs_npz}")
    learned = np.load(args.learned_graphs_npz, allow_pickle=True)
    print("[INFO] learned_graphs_npz keys:", list(learned.files))

    communities_sim = sim["communities"]  # object array of dicts
    graphs_true_arr = sim["graphs"]       # object array of (K_true, N, N)
    G_true = len(graphs_true_arr)

    graphs_learned_arr = learned["graphs"]
    G_learned = len(graphs_learned_arr)

    print(f"[INFO] #sim communities (graphs_true)   = {G_true}")
    print(f"[INFO] #learned communities (graphs_learned) = {G_learned}")

    G_eval = min(G_learned, args.max_communities)
    print(f"[INFO] Limiting evaluation to first {G_eval} learned communities.")

    # 2) Accumulators: overall and per-type
    tp_total = fp_total = fn_total = 0
    tp_type = {}
    fp_type = {}
    fn_type = {}

    n_eval = 0

    # 3) Loop over learned communities
    for learn_idx in range(G_eval):
        # Extract learned graph & node_ids, plus its global index
        node_ids_pred, A_pred, sim_idx = extract_graph_and_nodes_from_learned(
            graphs_learned_arr, learn_idx
        )

        if sim_idx < 0 or sim_idx >= G_true:
            if args.verbose:
                print(f"[WARN] Learned community {learn_idx} has invalid sim_idx={sim_idx}; skipping.")
            continue

        # Extract matching true graph
        node_ids_true, A_true = extract_graph_and_nodes_from_sim(
            communities_sim, graphs_true_arr, sim_idx
        )

        # Align nodes via intersection
        aligned = align_by_node_ids(
            node_ids_true, A_true, node_ids_pred, A_pred, min_overlap=args.min_overlap
        )
        if aligned[0] is None:
            if args.verbose:
                print(
                    f"[INFO] Community {learn_idx}: sim_idx={sim_idx}, "
                    f"overlap < {args.min_overlap}; skipping."
                )
            continue

        A_true_al, A_pred_al, node_ids_overlap = aligned
        N_overlap = len(node_ids_overlap)

        # Make sure #types match (if not, use min(K_true, K_pred))
        K_true, N1, N2 = A_true_al.shape
        K_pred, N3, N4 = A_pred_al.shape
        assert N1 == N2 == N3 == N4 == N_overlap

        K_use = min(K_true, K_pred)
        A_true_al = A_true_al[:K_use]
        A_pred_al = A_pred_al[:K_use]

        if args.transpose_pred:
            # transpose adjacency per type
            A_pred_al = A_pred_al.transpose(0, 2, 1)

        if args.top_k_pred is not None:
            A_pred_al = sparsify_topk_incoming(
                A_pred_al, top_k=args.top_k_pred, keep_diag=True
            )

        # Binarize both
        A_true_bin = binarize_adjacency(A_true_al)
        A_pred_bin = binarize_adjacency(A_pred_al)

        # Edge stats
        tp, fp, fn = compute_edge_metrics(A_true_bin, A_pred_bin)
        tp_total += tp
        fp_total += fp
        fn_total += fn

        # Per-type stats
        for k in range(K_use):
            tp_k, fp_k, fn_k = compute_edge_metrics(
                A_true_bin[k:k+1], A_pred_bin[k:k+1]
            )
            tp_type[k] = tp_type.get(k, 0) + tp_k
            fp_type[k] = fp_type.get(k, 0) + fp_k
            fn_type[k] = fn_type.get(k, 0) + fn_k

        n_eval += 1

        if args.verbose:
            print(
                f"[INFO] Community {learn_idx}: sim_idx={sim_idx}, "
                f"N_true={len(node_ids_true)}, N_pred={len(node_ids_pred)}, "
                f"N_overlap={N_overlap}"
            )

    print(f"[RESULTS] Evaluated {n_eval} communities.")

    # 4) Summaries
    summarize_results(tp_total, fp_total, fn_total,
                      label="Overall (all edge types, all evaluated communities)")

    for k in sorted(tp_type.keys()):
        summarize_results(tp_type[k], fp_type[k], fn_type[k],
                          label=f"Edge type {k} (type {k})")


if __name__ == "__main__":
    main()
