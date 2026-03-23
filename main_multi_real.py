# main_multi_real.py
#
# Entry point for REAL-data experiments only (Ian CBG communities).
#
# Example:
#   python main_multi_real.py \
#       --real_npz lee_ian_by_cbg_dr0.005_tr0.8.npz \
#       --max_communities 200 \
#       --ref_lon -81.8723 \
#       --ref_lat 26.6406 \
#       --max_nodes_real 900 \
#       --min_nodes_real 100 \
#       --num_epochs 30 \
#       --lr 5e-2 \
#       --label_mode both \
#       --horizon_months 6 \
#       --alpha_window 0.5

from __future__ import annotations
import argparse
import numpy as np
import torch
import os
from train_eval_multi_real import (
    select_communities_by_distance,
    train_model_real,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-data multi-scale coupled Hawkes on Ian CBG communities."
    )

    parser.add_argument(
        "--real_npz",
        type=str,
        required=True,
        help="Path to real-data npz produced by prepare_real_ian_by_cbg.py",
    )

    parser.add_argument("--device", type=str, default="auto",
                        help="'cuda', 'cpu', or 'auto' (default).")

    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--lambda_edge", type=float, default=0.0)

    parser.add_argument(
        "--label_mode",
        type=str,
        choices=["exact", "window", "both"],
        default="both",
        help="Loss uses exact labels, window labels, or a mixture.",
    )
    parser.add_argument(
        "--horizon_months",
        type=int,
        default=6,
        help="Number of months in the future for window labels.",
    )
    parser.add_argument(
        "--alpha_window",
        type=float,
        default=0.5,
        help="Weight of window loss if label_mode == 'both'.",
    )

    # per-type Hawkes decays
    parser.add_argument(
        "--gamma_sell",
        type=float,
        default=0.9,
        help="Temporal decay for sell events.",
    )
    parser.add_argument(
        "--gamma_repair",
        type=float,
        default=0.8,
        help="Temporal decay for repair events.",
    )
    # parser.add_argument(
    #     "--gamma_vacate",
    #     type=float,
    #     default=0.85,
    #     help="Temporal decay for vacate events.",
    # )

    # community selection by distance
    parser.add_argument(
        "--max_communities",
        type=int,
        default=None,
        help="If set, train only on this many closest communities.",
    )
    parser.add_argument(
        "--ref_lon", type=float, required=True,
        help="Reference longitude for selecting closest communities.",
    )
    parser.add_argument(
        "--ref_lat", type=float, required=True,
        help="Reference latitude for selecting closest communities.",
    )
    parser.add_argument(
        "--min_nodes_real",
        type=int,
        default=20,
        help="Minimum nodes per community to consider.",
    )
    parser.add_argument(
        "--max_nodes_real",
        type=int,
        default=900,
        help="Maximum nodes per community to consider.",
    )
    parser.add_argument(
        "--save_graphs",
        action="store_true",
        help="If set, save the learned adjacency matrices and node metadata for each community.",
    )
    parser.add_argument(
        "--graphs_out",
        type=str,
        default=None,
        help="Path to save learned graphs (npz). Defaults to <real_npz>_learned_graphs.npz",
    )

    return parser.parse_args()


def resolve_device(device_str: str) -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def run_real(args):
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Loading real data from {args.real_npz}...")

    npz = np.load(args.real_npz, allow_pickle=True)
    communities_np = npz["communities"]
    communities = list(communities_np)

    # Select subset by distance to reference point
    selected_indices = select_communities_by_distance(
        communities=communities,
        ref_lon=args.ref_lon,
        ref_lat=args.ref_lat,
        max_communities=args.max_communities,
        min_nodes=args.min_nodes_real,
        max_nodes=args.max_nodes_real,
    )
    communities_sel = communities_np[selected_indices]
    print(f"#communities after filtering = {len(communities_sel)}")

    meta = npz["meta"].item()
    T = int(meta["T"])
    T_train = int(meta["T_train"])
    T_val = int(meta["T_val"])
    print(f"[INFO] Real-data meta: T={T}, T_train={T_train}, T_val={T_val}")

    graphs_out = args.graphs_out
    if graphs_out is None:
        base, ext = os.path.splitext(args.real_npz)
        lam_str = str(args.lambda_edge).replace(".", "p")
        graphs_out = f"{base}_lambda{lam_str}_learned_graphs.npz"
    # train
    results = train_model_real(
        npz_path=args.real_npz,
        num_epochs=args.num_epochs,
        lr=args.lr,
        lambda_edge=args.lambda_edge,
        label_mode=args.label_mode,
        horizon_months=args.horizon_months,
        alpha_window=args.alpha_window,
        device=device,
        selected_indices=selected_indices,
        gamma_k=[args.gamma_sell, args.gamma_repair],
        verbose=True,
        save_graphs=args.save_graphs,
        graphs_out=graphs_out,
    )

    print("\n[Real-data coupled Hawkes results]")
    print(f"T_train = {results['T_train']}, T_val = {results['T_val']}")
    print(f"#communities (trained) = {len(communities_sel)}")

    for name in ["sell", "repair"]:
        auc = results.get(f"{name}_auc_tail", None)
        ap = results.get(f"{name}_ap_tail", None)
        auc_w = results.get(f"{name}_auc_window_tail", None)
        ap_w = results.get(f"{name}_ap_window_tail", None)
        print(f"{name:>8}_auc_tail:          {auc}")
        print(f"{name:>8}_ap_tail:           {ap}")
        print(f"{name:>8}_auc_window_tail:  {auc_w}")
        print(f"{name:>8}_ap_window_tail:   {ap_w}")


    if args.save_graphs:
        print(f"\n[INFO] Learned graphs saved to: {results.get('graphs_out', graphs_out)}")

def main():
    args = parse_args()
    run_real(args)


if __name__ == "__main__":
    main()
