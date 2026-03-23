"""
run_baselines_real.py

Driver script to run all real-data baselines (Cox, RUM, spatial lag,
RecovUS-style ABM, RNN / Neural TPP) using exactly the same NPZ input
and community selection logic as the Graph Hawkes model, and to
evaluate their predictive performance via evaluate_tail_metrics.

Requires:
  - baselines_real.py (the file we discussed earlier)
  - train_eval_multi_real.py (for evaluate_tail_metrics and
    select_communities_by_distance used inside baselines_real)
"""

from __future__ import annotations

import argparse
import numpy as np
import torch

from baseline_non_hawkes_real import (
    HAS_LIFELINES,
    load_real_communities,
    run_and_eval_panel_baseline,
    train_rnn_baseline,
    rnn_predict_logits_per_community,
    train_cox,
    cox_predict_logits_per_t,
)

from train_eval_multi_real import evaluate_tail_metrics


# ----------------------------------------------------------------------
# RNN / Neural TPP-style baseline runner
# ----------------------------------------------------------------------

def run_rnn_baseline_real(
    npz_path: str,
    ref_lon: float,
    ref_lat: float,
    max_communities: int,
    min_nodes_real: int,
    max_nodes_real: int,
    hidden_dim: int = 32,
    num_epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
    horizon_months: int = 6,
):
    """
    Load real communities, train RNN baseline, and evaluate with
    evaluate_tail_metrics using the same interface as Graph Hawkes.
    """
    communities, meta, _ = load_real_communities(
        npz_path=npz_path,
        ref_lon=ref_lon,
        ref_lat=ref_lat,
        max_communities=max_communities,
        min_nodes_real=min_nodes_real,
        max_nodes_real=max_nodes_real,
    )
    T_train = int(meta["T_train"])

    print(f"[RNN baseline] Loaded {len(communities)} communities from {npz_path}")
    print(f"[RNN baseline] Training RNN with hidden_dim={hidden_dim}, epochs={num_epochs}, lr={lr}")

    model = train_rnn_baseline(
        communities=communities,
        meta=meta,
        hidden_dim=hidden_dim,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
    )

    logits_dict = rnn_predict_logits_per_community(
        model=model,
        communities=communities,
        meta=meta,
        device=device,
    )

    # For simplicity we reuse exact logits as window logits
    window_logits_dict = logits_dict

    metrics = evaluate_tail_metrics(
        communities=communities,
        logits_dict=logits_dict,
        window_logits_dict=window_logits_dict,
        T_train=T_train,
        horizon=horizon_months,
    )
    return metrics


# ----------------------------------------------------------------------
# Cox / survival baseline runner
# ----------------------------------------------------------------------

def run_cox_baseline_real(
    npz_path: str,
    ref_lon: float,
    ref_lat: float,
    max_communities: int,
    min_nodes_real: int,
    max_nodes_real: int,
    horizon_months: int = 6,
):
    """
    Load communities, fit two Cox PH models (sell & repair), convert them
    to per-(t,i,k) logits, and evaluate with evaluate_tail_metrics.

    NOTE: Requires lifelines. If lifelines is not installed, this
    function will raise a RuntimeError; we handle this in main().
    """
    communities, meta, _ = load_real_communities(
        npz_path=npz_path,
        ref_lon=ref_lon,
        ref_lat=ref_lat,
        max_communities=max_communities,
        min_nodes_real=min_nodes_real,
        max_nodes_real=max_nodes_real,
    )
    T_train = int(meta["T_train"])

    # Use generic feature names x0..x{d-1}
    d_static = communities[0]["X"].shape[1]
    feature_names = [f"x{j}" for j in range(d_static)]

    print(f"[Cox baseline] Loaded {len(communities)} communities from {npz_path}")
    print("[Cox baseline] Fitting Cox model for sell (event_type=0)")
    cph_sell = train_cox(
        communities=communities,
        meta=meta,
        event_type=0,
        feature_names=feature_names,
    )
    print("[Cox baseline] Fitting Cox model for repair (event_type=1)")
    cph_rep = train_cox(
        communities=communities,
        meta=meta,
        event_type=1,
        feature_names=feature_names,
    )

    logits_dict = cox_predict_logits_per_t(
        cph_sell=cph_sell,
        cph_rep=cph_rep,
        communities=communities,
        meta=meta,
        feature_names=feature_names,
    )

    # Again, we reuse exact logits for window labels as a simple baseline
    window_logits_dict = logits_dict

    metrics = evaluate_tail_metrics(
        communities=communities,
        logits_dict=logits_dict,
        window_logits_dict=window_logits_dict,
        T_train=T_train,
        horizon=horizon_months,
    )
    return metrics


# ----------------------------------------------------------------------
# Main driver to run all baselines with common inputs
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run real-data baselines (Cox, RUM, spatial lag, RecovUS, RNN) "
                    "with the same NPZ and community selection as the Graph Hawkes model."
    )
    # Core inputs: mirror your Graph Hawkes script
    p.add_argument("--npz_path", type=str, required=True,
                   help="Path to real Ian-by-CBG NPZ (e.g., lee_ian_by_cbg_dr0.005_tr0.8.npz)")
    p.add_argument("--ref_lon", type=float, required=True,
                   help="Reference longitude for distance-based community selection")
    p.add_argument("--ref_lat", type=float, required=True,
                   help="Reference latitude for distance-based community selection")
    p.add_argument("--max_communities", type=int, default=200,
                   help="Max number of communities to use")
    p.add_argument("--min_nodes_real", type=int, default=100,
                   help="Min number of nodes per community")
    p.add_argument("--max_nodes_real", type=int, default=900,
                   help="Max number of nodes per community")

    # Panel (RUM / spatial / RecovUS) hyperparameters
    p.add_argument("--panel_epochs", type=int, default=20,
                   help="Training epochs for panel (RUM / spatial / RecovUS) baselines")
    p.add_argument("--panel_lr", type=float, default=1e-3,
                   help="Learning rate for panel baselines")
    p.add_argument("--panel_hidden_dim", type=int, default=32,
                   help="Hidden dim for panel MLP (0 = pure linear)")

    # RNN hyperparameters
    p.add_argument("--rnn_hidden_dim", type=int, default=32,
                   help="Hidden dim for RNN baseline")
    p.add_argument("--rnn_epochs", type=int, default=20,
                   help="Training epochs for RNN baseline")
    p.add_argument("--rnn_lr", type=float, default=1e-3,
                   help="Learning rate for RNN baseline")

    # Horizon for window labels
    p.add_argument("--horizon_months", type=int, default=6,
                   help="Future horizon H (months) for window labels")

    # Device and output
    p.add_argument("--device", type=str, default="cpu",
                   help="'cpu' or 'cuda'")
    p.add_argument("--metrics_out", type=str, default="baseline_metrics_real.npz",
                   help="Output NPZ file for all baseline metrics")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; falling back to CPU.")
        device = "cpu"
    else:
        device = args.device

    all_metrics = {}

    print("\n==================== RUM baseline ====================")
    metrics_rum = run_and_eval_panel_baseline(
        npz_path=args.npz_path,
        baseline="rum",
        ref_lon=args.ref_lon,
        ref_lat=args.ref_lat,
        max_communities=args.max_communities,
        min_nodes_real=args.min_nodes_real,
        max_nodes_real=args.max_nodes_real,
        num_epochs=args.panel_epochs,
        lr=args.panel_lr,
        hidden_dim=args.panel_hidden_dim,
        device=device,
        horizon_months=args.horizon_months,
    )
    all_metrics["rum"] = metrics_rum

    print("\n==================== Spatial lag baseline ====================")
    metrics_spatial = run_and_eval_panel_baseline(
        npz_path=args.npz_path,
        baseline="spatial_lag",
        ref_lon=args.ref_lon,
        ref_lat=args.ref_lat,
        max_communities=args.max_communities,
        min_nodes_real=args.min_nodes_real,
        max_nodes_real=args.max_nodes_real,
        num_epochs=args.panel_epochs,
        lr=args.panel_lr,
        hidden_dim=args.panel_hidden_dim,
        device=device,
        horizon_months=args.horizon_months,
    )
    all_metrics["spatial_lag"] = metrics_spatial

    print("\n==================== RecovUS-style baseline ====================")
    metrics_recovus = run_and_eval_panel_baseline(
        npz_path=args.npz_path,
        baseline="recovus",
        ref_lon=args.ref_lon,
        ref_lat=args.ref_lat,
        max_communities=args.max_communities,
        min_nodes_real=args.min_nodes_real,
        max_nodes_real=args.max_nodes_real,
        num_epochs=args.panel_epochs,
        lr=args.panel_lr,
        hidden_dim=args.panel_hidden_dim,
        device=device,
        horizon_months=args.horizon_months,
    )
    all_metrics["recovus"] = metrics_recovus

    print("\n==================== RNN (Neural TPP-style) baseline ====================")
    metrics_rnn = run_rnn_baseline_real(
        npz_path=args.npz_path,
        ref_lon=args.ref_lon,
        ref_lat=args.ref_lat,
        max_communities=args.max_communities,
        min_nodes_real=args.min_nodes_real,
        max_nodes_real=args.max_nodes_real,
        hidden_dim=args.rnn_hidden_dim,
        num_epochs=args.rnn_epochs,
        lr=args.rnn_lr,
        device=device,
        horizon_months=args.horizon_months,
    )
    all_metrics["rnn"] = metrics_rnn

    if HAS_LIFELINES:
        print("\n==================== Cox / survival baseline ====================")
        try:
            metrics_cox = run_cox_baseline_real(
                npz_path=args.npz_path,
                ref_lon=args.ref_lon,
                ref_lat=args.ref_lat,
                max_communities=args.max_communities,
                min_nodes_real=args.min_nodes_real,
                max_nodes_real=args.max_nodes_real,
                horizon_months=args.horizon_months,
            )
            all_metrics["cox"] = metrics_cox
        except Exception as e:
            print(f"[Cox baseline] Error during training/eval: {e}")
    else:
        print("\n[INFO] lifelines not installed; skipping Cox baseline.")

    # Save everything
    np.savez_compressed(args.metrics_out, metrics=all_metrics)
    print(f"\n[INFO] Saved all baseline metrics to {args.metrics_out}")

    # Also pretty-print summary
    print("\n==================== Baseline metrics summary ====================")
    for name, m in all_metrics.items():
        print(f"\n--- {name} ---")
        if isinstance(m, dict):
            for k, v in m.items():
                print(f"{k}: {v}")
        else:
            print(m)


if __name__ == "__main__":
    main()
