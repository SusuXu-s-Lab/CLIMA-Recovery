# Household-level Decision Diffusion (HDD): Graph Neural Hawkes Process for Post-Disaster Recovery

This repository provides the code and processed data to reproduce the main results from:

> **Repair and Sale Diffusion Drives Neighborhood Recovery Divergence after Hurricanes**
> Xuechun Li, Ruxiao Chen, Yanling Sang, Graham Hults, Jingxiao Liu, Maurizio Porfiri, Cristina Dragomir, Kenneth Harrison, Juan F. Fung, Yongjun Zhang, Luis Ceferino, and Susu Xu

The framework implements a **community-shared graph neural temporal point process** (HDD) that jointly infers latent neighborhood influence networks and decision diffusion mechanisms from sparse household-level repair and sale histories after major hurricanes.

---

## Repository Structure

```
Graph-Hawkes/
├── data/
│   ├── raw/                              # Raw input data (Hurricane Ian, Lee County FL)
│   │   ├── sales_data.csv                # Property sales transactions
│   │   ├── repair_data.csv               # Building repair permits
│   │   ├── hurricane_ian_damage_map.csv  # Hazard damage intensity / loss estimates
│   │   └── building_value.csv            # Building value attributes
│   └── processed/
│       └── lee_ian_by_cbg_2state_dr0.005_tr0.6.npz  # Pre-processed community-level dataset
├── simulation/                           # Simulation study for diffusion identifiability
│   ├── simulate_from_real_npz.py         # Generate synthetic events on real features
│   ├── eval_graph_recovery.py            # Evaluate graph recovery accuracy
│   └── README.md                         # Simulation experiment instructions
├── prepare_real_ian_by_cbg.py            # Data preprocessing: raw CSVs → community-level .npz
├── main_multi_real.py                    # Main entry: train & evaluate HDD model
├── train_eval_multi_real.py              # Training and evaluation logic
├── model_multi_real.py                   # Graph-based neural Hawkes model definition
├── run_baseline_real.py                  # Non-Hawkes baselines (RUM, Spatial Lag, RecovUS, RNN)
├── baseline_classic_hawkes_real.py       # Classic sparse Hawkes baseline
├── baseline_non_hawkes_real.py           # Additional baseline implementations
├── visualize_intensity_screen_all.py     # Diffusion intensity visualization
├── make_corr_figures_intensity_graph_features.py  # Correlation analysis figures
├── viz_from_processed_both.py            # Dataset visualization
├── summarize_cbg_from_npz.py             # Community summary statistics
├── graph_interpreter.py                  # Learned graph inspection utility
├── requirements.txt                      # Python dependencies
└── .gitignore
```

---

## 1. Environment Setup

Tested with Python 3.7+.

```bash
pip install -r requirements.txt
```

Core dependencies: `numpy`, `pandas`, `scipy`, `torch`, `scikit-learn`.

---

## 2. Quick Start: Reproduce Results with Pre-processed Data

The pre-processed dataset `data/processed/lee_ian_by_cbg_2state_dr0.005_tr0.6.npz` is included, so you can directly train and evaluate the model without running the data preparation pipeline.

### 2.1 Train HDD Model

```bash
python main_multi_real.py \
  --real_npz data/processed/lee_ian_by_cbg_2state_dr0.005_tr0.6.npz \
  --max_communities 200 \
  --ref_lon -81.8723 \
  --ref_lat 26.6406 \
  --max_nodes_real 900 \
  --min_nodes_real 100 \
  --num_epochs 500 \
  --lr 5e-4 \
  --label_mode both \
  --horizon_months 6 \
  --alpha_window 0.9 \
  --lambda_edge 0.5 \
  --save_graphs
```

### 2.2 Run Baseline Methods

**Non-Hawkes baselines** (RUM, Spatial Lag, RecovUS, Neural TPP):

```bash
python run_baseline_real.py \
  --npz_path data/processed/lee_ian_by_cbg_2state_dr0.005_tr0.6.npz \
  --ref_lon -81.8723 \
  --ref_lat 26.6406 \
  --max_communities 200 \
  --min_nodes_real 100 \
  --max_nodes_real 900 \
  --panel_epochs 500 \
  --panel_lr 1e-3 \
  --panel_hidden_dim 32 \
  --rnn_hidden_dim 32 \
  --rnn_epochs 500 \
  --rnn_lr 1e-3 \
  --horizon_months 6 \
  --device cpu
```

**Classic sparse Hawkes baseline**:

```bash
python baseline_classic_hawkes_real.py \
  --real_npz data/processed/lee_ian_by_cbg_2state_dr0.005_tr0.6.npz \
  --ref_lon -81.8723 \
  --ref_lat 26.6406 \
  --max_communities 200 \
  --min_nodes_real 100 \
  --max_nodes_real 900 \
  --beta 0.5 \
  --lr 5e-2 \
  --n_epochs 500 \
  --lambda_l1 1e-3 \
  --device cpu \
  --horizon_months 6
```

### 2.3 Expected Baseline Results

| Method | Sell AUC (tail) | Repair AUC (tail) | Sell AUC (window) | Repair AUC (window) |
|--------|----------------:|-------------------:|------------------:|--------------------:|
| RUM | 0.497 | 0.569 | 0.497 | 0.561 |
| Spatial Lag | 0.522 | 0.625 | 0.530 | 0.621 |
| RecovUS | 0.504 | 0.534 | 0.498 | 0.538 |
| Neural TPP | 0.500 | 0.512 | 0.510 | 0.521 |
| Sparse Hawkes | 0.580 | 0.835 | 0.729 | 0.882 |
| **HDD (ours)** | **0.816** | **0.981** | **0.788** | **0.980** |

---

## 3. Data Preparation from Raw Sources (Optional)

If you want to regenerate the processed dataset from raw data, you need the parcel-level table `fl_lee.csv` (~982 MB), which is not included in this repository due to its size. It can be obtained from [Regrid](https://app.regrid.com) for Lee County, Florida.

Once obtained, place it at `data/raw/fl_lee.csv` and run:

```bash
python prepare_real_ian_by_cbg.py \
  --parcel_csv data/raw/fl_lee.csv \
  --sales_csv  data/raw/sales_data.csv \
  --repair_csv data/raw/repair_data.csv \
  --damage_csv data/raw/hurricane_ian_damage_map.csv \
  --map_radius_deg 0.0007 \
  --damage_radius_deg 0.005 \
  --out_npz data/processed/lee_ian_by_cbg_2state.npz \
  --train_ratio 0.6 \
  --end_date 2023-09-01
```

This produces `data/processed/lee_ian_by_cbg_2state_dr0.005_tr0.6.npz` with:
- ~560,000 residential parcels aggregated into ~575 CBG communities
- Monthly time steps from Sep 2022 to Sep 2023 (T=13)
- 41-dimensional household features (building attributes, damage, socio-demographics)
- Binary event labels for sell and repair decisions

### Key Data Preparation Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--damage_radius_deg` | Radius (degrees) for associating damage points to parcels | `0.005` |
| `--map_radius_deg` | Radius for matching point events to parcels | `0.0007` |
| `--train_ratio` | Fraction of time steps for training | `0.6` |
| `--end_date` | End date for the observation window | `2023-09-01` |
| `--min_cbg_events` | Minimum events to include a CBG community | `1` |

---

## 4. Model Training: Key Arguments

| Argument | Description | Recommended |
|----------|-------------|-------------|
| `--real_npz` | Path to preprocessed `.npz` | `data/processed/lee_ian_by_cbg_2state_dr0.005_tr0.6.npz` |
| `--max_communities` | Max CBG communities to train on (nearest to ref point) | `200` |
| `--ref_lon`, `--ref_lat` | Reference location for community selection | `-81.8723`, `26.6406` |
| `--max_nodes_real` | Max parcels per CBG | `900` |
| `--min_nodes_real` | Min parcels per CBG | `100` |
| `--num_epochs` | Training epochs | `500` |
| `--lr` | Learning rate | `5e-4` |
| `--lambda_edge` | L1 sparsity penalty on inferred graph | `0.5` |
| `--label_mode` | Label construction (`exact`, `window`, `both`) | `both` |
| `--horizon_months` | Prediction horizon H (months) | `6` |
| `--alpha_window` | Weight for window vs exact objective | `0.9` |
| `--save_graphs` | Save learned influence graphs | (flag) |
| `--device` | `cpu` or `cuda` | `cpu` |

---

## 5. Simulation Study

The `simulation/` directory contains code for controlled experiments with known ground-truth graphs to validate the model's ability to recover diffusion structure. See [`simulation/README.md`](simulation/README.md) for details.

**Quick example:**

```bash
# Generate synthetic events on real community features
python simulation/simulate_from_real_npz.py \
  --in_real_npz data/processed/lee_ian_by_cbg_2state_dr0.005_tr0.6.npz \
  --out_sim_npz simulation/sim_output.npz \
  --T 48 --seed 123 --k_in 1 \
  --alpha_self 1.0 --alpha_neigh 3.0 --decay 0.5 --base_rate 0.001 \
  --min_nodes 40 --max_nodes 800 --max_communities 100

# Train HDD on simulated data
python main_multi_real.py \
  --real_npz simulation/sim_output.npz \
  --max_communities 200 --ref_lon -81.8723 --ref_lat 26.6406 \
  --max_nodes_real 800 --min_nodes_real 40 \
  --num_epochs 500 --lr 5e-4 \
  --label_mode both --horizon_months 6 --alpha_window 0.9 \
  --lambda_edge 3.0 --save_graphs

# Evaluate graph recovery
python simulation/eval_graph_recovery.py \
  --sim_npz simulation/sim_output.npz \
  --learned_graphs_npz simulation/sim_output_lambda3p0_learned_graphs.npz \
  --max_communities 200 --min_overlap 10 --top_k_pred 1 --transpose_pred
```

---

## 6. Data Sources

| Dataset | Source | Description |
|---------|--------|-------------|
| Parcel attributes | [Regrid](https://app.regrid.com) | Lee County parcel boundaries, assessor data |
| Sales transactions | [Regrid](https://app.regrid.com) | Property sale records |
| Repair permits | [Lee County Accela](https://aca-prod.accela.com/LEECO/Cap/CapHome.aspx) | Building permit filings |
| Damage assessment | [DesignSafe](https://www.designsafe-ci.org) | Post-Hurricane Ian impact products |
| Socio-demographics | [ACS 5-Year](https://www.census.gov/data/developers/data-sets/acs-5year.html) | Census block group attributes |
| Census boundaries | [TIGER/Line](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) | CBG shapefiles |

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{li2025repair,
  title={Repair and Sale Diffusion Drives Neighborhood Recovery Divergence after Hurricanes},
  author={Li, Xuechun and Chen, Ruxiao and Sang, Yanling and Hults, Graham and Liu, Jingxiao and Porfiri, Maurizio and Dragomir, Cristina and Harrison, Kenneth and Fung, Juan F. and Zhang, Yongjun and Ceferino, Luis and Xu, Susu},
  year={2025}
}
```

## License

Please see the license file for terms of use.
