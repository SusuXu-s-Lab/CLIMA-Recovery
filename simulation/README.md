# Simulation-based Experiments for Graph Hawkes Model

This README describes how to generate **synthetic event data** on real node features, how to **train the coupled Graph Hawkes model** on these simulations, and how to **evaluate graph recovery accuracy**. It also explains how to **sweep key parameters** for a systematic study.

The pipeline has three main scripts:

1. `prepare_real_ian_by_cbg.py` – build real feature communities from Hurricane Ian data (once).
2. `simulate_from_real_npz.py` – simulate events and true graphs on top of real features.
3. `main_multi_real.py` – train the coupled Graph Hawkes model on the simulated data.
4. `eval_graph_recovery.py` – evaluate how well the learned graphs match the true graphs.

---

## 1. Real-data preparation (one-time)

You first build the *feature-only* real dataset from raw CSVs:

```bash
python prepare_real_ian_by_cbg.py \
  --parcel_csv  hurricane_ian_data/fl_lee.csv \
  --sales_csv   hurricane_ian_data/sales_data.csv \
  --repair_csv  hurricane_ian_data/repair_data.csv \
  --damage_csv  hurricane_ian_data/hurricane_ian_damage_map.csv \
  --map_radius_deg   0.0007 \
  --damage_radius_deg 0.005 \
  --out_npz   lee_ian_by_cbg_2state_v2.npz \
  --train_ratio 0.6 \
  --end_date 2023-09-01
```

This creates a file such as:

- `lee_ian_by_cbg_2state_v2_dr0.005_tr0.6.npz`

with keys (conceptually):

- `communities`: list of CBG-based communities, each with
  - `node_ids`, `coords`, `X` (features), and `Y` (real events, here 2 types: sell/repair).
- `meta`: dictionary with
  - `T`, `T_train`, `T_val`, `K` (number of event types), etc.
- `time_index`: monthly timestamps.
- `X_all`: global feature matrix for all parcels.
- `Y`: global labels over all nodes.
- `node_ids`: global node index array.

For **simulation experiments**, we use **real features** (`X_all`, communities, node locations, etc.) but **we do not depend on the real labels `Y`**. Instead, we simulate new Hawkes event sequences and graphs.

---

## 2. Simulation on top of real features

We simulate multi-type discrete-time Hawkes processes over synthetic graphs that are built on the real communities. The script is:

```bash
python simulate_from_real_npz.py \
  --in_real_npz lee_ian_by_cbg_2state_v2_dr0.005_tr0.6.npz \
  --out_sim_npz lee_ian_sim_toy2.npz \
  --T 48 \
  --seed 123 \
  --k_in 1 \
  --alpha_self 1.0 \
  --alpha_neigh 3.0 \
  --decay 0.5 \
  --base_rate 0.001 \
  --min_nodes 40 \
  --max_nodes 800 \
  --max_communities 100
```

### 2.1. Important simulation arguments

- `--in_real_npz`  
  Real feature NPZ produced by `prepare_real_ian_by_cbg.py`.

- `--T`  
  Number of time steps to simulate (e.g. 48 months). This is **independent** of the real `Y`.  
  \[ T_{\text{sim}} = \texttt{--T} \]

- `--k_in`  
  Target number of *incoming neighbors* per node in the *true* graph (excluding the self-loop).  
  For each node \( i \), we build a structural adjacency \( S \in \{0,1\}^{N \times N} \) with:
  \[
  S_{ii} = 1, \quad
  \text{and for each } i \text{ up to } k_{\text{in}} \text{ random } j \neq i \text{ with } S_{ij} = 1.
  \]

- `--alpha_self`  
  Self-excitation weight. Events at node \( i \) at time \( t \) increase propensity for all types at the same node.

- `--alpha_neigh`  
  Neighbor-excitation weight. Events at neighbors increase propensity for all types.

- `--decay`  
  Temporal decay of influence, \( 0 < \texttt{decay} < 1 \). If \( h_t \) is the influence at time \( t \), then:
  \[
  h_{t+1} = \texttt{decay} \cdot h_t + \alpha_{\text{self}} \cdot \text{self-events} + \alpha_{\text{neigh}} \cdot \text{neighbor-events}.
  \]

- `--base_rate`  
  Per-node per-step base Bernoulli rate for each type, independent of history. We convert this to a base logit:
  \[
  \text{base\_logit} = \log \frac{\text{base\_rate}}{1 - \text{base\_rate}}.
  \]
  Smaller base rate forces the model to rely more on Hawkes dynamics and the graph.

- `--min_nodes`, `--max_nodes`  
  Filter communities by size: only simulate communities with
  \[
  \texttt{min\_nodes} \le N_g \le \texttt{max\_nodes}.
  \]

- `--max_communities`  
  Maximum number of communities to simulate after filtering by size. If there are more, a random subset is chosen (with the given seed).  
  \[ G_{\text{sim}} = \min(\#\text{filtered communities},\ \texttt{max\_communities}) \]

### 2.2. Discrete-time coupled Hawkes model

For a community with \( N_g \) nodes and \( K \) event types, at each step \( t = 0,\ldots,T-1 \), type \( k \), node \( i \):

- Probability of event:
  \[
  p_{t,i,k} = \sigma(\text{base\_logit}_k + h_{t,i,k}),
  \]
  where \( \sigma(\cdot) \) is the sigmoid.

- Influence dynamics:
  \[
  h_{t+1} = \texttt{decay} \cdot h_t
            + \alpha_{\text{self}} \cdot \text{SelfEvents}_t
            + \alpha_{\text{neigh}} \cdot \text{NeighEvents}_t.
  \]

- **Cross-type coupling**: SelfEvents and NeighEvents use *total* events across all types at each node, so one type’s activity can excite all types.

- The typed adjacency tensor \( A_g \in \mathbb{R}^{K \times N_g \times N_g} \) is defined from the structural adjacency \( S_g \) via
  \[
  A_g[k, :, :] = \alpha_{\text{neigh}} \cdot S_g,\quad
  A_g[k, i, i] \mathrel{+}= \alpha_{\text{self}}.
  \]

The simulation script saves:

- `graphs[g] = A_g` – the true typed adjacency for community \( g \)
- `events_per_node[g]` – simulated events of shape `(T_sim, K, N_g)`
- `Y` – global exact + window labels `(T_sim, K, 2)`
- `meta`, `time_index`, `communities`, `X_all`, `node_ids`

The output filename automatically encodes parameters, for example:

```text
lee_ian_sim_toy2_k1_as1p0_an3p0_d0p5_br0p001_T48_seed123_mn40_mx800_Gc100.npz
```

where `k1` means `k_in=1`, `as1p0` is `alpha_self=1.0`, `an3p0` is `alpha_neigh=3.0`, etc.

---

## 3. Training on simulated data

To train the coupled Graph Hawkes model on the simulated NPZ, use `main_multi_real.py` with the simulated file as `--real_npz`:

```bash
python main_multi_real.py \
  --real_npz lee_ian_sim_toy2_k1_as1p0_an3p0_d0p5_br0p001_T48_seed123_mn40_mx800_Gc100.npz \
  --max_communities 200 \
  --ref_lon -81.8723 \
  --ref_lat 26.6406 \
  --max_nodes_real 800 \
  --min_nodes_real 40 \
  --num_epochs 200 \
  --lr 5e-4 \
  --label_mode both \
  --horizon_months 6 \
  --alpha_window 0.9 \
  --lambda_edge 1.0 \
  --save_graphs
```


Key points:

- `--real_npz` is now the **simulated** NPZ. It still has real-world node features but synthetic events and graphs.
- `--min_nodes_real`, `--max_nodes_real` should be aligned with the simulation’s `min_nodes`, `max_nodes` so you train on the same communities.
- `--lambda_edge` controls sparsity of the learned graph (L1 on edge weights). It is crucial for graph recovery.
- `--save_graphs` ensures the learned adjacency tensors are written to a file:
  ```text
  <real_npz_prefix>_lambda{λ}_learned_graphs.npz
  ```
- `main_multi_real.py` prints out **forecasting performance**:
  - Tail AUC / AP (exact `T_val` labels)
  - Window AUC / AP (H-step window labels)

These are your *prediction* metrics on the simulated test period.

```bash
python main_multi_real.py   --real_npz lee_ian_sim_toy2_k1_as1p0_an3p0_d0p5_br0p001_T48_seed123_mn40_mx800_Gc100.npz   --max_communities 200   --ref_lon -81.8723   --ref_lat 26.6406   --max_nodes_real 800   --min_nodes_real 40   --num_epochs 500   --lr 5e-4   --label_mode both   --horizon_months 6   --alpha_window 0.9   --lambda_edge 3   --save_graphs
Using device: cpu
Loading real data from lee_ian_sim_toy2_k1_as1p0_an3p0_d0p5_br0p001_T48_seed123_mn40_mx800_Gc100.npz...
Selected communities (index, N_g, dist_km):
  idx= 93, N= 274, dist=0.544
  idx= 68, N= 244, dist=1.470
  idx= 88, N= 298, dist=1.671
  idx= 84, N= 238, dist=1.748
  idx= 92, N= 496, dist=1.878
  idx= 53, N= 406, dist=1.910
  idx= 73, N= 377, dist=2.766
  idx= 91, N= 474, dist=3.842
  idx= 76, N= 351, dist=3.900
  idx= 15, N= 316, dist=4.320
  idx= 12, N= 713, dist=4.472
  idx= 74, N= 181, dist=4.590
  idx=  0, N= 363, dist=5.533
  idx= 35, N= 306, dist=5.545
  idx= 26, N= 506, dist=5.585
  idx= 89, N= 461, dist=5.818
  idx= 94, N= 379, dist=6.360
  idx= 69, N= 301, dist=6.422
  idx= 13, N= 325, dist=6.806
  idx= 83, N= 465, dist=6.885
  idx= 82, N= 445, dist=7.129
  idx= 11, N= 602, dist=7.468
  idx= 72, N= 475, dist=7.518
  idx=  3, N= 596, dist=7.630
  idx= 78, N= 294, dist=7.636
  idx= 40, N= 506, dist=7.645
  idx= 39, N= 525, dist=8.307
  idx= 14, N= 426, dist=8.785
  idx= 90, N= 480, dist=8.844
  idx= 50, N= 347, dist=8.876
  idx= 51, N= 621, dist=8.941
  idx= 97, N= 665, dist=9.112
  idx= 18, N= 581, dist=9.324
  idx= 55, N= 550, dist=9.435
  idx= 32, N= 644, dist=9.536
  idx= 96, N= 426, dist=9.550
  idx= 20, N= 713, dist=9.704
  idx= 22, N= 718, dist=9.857
  idx=  2, N= 680, dist=9.933
  idx= 59, N= 562, dist=9.993
  idx= 81, N= 499, dist=10.598
  idx= 46, N= 418, dist=10.612
  idx= 30, N= 781, dist=10.642
  idx= 28, N= 703, dist=10.742
  idx= 25, N= 618, dist=11.148
  idx= 70, N= 566, dist=11.435
  idx= 17, N= 579, dist=11.600
  idx= 54, N= 497, dist=11.753
  idx= 52, N= 735, dist=11.809
  idx= 41, N= 461, dist=11.917
  idx= 77, N= 347, dist=11.963
  idx= 23, N= 512, dist=12.026
  idx= 75, N= 513, dist=12.119
  idx= 21, N= 423, dist=12.156
  idx= 66, N= 132, dist=12.843
  idx= 37, N= 614, dist=12.895
  idx= 65, N= 590, dist=12.967
  idx= 98, N= 674, dist=13.040
  idx= 33, N= 480, dist=13.080
  idx=  8, N= 701, dist=13.172
  idx=  1, N= 342, dist=13.425
  idx= 87, N= 757, dist=13.998
  idx= 45, N= 760, dist=14.115
  idx= 34, N= 684, dist=14.540
  idx= 16, N= 786, dist=14.551
  idx= 95, N= 613, dist=15.348
  idx= 60, N= 752, dist=16.071
  idx= 71, N= 480, dist=16.201
  idx= 38, N= 736, dist=16.348
  idx= 63, N= 606, dist=16.924
  idx= 64, N= 228, dist=18.155
  idx= 49, N= 424, dist=19.757
  idx= 19, N= 589, dist=20.109
  idx= 47, N= 616, dist=21.586
  idx= 44, N= 494, dist=22.462
  idx= 43, N= 586, dist=22.740
  idx= 86, N= 298, dist=22.752
  idx= 57, N= 527, dist=22.970
  idx= 48, N= 444, dist=23.643
  idx=  5, N= 796, dist=24.526
  idx= 62, N= 637, dist=24.580
  idx= 79, N= 578, dist=24.763
  idx= 42, N= 770, dist=25.596
  idx= 56, N= 474, dist=25.616
  idx=  6, N= 775, dist=25.671
  idx=  7, N= 494, dist=26.144
  idx=  4, N= 440, dist=26.749
  idx= 80, N= 663, dist=27.217
  idx= 99, N= 115, dist=31.654
  idx= 36, N= 696, dist=33.639
  idx= 58, N= 656, dist=34.443
  idx= 85, N= 611, dist=34.596
  idx= 24, N= 776, dist=34.974
  idx=  9, N= 688, dist=35.116
  idx= 27, N= 740, dist=35.261
  idx= 29, N= 633, dist=37.653
  idx= 31, N= 515, dist=38.450
  idx= 10, N= 550, dist=38.607
  idx= 61, N= 755, dist=40.697
  idx= 67, N= 392, dist=41.023
#communities after filtering = 100
[INFO] Real-data meta: T=48, T_train=29, T_val=19
[INFO] Real-data meta: T=48, T_train=29, T_val=19
[INFO] pos_weight per type (sell, repair) (capped): [10.25197145399543, 10.213706002301107]
[REAL-NEURAL epoch 001] loss=0.616 (nll=10.418, edge=0.079)
[REAL-NEURAL epoch 005] loss=0.481 (nll=6.127, edge=0.063)
[REAL-NEURAL epoch 010] loss=0.458 (nll=5.493, edge=0.063)
[REAL-NEURAL epoch 015] loss=0.444 (nll=5.147, edge=0.063)
[REAL-NEURAL epoch 020] loss=0.437 (nll=4.983, edge=0.063)
[REAL-NEURAL epoch 025] loss=0.433 (nll=4.858, edge=0.063)
[REAL-NEURAL epoch 030] loss=0.431 (nll=4.773, edge=0.063)
[REAL-NEURAL epoch 035] loss=0.427 (nll=4.778, edge=0.063)
[REAL-NEURAL epoch 040] loss=0.424 (nll=4.708, edge=0.063)
[REAL-NEURAL epoch 045] loss=0.422 (nll=4.617, edge=0.063)
[REAL-NEURAL epoch 050] loss=0.421 (nll=4.652, edge=0.063)
[REAL-NEURAL epoch 055] loss=0.417 (nll=4.576, edge=0.063)
[REAL-NEURAL epoch 060] loss=0.418 (nll=4.601, edge=0.063)
[REAL-NEURAL epoch 065] loss=0.413 (nll=4.459, edge=0.063)
[REAL-NEURAL epoch 070] loss=0.413 (nll=4.391, edge=0.063)
[REAL-NEURAL epoch 075] loss=0.412 (nll=4.477, edge=0.063)
[REAL-NEURAL epoch 080] loss=0.410 (nll=4.366, edge=0.063)
[REAL-NEURAL epoch 085] loss=0.413 (nll=4.445, edge=0.063)
[REAL-NEURAL epoch 090] loss=0.408 (nll=4.351, edge=0.063)
[REAL-NEURAL epoch 095] loss=0.407 (nll=4.284, edge=0.063)
[REAL-NEURAL epoch 100] loss=0.405 (nll=4.336, edge=0.063)
[REAL-NEURAL epoch 105] loss=0.409 (nll=4.272, edge=0.063)
[REAL-NEURAL epoch 110] loss=0.401 (nll=4.288, edge=0.063)
[REAL-NEURAL epoch 115] loss=0.407 (nll=4.381, edge=0.063)
[REAL-NEURAL epoch 120] loss=0.401 (nll=4.302, edge=0.063)
[REAL-NEURAL epoch 125] loss=0.399 (nll=4.300, edge=0.063)
[REAL-NEURAL epoch 130] loss=0.400 (nll=4.303, edge=0.063)
[REAL-NEURAL epoch 135] loss=0.400 (nll=4.271, edge=0.063)
[REAL-NEURAL epoch 140] loss=0.401 (nll=4.298, edge=0.063)
[REAL-NEURAL epoch 145] loss=0.399 (nll=4.285, edge=0.063)
[REAL-NEURAL epoch 150] loss=0.397 (nll=4.250, edge=0.063)
[REAL-NEURAL epoch 155] loss=0.398 (nll=4.245, edge=0.063)
[REAL-NEURAL epoch 160] loss=0.396 (nll=4.237, edge=0.063)
[REAL-NEURAL epoch 165] loss=0.395 (nll=4.273, edge=0.063)
[REAL-NEURAL epoch 170] loss=0.393 (nll=4.195, edge=0.063)
[REAL-NEURAL epoch 175] loss=0.396 (nll=4.290, edge=0.063)
[REAL-NEURAL epoch 180] loss=0.394 (nll=4.266, edge=0.063)
[REAL-NEURAL epoch 185] loss=0.392 (nll=4.232, edge=0.063)
[REAL-NEURAL epoch 190] loss=0.394 (nll=4.325, edge=0.063)
[REAL-NEURAL epoch 195] loss=0.390 (nll=4.166, edge=0.063)
[REAL-NEURAL epoch 200] loss=0.397 (nll=4.229, edge=0.063)
[REAL-NEURAL epoch 205] loss=0.387 (nll=4.164, edge=0.063)
[REAL-NEURAL epoch 210] loss=0.391 (nll=4.252, edge=0.063)
[REAL-NEURAL epoch 215] loss=0.391 (nll=4.280, edge=0.063)
[REAL-NEURAL epoch 220] loss=0.385 (nll=4.191, edge=0.063)
[REAL-NEURAL epoch 225] loss=0.389 (nll=4.190, edge=0.063)
[REAL-NEURAL epoch 230] loss=0.388 (nll=4.206, edge=0.063)
[REAL-NEURAL epoch 235] loss=0.385 (nll=4.115, edge=0.063)
[REAL-NEURAL epoch 240] loss=0.389 (nll=4.193, edge=0.063)
[REAL-NEURAL epoch 245] loss=0.386 (nll=4.194, edge=0.063)
[REAL-NEURAL epoch 250] loss=0.387 (nll=4.205, edge=0.063)
[REAL-NEURAL epoch 255] loss=0.384 (nll=4.201, edge=0.063)
[REAL-NEURAL epoch 260] loss=0.387 (nll=4.252, edge=0.063)
[REAL-NEURAL epoch 265] loss=0.382 (nll=4.147, edge=0.063)
[REAL-NEURAL epoch 270] loss=0.383 (nll=4.090, edge=0.063)
[REAL-NEURAL epoch 275] loss=0.384 (nll=4.099, edge=0.063)
[REAL-NEURAL epoch 280] loss=0.383 (nll=4.180, edge=0.063)
[REAL-NEURAL epoch 285] loss=0.382 (nll=4.200, edge=0.063)
[REAL-NEURAL epoch 290] loss=0.382 (nll=4.239, edge=0.063)
[REAL-NEURAL epoch 295] loss=0.379 (nll=4.158, edge=0.063)
[REAL-NEURAL epoch 300] loss=0.379 (nll=4.159, edge=0.063)
[REAL-NEURAL epoch 305] loss=0.375 (nll=4.021, edge=0.063)
[REAL-NEURAL epoch 310] loss=0.379 (nll=4.194, edge=0.063)
[REAL-NEURAL epoch 315] loss=0.375 (nll=4.059, edge=0.063)
[REAL-NEURAL epoch 320] loss=0.378 (nll=4.186, edge=0.063)
[REAL-NEURAL epoch 325] loss=0.378 (nll=4.157, edge=0.063)
[REAL-NEURAL epoch 330] loss=0.377 (nll=4.182, edge=0.063)
[REAL-NEURAL epoch 335] loss=0.375 (nll=4.110, edge=0.063)
[REAL-NEURAL epoch 340] loss=0.376 (nll=4.169, edge=0.063)
[REAL-NEURAL epoch 345] loss=0.374 (nll=4.002, edge=0.063)
[REAL-NEURAL epoch 350] loss=0.372 (nll=4.048, edge=0.063)
[REAL-NEURAL epoch 355] loss=0.376 (nll=4.101, edge=0.063)
[REAL-NEURAL epoch 360] loss=0.375 (nll=4.070, edge=0.063)
[REAL-NEURAL epoch 365] loss=0.374 (nll=4.207, edge=0.063)
[REAL-NEURAL epoch 370] loss=0.373 (nll=4.095, edge=0.063)
[REAL-NEURAL epoch 375] loss=0.371 (nll=4.048, edge=0.063)
[REAL-NEURAL epoch 380] loss=0.374 (nll=4.072, edge=0.063)
[REAL-NEURAL epoch 385] loss=0.372 (nll=4.046, edge=0.063)
[REAL-NEURAL epoch 390] loss=0.369 (nll=4.124, edge=0.063)
[REAL-NEURAL epoch 395] loss=0.369 (nll=4.060, edge=0.063)
[REAL-NEURAL epoch 400] loss=0.373 (nll=4.102, edge=0.063)
[REAL-NEURAL epoch 405] loss=0.374 (nll=4.132, edge=0.063)
[REAL-NEURAL epoch 410] loss=0.364 (nll=3.985, edge=0.063)
[REAL-NEURAL epoch 415] loss=0.371 (nll=4.154, edge=0.063)
[REAL-NEURAL epoch 420] loss=0.368 (nll=4.100, edge=0.063)
[REAL-NEURAL epoch 425] loss=0.367 (nll=4.124, edge=0.063)
[REAL-NEURAL epoch 430] loss=0.367 (nll=4.006, edge=0.063)
[REAL-NEURAL epoch 435] loss=0.370 (nll=4.082, edge=0.063)

[REAL-NEURAL epoch 440] loss=0.370 (nll=4.074, edge=0.063)
[REAL-NEURAL epoch 445] loss=0.367 (nll=4.106, edge=0.063)
[REAL-NEURAL epoch 450] loss=0.368 (nll=4.038, edge=0.063)
[REAL-NEURAL epoch 455] loss=0.368 (nll=4.056, edge=0.063)
[REAL-NEURAL epoch 460] loss=0.362 (nll=4.047, edge=0.063)
[REAL-NEURAL epoch 465] loss=0.369 (nll=4.027, edge=0.063)
[REAL-NEURAL epoch 470] loss=0.363 (nll=4.072, edge=0.063)
[REAL-NEURAL epoch 475] loss=0.366 (nll=4.132, edge=0.063)
[REAL-NEURAL epoch 480] loss=0.365 (nll=4.044, edge=0.063)
[REAL-NEURAL epoch 485] loss=0.361 (nll=3.978, edge=0.063)
[REAL-NEURAL epoch 490] loss=0.362 (nll=4.026, edge=0.063)
[REAL-NEURAL epoch 495] loss=0.363 (nll=4.003, edge=0.063)
[REAL-NEURAL epoch 500] loss=0.364 (nll=4.061, edge=0.063)

[Sanity check] Tail event counts over selected communities:
    sell  - exact_tail_count = 52154, window_tail_count (H=6) = 65310
    repair- exact_tail_count = 52185, window_tail_count (H=6) = 65167
[INFO] Saved learned graphs for 100 communities to lee_ian_sim_toy2_k1_as1p0_an3p0_d0p5_br0p001_T48_seed123_mn40_mx800_Gc100_lambda3p0_learned_graphs.npz

[Real-data coupled Hawkes results]
T_train = 29, T_val = 19
#communities (trained) = 100
    sell_auc_tail:          0.9512121637433973
    sell_ap_tail:           0.9002850597418652
    sell_auc_window_tail:  0.870780195288359
    sell_ap_window_tail:   0.6838812668672531
  repair_auc_tail:          0.9665103708410693
  repair_ap_tail:           0.919523541668219
  repair_auc_window_tail:  0.8816899826773223
  repair_ap_window_tail:   0.6944925125333454

[INFO] Learned graphs saved to: lee_ian_sim_toy2_k1_as1p0_an3p0_d0p5_br0p001_T48_seed123_mn40_mx800_Gc100_lambda3p0_learned_graphs.npz
```

---

## 4. Graph recovery evaluation

To evaluate how well the learned graphs match the simulated ground truth, use:

```bash
python eval_graph_recovery.py \
  --sim_npz lee_ian_sim_toy2_k1_as1p0_an3p0_d0p5_br0p001_T48_seed123_mn40_mx800_Gc100.npz \
  --learned_graphs_npz lee_ian_sim_toy2_k1_as1p0_an3p0_d0p5_br0p001_T48_seed123_mn40_mx800_Gc100_lambda1p0_learned_graphs.npz \
  --max_communities 200 \
  --min_overlap 10 \
  --top_k_pred 1 \
  --transpose_pred \
  --verbose
```

```bash
python eval_graph_recovery.py --sim_npz lee_ian_sim_toy2_k1_as1p0_an3p0_d0p5_br0p001_T48_seed123_mn40_mx800_Gc100.npz --learned_graphs_npz lee_ian_sim_toy2_k1_as1p0_an3p0_d0p5_br0p001_T48_seed123_mn40_mx800_Gc100_lambda3p0_learned_graphs.npz --max_communities 200 --min_overlap 10 --top_k_pred 1 --transpose_pred --verbose
[INFO] sim_npz: lee_ian_sim_toy2_k1_as1p0_an3p0_d0p5_br0p001_T48_seed123_mn40_mx800_Gc100.npz
[INFO] sim_npz keys: ['communities', 'meta', 'time_index', 'X_all', 'Y', 'node_ids', 'graphs', 'events_per_node']
[INFO] learned_graphs_npz: lee_ian_sim_toy2_k1_as1p0_an3p0_d0p5_br0p001_T48_seed123_mn40_mx800_Gc100_lambda3p0_learned_graphs.npz
[INFO] learned_graphs_npz keys: ['graphs']
[INFO] #sim communities (graphs_true)   = 100
[INFO] #learned communities (graphs_learned) = 100
[INFO] Limiting evaluation to first 100 learned communities.
[INFO] Community 0: sim_idx=93, N_true=274, N_pred=274, N_overlap=274
[INFO] Community 1: sim_idx=68, N_true=244, N_pred=244, N_overlap=244
[INFO] Community 2: sim_idx=88, N_true=298, N_pred=298, N_overlap=298
[INFO] Community 3: sim_idx=84, N_true=238, N_pred=238, N_overlap=238
[INFO] Community 4: sim_idx=92, N_true=496, N_pred=496, N_overlap=496
[INFO] Community 5: sim_idx=53, N_true=406, N_pred=406, N_overlap=406
[INFO] Community 6: sim_idx=73, N_true=377, N_pred=377, N_overlap=377
[INFO] Community 7: sim_idx=91, N_true=474, N_pred=474, N_overlap=474
[INFO] Community 8: sim_idx=76, N_true=351, N_pred=351, N_overlap=351
[INFO] Community 9: sim_idx=15, N_true=316, N_pred=316, N_overlap=316
[INFO] Community 10: sim_idx=12, N_true=713, N_pred=713, N_overlap=713
[INFO] Community 11: sim_idx=74, N_true=181, N_pred=181, N_overlap=181
[INFO] Community 12: sim_idx=0, N_true=363, N_pred=363, N_overlap=363
[INFO] Community 13: sim_idx=35, N_true=306, N_pred=306, N_overlap=306
[INFO] Community 14: sim_idx=26, N_true=506, N_pred=506, N_overlap=506
[INFO] Community 15: sim_idx=89, N_true=461, N_pred=461, N_overlap=461
[INFO] Community 16: sim_idx=94, N_true=379, N_pred=379, N_overlap=379
[INFO] Community 17: sim_idx=69, N_true=301, N_pred=301, N_overlap=301
[INFO] Community 18: sim_idx=13, N_true=325, N_pred=325, N_overlap=325
[INFO] Community 19: sim_idx=83, N_true=465, N_pred=465, N_overlap=465
[INFO] Community 20: sim_idx=82, N_true=445, N_pred=445, N_overlap=445
[INFO] Community 21: sim_idx=11, N_true=602, N_pred=602, N_overlap=602
[INFO] Community 22: sim_idx=72, N_true=475, N_pred=475, N_overlap=475
[INFO] Community 23: sim_idx=3, N_true=596, N_pred=596, N_overlap=596
[INFO] Community 24: sim_idx=78, N_true=294, N_pred=294, N_overlap=294
[INFO] Community 25: sim_idx=40, N_true=506, N_pred=506, N_overlap=506
[INFO] Community 26: sim_idx=39, N_true=525, N_pred=525, N_overlap=525
[INFO] Community 27: sim_idx=14, N_true=426, N_pred=426, N_overlap=426
[INFO] Community 28: sim_idx=90, N_true=480, N_pred=480, N_overlap=480
[INFO] Community 29: sim_idx=50, N_true=347, N_pred=347, N_overlap=347
[INFO] Community 30: sim_idx=51, N_true=621, N_pred=621, N_overlap=621
[INFO] Community 31: sim_idx=97, N_true=665, N_pred=665, N_overlap=665
[INFO] Community 32: sim_idx=18, N_true=581, N_pred=581, N_overlap=581
[INFO] Community 33: sim_idx=55, N_true=550, N_pred=550, N_overlap=550
[INFO] Community 34: sim_idx=32, N_true=644, N_pred=644, N_overlap=644
[INFO] Community 35: sim_idx=96, N_true=426, N_pred=426, N_overlap=426
[INFO] Community 36: sim_idx=20, N_true=713, N_pred=713, N_overlap=713
[INFO] Community 37: sim_idx=22, N_true=718, N_pred=718, N_overlap=718
[INFO] Community 38: sim_idx=2, N_true=680, N_pred=680, N_overlap=680
[INFO] Community 39: sim_idx=59, N_true=562, N_pred=562, N_overlap=562
[INFO] Community 40: sim_idx=81, N_true=499, N_pred=499, N_overlap=499
[INFO] Community 41: sim_idx=46, N_true=418, N_pred=418, N_overlap=418
[INFO] Community 42: sim_idx=30, N_true=781, N_pred=781, N_overlap=781
[INFO] Community 43: sim_idx=28, N_true=703, N_pred=703, N_overlap=703
[INFO] Community 44: sim_idx=25, N_true=618, N_pred=618, N_overlap=618
[INFO] Community 45: sim_idx=70, N_true=566, N_pred=566, N_overlap=566
[INFO] Community 46: sim_idx=17, N_true=579, N_pred=579, N_overlap=579
[INFO] Community 47: sim_idx=54, N_true=497, N_pred=497, N_overlap=497
[INFO] Community 48: sim_idx=52, N_true=735, N_pred=735, N_overlap=735
[INFO] Community 49: sim_idx=41, N_true=461, N_pred=461, N_overlap=461
[INFO] Community 50: sim_idx=77, N_true=347, N_pred=347, N_overlap=347
[INFO] Community 51: sim_idx=23, N_true=512, N_pred=512, N_overlap=512
[INFO] Community 52: sim_idx=75, N_true=513, N_pred=513, N_overlap=513
[INFO] Community 53: sim_idx=21, N_true=423, N_pred=423, N_overlap=423
[INFO] Community 54: sim_idx=66, N_true=132, N_pred=132, N_overlap=132
[INFO] Community 55: sim_idx=37, N_true=614, N_pred=614, N_overlap=614
[INFO] Community 56: sim_idx=65, N_true=590, N_pred=590, N_overlap=590
[INFO] Community 57: sim_idx=98, N_true=674, N_pred=674, N_overlap=674
[INFO] Community 58: sim_idx=33, N_true=480, N_pred=480, N_overlap=480
[INFO] Community 59: sim_idx=8, N_true=701, N_pred=701, N_overlap=701
[INFO] Community 60: sim_idx=1, N_true=342, N_pred=342, N_overlap=342
[INFO] Community 61: sim_idx=87, N_true=757, N_pred=757, N_overlap=757
[INFO] Community 62: sim_idx=45, N_true=760, N_pred=760, N_overlap=760
[INFO] Community 63: sim_idx=34, N_true=684, N_pred=684, N_overlap=684
[INFO] Community 64: sim_idx=16, N_true=786, N_pred=786, N_overlap=786
[INFO] Community 65: sim_idx=95, N_true=613, N_pred=613, N_overlap=613
[INFO] Community 66: sim_idx=60, N_true=752, N_pred=752, N_overlap=752
[INFO] Community 67: sim_idx=71, N_true=480, N_pred=480, N_overlap=480
[INFO] Community 68: sim_idx=38, N_true=736, N_pred=736, N_overlap=736
[INFO] Community 69: sim_idx=63, N_true=606, N_pred=606, N_overlap=606
[INFO] Community 70: sim_idx=64, N_true=228, N_pred=228, N_overlap=228
[INFO] Community 71: sim_idx=49, N_true=424, N_pred=424, N_overlap=424
[INFO] Community 72: sim_idx=19, N_true=589, N_pred=589, N_overlap=589
[INFO] Community 73: sim_idx=47, N_true=616, N_pred=616, N_overlap=616
[INFO] Community 74: sim_idx=44, N_true=494, N_pred=494, N_overlap=494
[INFO] Community 75: sim_idx=43, N_true=586, N_pred=586, N_overlap=586
[INFO] Community 76: sim_idx=86, N_true=298, N_pred=298, N_overlap=298
[INFO] Community 77: sim_idx=57, N_true=527, N_pred=527, N_overlap=527
[INFO] Community 78: sim_idx=48, N_true=444, N_pred=444, N_overlap=444
[INFO] Community 79: sim_idx=5, N_true=796, N_pred=796, N_overlap=796
[INFO] Community 80: sim_idx=62, N_true=637, N_pred=637, N_overlap=637
[INFO] Community 81: sim_idx=79, N_true=578, N_pred=578, N_overlap=578
[INFO] Community 82: sim_idx=42, N_true=770, N_pred=770, N_overlap=770
[INFO] Community 83: sim_idx=56, N_true=474, N_pred=474, N_overlap=474
[INFO] Community 84: sim_idx=6, N_true=775, N_pred=775, N_overlap=775
[INFO] Community 85: sim_idx=7, N_true=494, N_pred=494, N_overlap=494
[INFO] Community 86: sim_idx=4, N_true=440, N_pred=440, N_overlap=440
[INFO] Community 87: sim_idx=80, N_true=663, N_pred=663, N_overlap=663
[INFO] Community 88: sim_idx=99, N_true=115, N_pred=115, N_overlap=115
[INFO] Community 89: sim_idx=36, N_true=696, N_pred=696, N_overlap=696
[INFO] Community 90: sim_idx=58, N_true=656, N_pred=656, N_overlap=656
[INFO] Community 91: sim_idx=85, N_true=611, N_pred=611, N_overlap=611
[INFO] Community 92: sim_idx=24, N_true=776, N_pred=776, N_overlap=776
[INFO] Community 93: sim_idx=9, N_true=688, N_pred=688, N_overlap=688
[INFO] Community 94: sim_idx=27, N_true=740, N_pred=740, N_overlap=740
[INFO] Community 95: sim_idx=29, N_true=633, N_pred=633, N_overlap=633
[INFO] Community 96: sim_idx=31, N_true=515, N_pred=515, N_overlap=515
[INFO] Community 97: sim_idx=10, N_true=550, N_pred=550, N_overlap=550
[INFO] Community 98: sim_idx=61, N_true=755, N_pred=755, N_overlap=755
[INFO] Community 99: sim_idx=67, N_true=392, N_pred=392, N_overlap=392
[RESULTS] Evaluated 100 communities.

=== Overall (all edge types, all evaluated communities) ===
  TP=105291, FP=0, FN=105305
  Precision: 1.0000
  Recall:    0.5000
  F1-score:  0.6666
  Jaccard:   0.5000

=== Edge type 0 (type 0) ===
  TP=52642, FP=0, FN=52656
  Precision: 1.0000
  Recall:    0.4999
  F1-score:  0.6666
  Jaccard:   0.4999

=== Edge type 1 (type 1) ===
  TP=52649, FP=0, FN=52649
  Precision: 1.0000
  Recall:    0.5000
  F1-score:  0.6667
  Jaccard:   0.5000

```

### 4.1. Important evaluation flags

- `--sim_npz`  
  Simulation NPZ with true graphs (`graphs`) and communities.

- `--learned_graphs_npz`  
  NPZ saved by `main_multi_real.py` when `--save_graphs` is enabled.

- `--top_k_pred`  
  To align with the generative graph, set
  \[
  \texttt{top\_k\_pred} = k_{\text{in}}
  \]
  used during simulation.  
  For the special case `k_in = 1`, use `--top_k_pred 1`.  
  The script sorts predicted edge weights per node and keeps only the top \( k_{\text{in}} \) incoming edges (plus any implied self-loop) before evaluation.

- `--transpose_pred`  
  The simulator uses the convention
  \[
  S[i,j] = 1 \ \Longrightarrow\ j \to i,
  \]
  while the learned adjacency \( W \) is often stored with the opposite orientation.  
  `--transpose_pred` ensures the predicted matrix is transposed before comparison so directions match.

- `--max_communities`, `--min_overlap`  
  Control which communities are evaluated. Typically use:
  - `--max_communities` = total number of simulated/learned communities.
  - `--min_overlap 10` to skip any degenerate overlaps (should not matter in clean simulated runs).

The script reports:

- Overall TP, FP, FN across all communities and types.
- Precision, recall, F1, and Jaccard.
- Also per-edge-type metrics when \( K \ge 2 \).

In the perfectly aligned toy case with `k_in = 1`, `top_k_pred = 1`, and `--transpose_pred`, you should see very high precision (often near 1.0) and recall around 0.5–1.0, confirming that the learner recovers the true graph structure reasonably well.

---

## 5. Parameter sweep strategy

This section describes how to systematically sweep parameters to study both **prediction** and **graph recovery** performance.

### 5.1. Parameters to sweep in simulation (`simulate_from_real_npz.py`)

1. **Graph sparsity**: `k_in`
   - Values: `1, 2, 4`
   - Interpretation: sparser graphs are easier to recover, denser graphs produce more complex interactions.

2. **Hawkes signal strength**: `alpha_self`, `alpha_neigh`, `decay`
   - Graph-dominated regime (good for graph recovery):
     - `alpha_self in {0.5, 1.0}`
     - `alpha_neigh in {2.0, 3.0, 4.0}`
     - `decay in {0.5, 0.7}`
   - Balanced regime:
     - `alpha_self = 1.0`
     - `alpha_neigh = 1.0–2.0`
   - Self-dominated regime (negative control):
     - `alpha_self = 3.0`
     - `alpha_neigh = 0.5–1.0`

3. **Base rate**: `base_rate`
   - Values: `1e-4, 3e-4, 1e-3`
   - Lower base rate ⇒ events more driven by the graph and Hawkes dynamics.

4. **Time horizon**: `T`
   - Values: `24, 36, 48`
   - Longer sequences provide more information for both prediction and graph recovery.

5. **Community size & count**: `min_nodes`, `max_nodes`, `max_communities`
   - Toy setting (sanity-check):
     - `min_nodes=40`, `max_nodes=80`, `max_communities=4–10`
   - Medium/realistic:
     - `min_nodes=40`, `max_nodes=800`, `max_communities=50–100`

### 5.2. Parameters to sweep in training (`main_multi_real.py`)

1. **Graph sparsity penalty**: `lambda_edge`
   - Primary knob to trade off fit vs. sparsity.
   - Suggested sweep (log-scale):
     \[
     \lambda_{\text{edge}} \in \{0.1, 0.5, 1.0, 2.0, 3.0\}
     \]

2. **Learning schedule**: `num_epochs`, `lr`
   - Typically keep fixed:
     - `num_epochs = 200` (or 300 if needed)
     - `lr = 5e-4`

3. **Community filtering**: `min_nodes_real`, `max_nodes_real`
   - Match the simulator:
     - If `min_nodes=40`, `max_nodes=800` during simulation, use:
       - `--min_nodes_real 40 --max_nodes_real 800`

4. **Label configuration**: `label_mode`, `horizon_months`, `alpha_window`
   - Fix to the configuration you will report in the paper:
     - `--label_mode both`
     - `--horizon_months 6`
     - `--alpha_window 0.9`

### 5.3. Parameters to set in evaluation (`eval_graph_recovery.py`)

1. **Alignment with generative graph**:
   - `--top_k_pred k_in`  
     Always set `top_k_pred` equal to the `k_in` used in simulation.
   - `--transpose_pred`  
     Ensure direction conventions match.

2. **Community inclusion**:
   - `--max_communities` = number of learned communities.
   - `--min_overlap 10` (or similar).

### 5.4. Example sweep loop (conceptual)

**Step 1 – sweep simulation parameters:**

```bash
for k_in in 1 2 4; do
  for alpha_neigh in 1.0 2.0 3.0; do
    for base_rate in 1e-4 3e-4 1e-3; do

      python simulate_from_real_npz.py \
        --in_real_npz lee_ian_by_cbg_2state_v2_dr0.005_tr0.6.npz \
        --out_sim_npz lee_ian_sim_exp.npz \
        --T 48 \
        --seed 123 \
        --k_in $k_in \
        --alpha_self 1.0 \
        --alpha_neigh $alpha_neigh \
        --decay 0.5 \
        --base_rate $base_rate \
        --min_nodes 40 \
        --max_nodes 800 \
        --max_communities 100

      # This produces a parameter-encoded NPZ file, e.g.:
      # lee_ian_sim_exp_k${k_in}_as1p0_an${...}_d0p5_br${...}_T48_seed123_mn40_mx800_GcXX.npz

    done
  done
done
```

**Step 2 – for each simulated NPZ, sweep `lambda_edge` in training:**

```bash
REAL_NPZ=lee_ian_sim_toy2_k1_as1p0_an3p0_d0p5_br0p001_T48_seed123_mn40_mx800_Gc100.npz

for lambda_edge in 0.1 0.5 1.0 2.0 3.0; do
  python main_multi_real.py \
    --real_npz $REAL_NPZ \
    --max_communities 200 \
    --ref_lon -81.8723 \
    --ref_lat 26.6406 \
    --max_nodes_real 800 \
    --min_nodes_real 40 \
    --num_epochs 200 \
    --lr 5e-4 \
    --label_mode both \
    --horizon_months 6 \
    --alpha_window 0.9 \
    --lambda_edge $lambda_edge \
    --save_graphs
done
```

**Step 3 – evaluate graph recovery for each `lambda_edge`:**

```bash
for lambda_edge in 0.1 0.5 1.0 2.0 3.0; do
  python eval_graph_recovery.py \
    --sim_npz $REAL_NPZ \
    --learned_graphs_npz ${REAL_NPZ%.npz}_lambda${lambda_edge/./p}_learned_graphs.npz \
    --max_communities 200 \
    --min_overlap 10 \
    --top_k_pred 1 \
    --transpose_pred
done
```

You can then collect:

- **Prediction metrics** from `main_multi_real.py` output:
  - Tail AUC / AP and window AUC / AP, per type.
- **Graph recovery metrics** from `eval_graph_recovery.py`:
  - Precision, recall, F1, Jaccard.

Plotting these as a function of `lambda_edge` (and across simulation regimes) will clearly show:

- In which regimes the model can **accurately predict future events**.
- In which regimes it simultaneously **recovers the underlying graph** with good precision/recall.

---

## 6. Summary

- Real-data pipeline (`prepare_real_ian_by_cbg.py`) gives you **realistic node features and communities**.
- Simulation pipeline (`simulate_from_real_npz.py`) gives you **ground-truth graphs and Hawkes events** with controllable sparsity and signal strength.
- Training (`main_multi_real.py`) measures **forecasting quality**.
- Evaluation (`eval_graph_recovery.py`) measures **graph recovery** when properly aligned (top-k and transpose).

This setup allows you to design **clean, controlled simulation experiments** that demonstrate both:

1. The ability of your Graph Hawkes model to forecast future events, and  
2. Its ability to reconstruct the underlying interaction graph from observed cascades.
