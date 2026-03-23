import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--graphs_npz", default="data/processed/lee_ian_by_cbg_2state_dr0.005_tr0.6_learned_graphs.npz")
parser.add_argument("--parcel_csv", default="data/raw/fl_lee.csv")
args = parser.parse_args()

# Load the graphs
graphs_npz = np.load(args.graphs_npz, allow_pickle=True)
graphs = graphs_npz["graphs"]

# Load the parcel CSV
parcel_df = pd.read_csv(args.parcel_csv)
parcel_df = parcel_df.reset_index().rename(columns={"index": "node_id"})

g0 = graphs[0].item()
node_ids = g0["node_ids"]
coords = g0["coords"]
A = g0["A"]  # [K, N_g, N_g]

# Join to get parcel identifiers
sub_parcels = parcel_df.loc[node_ids, ["node_id", "strap", "parcelnumb", "lat", "lon"]]
print(sub_parcels.head())
