# prepare_real_ian_by_cbg.py
#
# Build real-data communities for Lee County, FL, around Hurricane Ian.
#
# Inputs:
#   - parcel_csv: fl_lee.csv
#   - sales_csv:  sales_data.csv
#   - repair_csv: repair_data.csv
#   - damage_csv: hurricane_ian_damage_map.csv
#
# Output:
#   - NPZ with:
#       communities: np.array([...], dtype=object)
#         each element is dict with keys:
#           'cbg'      : str   (census blockgroup id)
#           'node_ids' : np.ndarray [N_g]
#           'coords'   : np.ndarray [N_g,2]  (lon, lat)
#           'X'        : np.ndarray [N_g,d]  (features)
#           'Y'        : np.ndarray [T,N_g,3]  (sell, repair, vacate)
#       meta: dict with keys:
#           'T', 'T_train', 'T_val', 'K', 'train_ratio',
#           'ian_date', 'start_date', 'end_date'
#       time_index: np.ndarray [T] of month-start timestamps
#       X_all: np.ndarray [N,d]  (global feature matrix)
#       Y: np.ndarray [T,N,3]  (full grid, all parcels)
#       node_ids: np.ndarray [N] == np.arange(N)
#
# Notes:
#   - All parcels are kept as nodes; communities may be filtered by min_cbg_events.
#   - Vacancy events use build_vacancy_events_from_usps with stricter "vacant" logic.

import argparse
import os
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree


# -----------------------------
# Utilities
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--parcel_csv", type=str, required=True)
    parser.add_argument("--sales_csv", type=str, required=True)
    parser.add_argument("--repair_csv", type=str, required=True)
    parser.add_argument("--damage_csv", type=str, required=True)
    parser.add_argument("--out_npz", type=str, required=True)

    parser.add_argument(
        "--ian_date",
        type=str,
        default="2022-09-28",
        help="Landfall date of Hurricane Ian (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="Optional end date (YYYY-MM-DD). If None, inferred from events.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.6,
        help="Fraction of time steps used for training (rest for validation).",
    )
    parser.add_argument(
        "--damage_radius_deg",
        type=float,
        default=0.02,
        help="Max distance in degrees when mapping Ian damage points to parcels.",
    )
    parser.add_argument(
        "--map_radius_deg",
        type=float,
        default=0.02,
        help="Max distance in degrees when mapping sales/repairs points to parcels.",
    )
    parser.add_argument(
        "--min_cbg_events",
        type=int,
        default=1,
        help="Minimum total events in a CBG community to keep it.",
    )

    args = parser.parse_args()
    print("Args:", args)
    return args


def parse_date_series(s: pd.Series) -> pd.Series:
    """Robust datetime parser."""
    return pd.to_datetime(s, errors="coerce", utc=False)


# -----------------------------
# Attach Ian damage to parcels
# -----------------------------

def attach_ian_damage(
    parcel_df: pd.DataFrame,
    damage_df: pd.DataFrame,
    radius_deg: float = 0.02,
    lat_col_parcel: str = "lat",
    lon_col_parcel: str = "lon",
) -> pd.DataFrame:
    """
    Map each parcel to the nearest Ian damage point within radius_deg, and attach:
      - ian_FloodDepth
      - ian_BldgValue
      - ian_EstLoss
      - ian_Occupancy
      - ian_DamageLevel
    """

    dmg = damage_df.copy()

    # Numeric lat/lon for damage
    dmg_lat = pd.to_numeric(dmg.get("Latitude"), errors="coerce")
    dmg_lon = pd.to_numeric(dmg.get("Longitude"), errors="coerce")
    mask_valid = dmg_lat.notna() & dmg_lon.notna()
    dmg = dmg.loc[mask_valid].reset_index(drop=True)
    dmg_lat = dmg_lat.loc[mask_valid].reset_index(drop=True)
    dmg_lon = dmg_lon.loc[mask_valid].reset_index(drop=True)

    coords_damage = np.column_stack([dmg_lon.values, dmg_lat.values])
    print(f"Loaded damage_df with shape: {damage_df.shape}")
    print(f"Damage rows with valid coords: {coords_damage.shape[0]}")

    if coords_damage.shape[0] == 0:
        print("[WARN] No valid damage coordinates; skipping Ian damage attachment.")
        for col in ["ian_FloodDepth", "ian_BldgValue", "ian_EstLoss"]:
            if col not in parcel_df.columns:
                parcel_df[col] = np.nan
        for col in ["ian_Occupancy", "ian_DamageLevel"]:
            if col not in parcel_df.columns:
                parcel_df[col] = pd.Series([None] * len(parcel_df), dtype=object)
        return parcel_df

    # KDTree on damage points
    tree = KDTree(coords_damage)

    # Parcel coordinates
    p_lat = pd.to_numeric(parcel_df[lat_col_parcel], errors="coerce")
    p_lon = pd.to_numeric(parcel_df[lon_col_parcel], errors="coerce")
    coords_parcel = np.column_stack([p_lon.values, p_lat.values])

    dist, ind = tree.query(coords_parcel, k=1)
    dist = dist.reshape(-1)
    ind = ind.reshape(-1)

    matched_mask = np.isfinite(dist) & (dist <= radius_deg)
    n_matched = matched_mask.sum()
    print(f"Parcels matched to at least one Ian damage point: {n_matched}")

    # Ensure columns exist with correct dtypes
    for col in ["ian_FloodDepth", "ian_BldgValue", "ian_EstLoss"]:
        if col not in parcel_df.columns:
            parcel_df[col] = np.nan
        parcel_df[col] = pd.to_numeric(parcel_df[col], errors="coerce")

    for col in ["ian_Occupancy", "ian_DamageLevel"]:
        if col not in parcel_df.columns:
            parcel_df[col] = pd.Series([None] * len(parcel_df), dtype=object)
        else:
            parcel_df[col] = parcel_df[col].astype(object)

    # Vectorized assignment for matched parcels
    parcel_idx = np.where(matched_mask)[0]
    dmg_idx = ind[matched_mask]

    # Numeric damage fields
    for src, dst in [
        ("FloodDepth", "ian_FloodDepth"),
        ("BldgValue", "ian_BldgValue"),
        ("EstLoss", "ian_EstLoss"),
    ]:
        if src in dmg.columns:
            vals = pd.to_numeric(dmg[src], errors="coerce").values
            parcel_df.loc[parcel_idx, dst] = vals[dmg_idx]
        else:
            print(f"[WARN] Column {src} not found in damage_df; {dst} will remain NaN.")

    # Categorical damage fields
    if "Occupancy" in dmg.columns:
        occ_vals = dmg["Occupancy"].astype(object).values
        parcel_df.loc[parcel_idx, "ian_Occupancy"] = occ_vals[dmg_idx]
    else:
        print("[WARN] Occupancy not found in damage_df.")

    if "DamageLevel" in dmg.columns:
        dl_vals = dmg["DamageLevel"].astype(object).values
        parcel_df.loc[parcel_idx, "ian_DamageLevel"] = dl_vals[dmg_idx]
    else:
        print("[WARN] DamageLevel not found in damage_df.")

    return parcel_df


# -----------------------------
# Sales extraction
# -----------------------------

def extract_parcel_sales_from_saledate(parcel_df: pd.DataFrame, ian_date: str) -> pd.DataFrame:
    """
    Use saledate, saledate2, saledate3, saledate4 in fl_lee.csv to infer
    first post-Ian sale per parcel.
    Returns DataFrame ['node_id', 'sale_ts'] with at most one row per node.
    """
    ian_dt = pd.to_datetime(ian_date)

    sale_cols = ["saledate", "saledate2", "saledate3", "saledate4"]
    sale_dates = {}
    for c in sale_cols:
        if c in parcel_df.columns:
            sale_dates[c] = parse_date_series(parcel_df[c])
        else:
            sale_dates[c] = pd.Series(pd.NaT, index=parcel_df.index)

    sale_df = pd.DataFrame(sale_dates)
    for c in sale_cols:
        sale_df[c] = sale_df[c].where(sale_df[c] >= ian_dt)

    sale_ts = sale_df.min(axis=1)  # row-wise min of post-Ian dates
    mask = sale_ts.notna()
    node_ids = parcel_df.loc[mask, "node_id"].astype(int).values
    sale_ts = sale_ts.loc[mask]

    df = pd.DataFrame({"node_id": node_ids, "sale_ts": sale_ts.values})
    print(f"Parcel-saledate based sales events: {df.shape[0]}")
    return df


def map_points_to_parcels(
    points_df: pd.DataFrame,
    parcel_df: pd.DataFrame,
    lat_col_pts: str,
    lon_col_pts: str,
    ts_col: str,
    map_radius_deg: float,
) -> pd.DataFrame:
    """
    Generic helper: map point-based events (lon, lat, timestamp) to nearest parcel within map_radius_deg.
    Returns DataFrame ['node_id', '<ts_col_out>'] with node_id referencing parcel_df['node_id'].
    """
    pts = points_df.copy()

    pts_lat = pd.to_numeric(pts[lat_col_pts], errors="coerce")
    pts_lon = pd.to_numeric(pts[lon_col_pts], errors="coerce")
    pts_ts = parse_date_series(pts[ts_col])

    mask_valid = pts_lat.notna() & pts_lon.notna() & pts_ts.notna()
    pts_lat = pts_lat.loc[mask_valid].reset_index(drop=True)
    pts_lon = pts_lon.loc[mask_valid].reset_index(drop=True)
    pts_ts = pts_ts.loc[mask_valid].reset_index(drop=True)

    coords_pts = np.column_stack([pts_lon.values, pts_lat.values])
    if coords_pts.shape[0] == 0:
        return pd.DataFrame(columns=["node_id", ts_col])

    # Parcel KDTree
    p_lat = pd.to_numeric(parcel_df["lat"], errors="coerce")
    p_lon = pd.to_numeric(parcel_df["lon"], errors="coerce")
    coords_parcel = np.column_stack([p_lon.values, p_lat.values])

    tree = KDTree(coords_parcel)
    dist, ind = tree.query(coords_pts, k=1)
    dist = dist.reshape(-1)
    ind = ind.reshape(-1)

    matched_mask = np.isfinite(dist) & (dist <= map_radius_deg)
    pts_idx = np.where(matched_mask)[0]
    if len(pts_idx) == 0:
        return pd.DataFrame(columns=["node_id", ts_col])

    node_ids = parcel_df.loc[ind[matched_mask], "node_id"].astype(int).values
    ts_vals = pts_ts.loc[pts_idx].values

    df = pd.DataFrame({"node_id": node_ids, ts_col: ts_vals})
    return df


def extract_sales_from_points(
    sales_df: pd.DataFrame,
    parcel_df: pd.DataFrame,
    ian_date: str,
    map_radius_deg: float,
) -> pd.DataFrame:
    """
    Map point-based sales (sales_data.csv) to parcels.
    Uses columns: 'lon', 'lat', 'first_sale_in_period'.
    Returns ['node_id', 'sale_ts'].
    """
    if not {"lon", "lat", "first_sale_in_period"}.issubset(set(sales_df.columns)):
        print("[WARN] sales_data.csv missing {lon,lat,first_sale_in_period}; no point-based sales.")
        return pd.DataFrame(columns=["node_id", "sale_ts"])

    # Map to parcels
    mapped = map_points_to_parcels(
        sales_df,
        parcel_df,
        lat_col_pts="lat",
        lon_col_pts="lon",
        ts_col="first_sale_in_period",
        map_radius_deg=map_radius_deg,
    )
    if mapped.empty:
        print("Point-based sales events mapped to parcels: 0")
        return pd.DataFrame(columns=["node_id", "sale_ts"])

    # Filter to post-Ian
    ian_dt = pd.to_datetime(ian_date)
    mapped["sale_ts"] = parse_date_series(mapped["first_sale_in_period"])
    mapped = mapped.drop(columns=["first_sale_in_period"])
    mapped = mapped[mapped["sale_ts"] >= ian_dt]

    print(f"Point-based sales events mapped to parcels: {mapped.shape[0]}")
    return mapped[["node_id", "sale_ts"]]


def combine_sales(parcel_sales: pd.DataFrame, point_sales: pd.DataFrame) -> pd.DataFrame:
    """
    Combine parcel-based and point-based sales; keep earliest sale per node_id.
    """
    all_sales = pd.concat([parcel_sales, point_sales], ignore_index=True, sort=False)
    all_sales = all_sales.dropna(subset=["sale_ts"])

    if all_sales.empty:
        print("[WARN] No sales events found.")
        return pd.DataFrame(columns=["node_id", "sale_ts"])

    all_sales["sale_ts"] = parse_date_series(all_sales["sale_ts"])
    all_sales = all_sales.dropna(subset=["sale_ts"])

    # Earliest sale per node
    all_sales = all_sales.sort_values(["node_id", "sale_ts"])
    all_sales = all_sales.groupby("node_id", as_index=False)["sale_ts"].min()

    print(f"Total sales events (any source): {all_sales.shape[0]}")
    return all_sales


# -----------------------------
# Repair extraction
# -----------------------------

def extract_repairs_from_points(
    repair_df: pd.DataFrame,
    parcel_df: pd.DataFrame,
    ian_date: str,
    map_radius_deg: float,
) -> pd.DataFrame:
    """
    Map point-based repair events (repair_data.csv) to parcels.
    Assumes columns: lon, lat, record_date.
    Returns ['node_id', 'repair_ts'].
    """
    cols_needed = {"lon", "lat", "record_date"}
    if not cols_needed.issubset(set(repair_df.columns)):
        print("[WARN] repair_data.csv missing {lon,lat,record_date}; no point-based repairs.")
        return pd.DataFrame(columns=["node_id", "repair_ts"])

    mapped = map_points_to_parcels(
        repair_df,
        parcel_df,
        lat_col_pts="lat",
        lon_col_pts="lon",
        ts_col="record_date",
        map_radius_deg=map_radius_deg,
    )
    if mapped.empty:
        print("Repair events mapped to parcels (from repair_data.csv): 0")
        return pd.DataFrame(columns=["node_id", "repair_ts"])

    ian_dt = pd.to_datetime(ian_date)
    mapped["repair_ts"] = parse_date_series(mapped["record_date"])
    mapped = mapped.drop(columns=["record_date"])
    mapped = mapped[mapped["repair_ts"] >= ian_dt]

    print(f"Repair events mapped to parcels (from repair_data.csv): {mapped.shape[0]}")
    return mapped[["node_id", "repair_ts"]]


def extract_parcel_repairs_from_fl_lee(parcel_df: pd.DataFrame, ian_date: str) -> pd.DataFrame:
    """
    Infer repair events from fl_lee.csv using reviseddate/maintdate + new_* fields.
    Returns ['node_id', 'repair_ts'].
    """
    ian_dt = pd.to_datetime(ian_date)

    # Candidate dates
    rev = parse_date_series(parcel_df.get("reviseddate", pd.Series(index=parcel_df.index)))
    mnt = parse_date_series(parcel_df.get("maintdate", pd.Series(index=parcel_df.index)))
    # choose earliest non-null
    repair_ts = rev.combine_first(mnt)

    # Indicators of structural change / new construction
    def _get(col):
        return pd.to_numeric(parcel_df.get(col, 0), errors="coerce").fillna(0)

    new_heated = _get("new_heatedarea")
    new_total = _get("new_totalarea")
    new_bath = _get("new_bathrooms")
    new_bed = _get("new_bedrooms")

    mask_new = (new_heated > 0) | (new_total > 0) | (new_bath > 0) | (new_bed > 0)
    mask = mask_new & repair_ts.notna() & (repair_ts >= ian_dt)

    df = pd.DataFrame(
        {
            "node_id": parcel_df.loc[mask, "node_id"].astype(int).values,
            "repair_ts": repair_ts.loc[mask].values,
        }
    )

    print(f"Parcel-based inferred repairs (post-Ian, from fl_lee.csv): {df.shape[0]}")
    return df


def combine_repairs(point_repairs: pd.DataFrame, parcel_repairs: pd.DataFrame) -> pd.DataFrame:
    """
    Combine point-based and parcel-based repairs; keep earliest repair per node_id.
    """
    all_rep = pd.concat([point_repairs, parcel_repairs], ignore_index=True, sort=False)
    all_rep = all_rep.dropna(subset=["repair_ts"])

    if all_rep.empty:
        print("[WARN] No repair events found.")
        return pd.DataFrame(columns=["node_id", "repair_ts"])

    all_rep["repair_ts"] = parse_date_series(all_rep["repair_ts"])
    all_rep = all_rep.dropna(subset=["repair_ts"])

    all_rep = all_rep.sort_values(["node_id", "repair_ts"])
    all_rep = all_rep.groupby("node_id", as_index=False)["repair_ts"].min()

    print(f"Total repair events (point + parcel-based): {all_rep.shape[0]}")
    return all_rep


# # -----------------------------
# # Vacancy events
# # -----------------------------

# def build_vacancy_events_from_usps(parcel_df: pd.DataFrame, ian_date: str) -> pd.DataFrame:
#     """
#     Use USPS vacancy flag + date to define first vacancy event per parcel after Ian.
#     Returns DataFrame ['node_id','vacate_ts'].

#     Logic:
#       - Normalize usps_vacancy to lowercase string.
#       - Keep rows where vacancy flag ∈ VACANT_VALUES and usps_vacancy_date >= Ian.
#       - One vacate event per node (earliest date).
#     """
#     if ("usps_vacancy" not in parcel_df.columns) or ("usps_vacancy_date" not in parcel_df.columns):
#         print("[WARN] usps_vacancy/usps_vacancy_date not found; no vacancy events.")
#         return pd.DataFrame(columns=["node_id", "vacate_ts"])


#     df = parcel_df[["node_id", "usps_vacancy", "usps_vacancy_date"]].copy()

#     # Normalize vacancy flag
#     def _norm_flag(x):
#         if pd.isna(x):
#             return ""
#         return str(x).strip().lower()

#     df["usps_vacancy_norm"] = df["usps_vacancy"].apply(_norm_flag)
#     print("Unique usps_vacancy_norm values:", df["usps_vacancy_norm"].value_counts())

#     # You may adjust this set after inspecting value_counts in your own notebook.
#     VACANT_VALUES = {
#         "y",
#         "yes",
#         "vacant",
#         "v",
#         "1",
#         "true",
#         "t",
#         "occupied to vacant",  # just in case there are descriptive strings
#     }

#     df["vacate_ts"] = parse_date_series(df["usps_vacancy_date"])
#     ian_dt = pd.to_datetime(ian_date)

#     mask = df["vacate_ts"].notna() & (df["vacate_ts"] >= ian_dt) & df["usps_vacancy_norm"].isin(VACANT_VALUES)
#     df = df.loc[mask]

#     if df.empty:
#         print("[INFO] No post-Ian USPS vacancy events after applying flag+date logic.")
#         return pd.DataFrame(columns=["node_id", "vacate_ts"])

#     df = df.sort_values(["node_id", "vacate_ts"])
#     df = df.groupby("node_id", as_index=False)["vacate_ts"].min()

#     print(f"Vacancy events from USPS (flag+date): {df.shape[0]}")
#     return df[["node_id", "vacate_ts"]]


# -----------------------------
# Time index & binning
# -----------------------------

def build_time_index(
    ian_date: str,
    sales: pd.DataFrame,
    repairs: pd.DataFrame,
    # vacates: pd.DataFrame,
    end_date: str = None,
) -> Tuple[pd.DatetimeIndex, pd.Timestamp, pd.Timestamp]:
    """
    Build monthly time index [T] from:
      start_date = month start of Ian's month
      end_date   = if provided, its month start; else max event month
    """
    ian_dt = pd.to_datetime(ian_date)
    start_date = ian_dt.to_period("M").to_timestamp()  # month start

    # Gather all event timestamps
    ts_all = []
    if not sales.empty:
        ts_all.append(parse_date_series(sales["sale_ts"]))
    if not repairs.empty:
        ts_all.append(parse_date_series(repairs["repair_ts"]))
    # if not vacates.empty:
    #     ts_all.append(parse_date_series(vacates["vacate_ts"]))

    if end_date is not None:
        end_dt = pd.to_datetime(end_date)
        end_month_start = end_dt.to_period("M").to_timestamp()
    else:
        if len(ts_all) == 0:
            # fallback horizon: 24 months after Ian
            end_month_start = start_date + pd.DateOffset(months=24)
        else:
            all_ts = pd.concat(ts_all)
            all_ts = all_ts.dropna()
            if all_ts.empty:
                end_month_start = start_date + pd.DateOffset(months=24)
            else:
                max_ts = all_ts.max()
                end_month_start = max_ts.to_period("M").to_timestamp()

    time_index = pd.date_range(start=start_date, end=end_month_start, freq="MS")
    print(f"Time index from {time_index[0].date()} to {time_index[-1].date()} (T={len(time_index)})")
    return time_index, start_date, end_month_start


def bin_events_to_month_indices(
    events_df: pd.DataFrame,
    ts_col: str,
    node_col: str,
    time_index: pd.DatetimeIndex,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    label: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert event timestamps to 0..T-1 month indices, aligned with time_index.
    Returns (node_ids, t_idx) as 1D np.int arrays of equal length.
    """
    if events_df.empty:
        print(f"[DEBUG] bin_events_to_month_indices({label}): no events.")
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    ts = parse_date_series(events_df[ts_col])
    nodes = events_df[node_col].astype(int).values

    mask = ts.notna() & (ts >= start_date) & (ts <= end_date)
    ts = ts.loc[mask]
    nodes = nodes[mask.values]

    if len(ts) == 0:
        print(f"[DEBUG] bin_events_to_month_indices({label}): no events in [{start_date.date()}, {end_date.date()}].")
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    month_start = ts.dt.to_period("M").dt.to_timestamp()
    mapping: Dict[pd.Timestamp, int] = {d: i for i, d in enumerate(time_index)}

    t_idx = month_start.map(mapping)
    valid_mask = t_idx.notna()
    t_idx = t_idx.loc[valid_mask].astype(int).values
    nodes = nodes[valid_mask.values]

    if len(t_idx) == 0:
        print(f"[DEBUG] bin_events_to_month_indices({label}): all events fell outside time_index.")
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    print(
        f"[DEBUG] bin_events_to_month_indices({label}): kept {len(t_idx)} events in [0,{len(time_index)}) "
        f", t_idx min={t_idx.min()}, max={t_idx.max()}"
    )
    return nodes, t_idx


# -----------------------------
# Feature encoding
# -----------------------------

def encode_categorical_column(s: pd.Series) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Factorize a categorical series to integer codes with UNK.
    Returns (codes, mapping dict).
    """
    s = s.fillna("UNK").astype(str)
    codes, uniques = pd.factorize(s, sort=True)
    mapping = {str(u): int(i) for i, u in enumerate(uniques)}
    return codes.astype("int32"), mapping


def build_feature_matrix(parcel_df: pd.DataFrame) -> Tuple[np.ndarray, List[str], Dict[str, Dict[str, int]]]:
    """
    Build numeric + categorical feature matrix X_all [N,d].
    Returns:
      X_all: [N,d] float32
      numeric_features: list of numeric feature names
      cat_mappings: mapping per categorical feature
    """
    numeric_features = [
        "lat",
        "lon",
        "yearbuilt",
        "improvval",
        "landval",
        "parval",
        "agval",
        "sqft",
        "totalarea",
        "existing_heatedarea",
        "existing_bathrooms",
        "existing_bedrooms",
        "new_heatedarea",
        "new_totalarea",
        "new_bathrooms",
        "new_bedrooms",
        "highest_parcel_elevation",
        "lowest_parcel_elevation",
        "population_density",
        "population_growth_past_5_years",
        "population_growth_next_5_years",
        "housing_growth_past_5_years",
        "housing_growth_next_5_years",
        "household_income_growth_next_5_years",
        "median_household_income",
        "fema_nri_risk_rating",
        "housing_affordability_index",
        "transmission_line_distance",
        "roughness_rating",
        "ian_FloodDepth",
        "ian_BldgValue",
        "ian_EstLoss",
    ]

    for f in numeric_features:
        if f not in parcel_df.columns:
            parcel_df[f] = np.nan

    X_num = parcel_df[numeric_features].apply(pd.to_numeric, errors="coerce").astype("float32").fillna(0.0).values
    print(f"Numeric feature matrix X_num shape: {X_num.shape}")
    print("Using numeric features:", numeric_features)

    cat_features = [
        "usecode",
        "zoning",
        "lbcs_activity",
        "lbcs_function",
        "lbcs_structure",
        "owntype",
        # "usps_vacancy",
        "fema_flood_zone",
        "ian_Occupancy",
        "ian_DamageLevel",
    ]

    X_cat_list = []
    cat_mappings: Dict[str, Dict[str, int]] = {}
    for col in cat_features:
        if col not in parcel_df.columns:
            parcel_df[col] = "UNK"
        codes, mapping = encode_categorical_column(parcel_df[col])
        X_cat_list.append(codes.reshape(-1, 1))
        cat_mappings[col] = mapping
        print(f"Categorical feature '{col}' -> {len(mapping)} categories (incl. UNK)")

    if len(X_cat_list) > 0:
        X_cat = np.concatenate(X_cat_list, axis=1).astype("float32")
        print(f"Categorical code matrix X_cat shape: {X_cat.shape}")
        X_all = np.concatenate([X_num, X_cat], axis=1).astype("float32")
    else:
        X_cat = np.zeros((X_num.shape[0], 0), dtype="float32")
        X_all = X_num

    print(f"Final X feature matrix shape: {X_all.shape}")
    return X_all, numeric_features, cat_mappings


# -----------------------------
# Community building
# -----------------------------

def build_communities(
    parcel_df: pd.DataFrame,
    Y: np.ndarray,
    X_all: np.ndarray,
    min_cbg_events: int,
) -> List[Dict]:
    """
    Build communities based on census_blockgroup.
    Each community dict has:
      - 'cbg', 'node_ids', 'coords', 'X', 'Y'
    """
    if "census_blockgroup" not in parcel_df.columns:
        raise ValueError("census_blockgroup column is required in parcel_df.")

    N = parcel_df.shape[0]
    T = Y.shape[0]

    node_ids = parcel_df["node_id"].astype(int).values
    cbg_vals = parcel_df["census_blockgroup"].astype(str).values
    coords_all = parcel_df[["lon", "lat"]].apply(pd.to_numeric, errors="coerce").values

    cbg_to_nodes: Dict[str, List[int]] = {}
    for nid, cbg in zip(node_ids, cbg_vals):
        cbg_to_nodes.setdefault(cbg, []).append(int(nid))

    communities: List[Dict] = []
    total_events_kept = np.zeros(2, dtype=np.float64)
    total_events_all = Y.sum(axis=(0, 1))
    print(f"CBGs with any nodes: {len(cbg_to_nodes)}")
    print("Total events across all parcels (sell, repair):", total_events_all.tolist())

    for cbg, nodes in cbg_to_nodes.items():
        nodes_arr = np.array(nodes, dtype=int)
        Y_g = Y[:, nodes_arr, :]  # [T, N_g, 3]
        total_events = Y_g.sum()
        if total_events < min_cbg_events:
            continue

        coords_g = coords_all[nodes_arr, :]  # [N_g,2]
        X_g = X_all[nodes_arr, :]  # [N_g,d]

        communities.append(
            {
                "cbg": cbg,
                "node_ids": nodes_arr,
                "coords": coords_g,
                "X": X_g,
                "Y": Y_g,
            }
        )
        total_events_kept += Y_g.sum(axis=(0, 1))

    print(f"CBGs with at least {min_cbg_events} events: {len(communities)}")
    print("Total events across kept communities (sell, repair):", total_events_kept.tolist())
    return communities


# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()

    # 1) Load parcels
    parcel_df = pd.read_csv(args.parcel_csv, low_memory=False)
    print(f"Loaded parcel_df with shape: {parcel_df.shape}")

    # Basic cleaning: keep rows with lat/lon and census_blockgroup
    parcel_df["lat"] = pd.to_numeric(parcel_df["lat"], errors="coerce")
    parcel_df["lon"] = pd.to_numeric(parcel_df["lon"], errors="coerce")
    mask = parcel_df["lat"].notna() & parcel_df["lon"].notna() & parcel_df["census_blockgroup"].notna()
    parcel_df = parcel_df.loc[mask].reset_index(drop=True)
    print(f"After cleaning, parcels with coords and CBG: {parcel_df.shape[0]}")

    # Assign node_id = row index
    parcel_df["node_id"] = np.arange(len(parcel_df), dtype=int)

    # 2) Attach Ian damage
    damage_df = pd.read_csv(args.damage_csv)
    parcel_df = attach_ian_damage(parcel_df, damage_df, radius_deg=args.damage_radius_deg)

    # 3) Sales: parcel-based & point-based
    parcel_sales = extract_parcel_sales_from_saledate(parcel_df, args.ian_date)
    sales_df = pd.read_csv(args.sales_csv)
    point_sales = extract_sales_from_points(sales_df, parcel_df, args.ian_date, args.map_radius_deg)
    sales_all = combine_sales(parcel_sales, point_sales)

    # 4) Repairs: repair_data.csv + parcel-based from fl_lee.csv
    repair_df = pd.read_csv(args.repair_csv)
    point_repairs = extract_repairs_from_points(repair_df, parcel_df, args.ian_date, args.map_radius_deg)
    parcel_repairs = extract_parcel_repairs_from_fl_lee(parcel_df, args.ian_date)
    repairs_all = combine_repairs(point_repairs, parcel_repairs)

    # # 5) Vacancy events from USPS
    # vacates_all = build_vacancy_events_from_usps(parcel_df, args.ian_date)
    # n_parcels = parcel_df["node_id"].nunique()
    # n_vac = vacates_all["node_id"].nunique()
    # print("Fraction of parcels with vacate event:", n_vac / n_parcels)

    # 6) Build time index
    time_index, start_date, end_date = build_time_index(
        args.ian_date, sales_all, repairs_all, args.end_date
    )
    T = len(time_index)

    # 7) Bin events to month indices
    sales_nodes, sales_t_idx = bin_events_to_month_indices(
        sales_all, "sale_ts", "node_id", time_index, start_date, end_date, label="sales"
    )
    repairs_nodes, repairs_t_idx = bin_events_to_month_indices(
        repairs_all, "repair_ts", "node_id", time_index, start_date, end_date, label="repairs"
    )
    # vac_nodes, vac_t_idx = bin_events_to_month_indices(
    #     vacates_all, "vacate_ts", "node_id", time_index, start_date, end_date, label="vacates"
    # )

    # 8) Build Y[T, N, 3] over all parcels
    N = parcel_df.shape[0]
    K = 2  # sell, repair
    Y = np.zeros((T, N, K), dtype=np.int16)

    for t, n in zip(sales_t_idx, sales_nodes):
        Y[t, n, 0] += 1
    for t, n in zip(repairs_t_idx, repairs_nodes):
        Y[t, n, 1] += 1
    # for t, n in zip(vac_t_idx, vac_nodes):
    #     Y[t, n, 2] += 1

    total_events = Y.sum(axis=(0, 1))
    print("Total events per type in Y (sell, repair):", total_events.tolist())

    # 9) Build feature matrix X_all for all parcels
    X_all, numeric_features, cat_mappings = build_feature_matrix(parcel_df)

    # 10) Build communities by CBG
    communities = build_communities(parcel_df, Y, X_all, min_cbg_events=args.min_cbg_events)
    print(f"Built {len(communities)} communities.")

    # 11) Train/val split in time
    T_train = int(np.floor(args.train_ratio * T))
    T_val = T - T_train
    print(f"T_train={T_train}, T_val={T_val} (train_ratio={args.train_ratio})")


    K = 2  # sell, repair
    names = ["sell", "repair"]

    # Reconstruct the actual monthly dates used for the time index
    start_date = pd.to_datetime("2022-09-01")  # same as you used to build time_index
    dates = [start_date + pd.DateOffset(months=t) for t in range(T)]

    # Aggregate counts per (t, type) over all communities
    per_t_counts = np.zeros((T, K), dtype=np.int64)
    for comm in communities:
        Y = comm["Y"]  # shape: (T, N_g, K)
        per_t_counts += Y.sum(axis=1)  # sum over nodes, keep time × type

    print("\n[PER-TIME] Events per month (date, t, sell, repair):")
    for t in range(T):
        d = dates[t].strftime("%Y-%m-%d")
        counts = per_t_counts[t].tolist()
        mark = ""
        if t < T_train:
            mark = " (train)"
        else:
            mark = " (val)"
        print(f"{d}  t={t:02d}{mark}  {counts}")

    # 12) Meta dict
    meta = {
        "T": T,
        "T_train": T_train,
        "T_val": T_val,
        "K": K,
        "train_ratio": args.train_ratio,
        "ian_date": args.ian_date,
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
        "numeric_features": numeric_features,
        "cat_mappings": cat_mappings,
    }

    # 13) Save NPZ (main + parameter-labeled copy)
    communities_arr = np.array(communities, dtype=object)
    node_ids = parcel_df["node_id"].astype(int).values

    # --------------------------------------------------
    # Sanity check: per-type event counts in train vs val
    # --------------------------------------------------

    K = 2  # (sell, repair)
    train_counts = np.zeros(K, dtype=np.int64)
    val_counts   = np.zeros(K, dtype=np.int64)

    for comm in communities:
        Y = comm["Y"]  # shape: (T, N_g, K)
        # exact events (no windowing)
        train_counts += Y[:T_train].sum(axis=(0, 1))
        val_counts   += Y[T_train:].sum(axis=(0, 1))

    print("[SANITY] Exact events per type (sell, repair):")
    print("    Train:", train_counts.tolist())
    print("    Val  :", val_counts.tolist())
    print("    Total:", (train_counts + val_counts).tolist())

    # (Optional) also check windowed labels for a given horizon H
    H = 6  # or whatever horizon_months you use in training
    train_window_counts = np.zeros(K, dtype=np.int64)
    val_window_counts   = np.zeros(K, dtype=np.int64)

    for comm in communities:
        Y = comm["Y"]  # (T, N_g, K)
        T, N_g, K = Y.shape
        Yw = np.zeros_like(Y, dtype=np.int8)

        # Build window labels: if an event at time t, mark [t, t+H) as positive
        for t in range(T):
            if Y[t].any():
                t_end = min(T, t + H)
                # broadcast OR over time window
                Yw[t:t_end] |= Y[t]

        train_window_counts += Yw[:T_train].sum(axis=(0, 1))
        val_window_counts   += Yw[T_train:].sum(axis=(0, 1))

    print(f"[SANITY] Window-{H}-month events per type (sell, repair):")
    print("    Train window:", train_window_counts.tolist())
    print("    Val window  :", val_window_counts.tolist())
    print("    Total window:", (train_window_counts + val_window_counts).tolist())

    base, ext = os.path.splitext(args.out_npz)
    param_fname = f"{base}_dr{args.damage_radius_deg}_tr{args.train_ratio}.npz"
    np.savez_compressed(
        param_fname,
        communities=communities_arr,
        meta=meta,
        time_index=time_index.values,
        X_all=X_all,
        Y=Y,
        node_ids=node_ids,
    )
    print(f"Saved parameter-labeled copy to: {param_fname}")


if __name__ == "__main__":
    main()
