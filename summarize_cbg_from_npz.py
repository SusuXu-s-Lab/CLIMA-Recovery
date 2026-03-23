#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from datetime import datetime


def _safe_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else np.nan


def _safe_mean_positive(x):
    """Mean over strictly positive values (avoids fillna(0) bias when missing -> 0)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(x.mean()) if x.size else np.nan


def _idx(name_to_j, name):
    return name_to_j.get(name, None)


def _invert_mapping(cat_to_code: dict):
    # cat_to_code: {category: code}
    # return {code: category}
    inv = {}
    for k, v in cat_to_code.items():
        try:
            inv[int(v)] = str(k)
        except Exception:
            continue
    return inv


def _cat_proportions(codes_1d, inv_code_to_cat):
    """Return dict: category -> proportion."""
    codes = np.asarray(codes_1d)
    codes = codes[np.isfinite(codes)].astype(int)
    if codes.size == 0:
        return {}

    uniq, cnt = np.unique(codes, return_counts=True)
    total = cnt.sum()
    out = {}
    for u, c in zip(uniq, cnt):
        cat = inv_code_to_cat.get(int(u), f"CODE_{int(u)}")
        out[cat] = float(c / total)
    return out


def _damage_expected_severity(dmg_props: dict):
    """
    Expected ordinal severity if the category names look like FEMA-ish damage levels.
    Otherwise returns NaN.
    """
    if not dmg_props:
        return np.nan

    # low -> high mapping by substring
    mapping = [
        ("no", 0.0),
        ("none", 0.0),
        ("affected", 1.0),
        ("minor", 2.0),
        ("major", 3.0),
        ("destroy", 4.0),
    ]

    exp = 0.0
    used = 0.0
    for cat, p in dmg_props.items():
        cl = str(cat).lower()
        score = None
        for key, s in mapping:
            if key in cl:
                score = s
                break
        if score is None:
            continue
        exp += score * p
        used += p

    return float(exp) if used >= 0.5 else np.nan


def summarize(npz_path: str, out_csv: str, ref_year: int = None):
    data = np.load(npz_path, allow_pickle=True)
    print("[INFO] NPZ keys:", data.files)

    communities = list(data["communities"])
    meta = data["meta"].item()

    numeric_features = list(meta.get("numeric_features", []))
    cat_mappings = meta.get("cat_mappings", {})  # {col: {category: code}}

    # reconstruct feature ordering exactly as your prep uses:
    # X_all = [X_num (numeric_features), X_cat (one code per cat feature in cat_features order)]
    # cat_mappings was built by iterating cat_features in order, and dict preserves insertion order in py3.7+.
    cat_features_in_order = list(cat_mappings.keys())

    # determine reference year for house age
    if ref_year is None:
        # script stores meta["ian_date"] like "2022-09-28"
        ian_date = meta.get("ian_date", None)
        if ian_date is not None:
            try:
                ref_year = datetime.strptime(ian_date, "%Y-%m-%d").year
            except Exception:
                ref_year = 2022
        else:
            ref_year = 2022

    name_to_j = {name: j for j, name in enumerate(numeric_features)}

    # numeric columns you asked for (present in your script)
    j_yearbuilt = _idx(name_to_j, "yearbuilt")
    j_income = _idx(name_to_j, "median_household_income")

    # “market/assessed value” components (present)
    j_parval = _idx(name_to_j, "parval")
    j_landval = _idx(name_to_j, "landval")
    j_improvval = _idx(name_to_j, "improvval")

    # other strong characterizers already in your numeric list
    j_popdens = _idx(name_to_j, "population_density")
    j_risk = _idx(name_to_j, "fema_nri_risk_rating")
    j_afford = _idx(name_to_j, "housing_affordability_index")
    j_flood = _idx(name_to_j, "ian_FloodDepth")
    j_bldgval = _idx(name_to_j, "ian_BldgValue")
    j_estloss = _idx(name_to_j, "ian_EstLoss")

    # categorical columns you asked about / useful
    dmg_col = "ian_DamageLevel" if "ian_DamageLevel" in cat_mappings else None
    occ_col = "ian_Occupancy" if "ian_Occupancy" in cat_mappings else None

    # NOTE: Your prep script does NOT include resident age or education in numeric_features.
    # We’ll output NaN for those unless you add columns upstream.
    rows = []

    for comm in communities:
        cbg = comm.get("cbg", comm.get("cbg_id", None))
        X = comm["X"]        # [N, d] where d = len(numeric_features)+len(cat_features_in_order)
        coords = comm["coords"]  # [N,2] stored as (lon, lat)
        Y = comm["Y"]        # [T,N,K] where K=3 in this script: (sell, repair, vacate)

        N = X.shape[0]
        T = Y.shape[0]
        K = Y.shape[2]

        X_num = X[:, :len(numeric_features)]
        X_cat = X[:, len(numeric_features):]  # one column per categorical feature (integer codes)

        # numeric means
        parval = X_num[:, j_parval] if j_parval is not None else np.full(N, np.nan)
        landval = X_num[:, j_landval] if j_landval is not None else np.full(N, np.nan)
        improvval = X_num[:, j_improvval] if j_improvval is not None else np.full(N, np.nan)

        assessed_total = parval + landval + improvval  # will be 0 if missing filled as 0 in prep

        mean_assessed = _safe_mean_positive(assessed_total)  # avoid fillna(0) downward bias
        mean_parval = _safe_mean_positive(parval)
        mean_landval = _safe_mean_positive(landval)
        mean_improvval = _safe_mean_positive(improvval)

        income = X_num[:, j_income] if j_income is not None else np.full(N, np.nan)
        mean_income = _safe_mean_positive(income)

        yearbuilt = X_num[:, j_yearbuilt] if j_yearbuilt is not None else np.full(N, np.nan)
        # treat <= 0 as missing (because your prep fills NaN->0)
        valid_yb = yearbuilt[(np.isfinite(yearbuilt)) & (yearbuilt > 0)]
        mean_yearbuilt = float(valid_yb.mean()) if valid_yb.size else np.nan
        mean_house_age = float(ref_year - valid_yb.mean()) if valid_yb.size else np.nan

        # damage severity: categorical code -> proportions + expected severity
        dmg_expected = np.nan
        dmg_mode = None
        dmg_props = {}
        if dmg_col is not None:
            dmg_j = cat_features_in_order.index(dmg_col)
            dmg_codes = X_cat[:, dmg_j]
            inv = _invert_mapping(cat_mappings[dmg_col])
            dmg_props = _cat_proportions(dmg_codes, inv)
            dmg_expected = _damage_expected_severity(dmg_props)
            if dmg_props:
                dmg_mode = max(dmg_props, key=dmg_props.get)

        # other useful numeric characterizers
        popdens = X_num[:, j_popdens] if j_popdens is not None else np.full(N, np.nan)
        risk = X_num[:, j_risk] if j_risk is not None else np.full(N, np.nan)
        afford = X_num[:, j_afford] if j_afford is not None else np.full(N, np.nan)
        flood = X_num[:, j_flood] if j_flood is not None else np.full(N, np.nan)
        bldgval = X_num[:, j_bldgval] if j_bldgval is not None else np.full(N, np.nan)
        estloss = X_num[:, j_estloss] if j_estloss is not None else np.full(N, np.nan)

        # event rates per household
        sell_rate = float(Y[:, :, 0].sum() / max(N, 1)) if K >= 1 else np.nan
        repair_rate = float(Y[:, :, 1].sum() / max(N, 1)) if K >= 2 else np.nan
        vacate_rate = float(Y[:, :, 2].sum() / max(N, 1)) if K >= 3 else np.nan

        # spatial descriptors
        lon = coords[:, 0].astype(float)
        lat = coords[:, 1].astype(float)

        row = {
            "cbg": cbg,
            "community_size_households": int(N),
            "T_months": int(T),

            # requested
            "mean_house_market_or_assessed_value": mean_assessed,
            "mean_parval": mean_parval,
            "mean_landval": mean_landval,
            "mean_improvval": mean_improvval,

            "mean_income": mean_income,

            # not in your current prep -> will be NaN unless you add features
            "mean_age": np.nan,
            "mean_educational_level": np.nan,

            # available (your numeric has yearbuilt)
            "mean_yearbuilt": mean_yearbuilt,
            "mean_house_age": mean_house_age,

            # damage
            "damage_expected_severity": dmg_expected,
            "damage_mode_category": dmg_mode,

            # other strong cbg characterizers already in your features
            "mean_population_density": _safe_mean(popdens),
            "mean_fema_nri_risk_rating": _safe_mean(risk),
            "mean_housing_affordability_index": _safe_mean(afford),
            "mean_ian_flood_depth": _safe_mean(flood),
            "mean_ian_bldg_value": _safe_mean_positive(bldgval),
            "mean_ian_est_loss": _safe_mean_positive(estloss),

            # behavior rates
            "sell_events_per_household": sell_rate,
            "repair_events_per_household": repair_rate,
            "vacate_events_per_household": vacate_rate,

            # spatial spread
            "lon_mean": float(np.nanmean(lon)),
            "lat_mean": float(np.nanmean(lat)),
            "lon_std": float(np.nanstd(lon)),
            "lat_std": float(np.nanstd(lat)),
        }

        # add damage proportions as wide columns (helpful)
        for cat, p in dmg_props.items():
            row[f"damage_prop_{cat}"] = p

        # add occupancy proportions (also helpful) if present
        if occ_col is not None:
            occ_j = cat_features_in_order.index(occ_col)
            occ_codes = X_cat[:, occ_j]
            inv = _invert_mapping(cat_mappings[occ_col])
            occ_props = _cat_proportions(occ_codes, inv)
            for cat, p in occ_props.items():
                row[f"occupancy_prop_{cat}"] = p

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print(f"[DONE] wrote {out_csv} | rows={len(df)}")
    print("[INFO] NOTE: mean_age and mean_educational_level are NaN because prepare_real_ian_by_cbg.py does not include those features.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_path", required=True)
    ap.add_argument("--out_csv", default="cbg_summary.csv")
    ap.add_argument("--ref_year", type=int, default=None, help="Reference year to compute house_age = ref_year - yearbuilt. Default uses Ian year.")
    args = ap.parse_args()

    summarize(args.npz_path, args.out_csv, args.ref_year)


if __name__ == "__main__":
    main()
