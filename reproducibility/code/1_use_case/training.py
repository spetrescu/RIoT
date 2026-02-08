import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import json
import argparse
from pathlib import Path
import random

DATA_DIR = "../../data/figure3"
RESAMPLE_RULE = "15min"
EUROPE_BUCHAREST = "Europe/Bucharest"
RANDOM_STATE = 7
TEST_DAYS = 10
SCENARIO_A_ROOM = "room_1"
SCENARIO_B_ROOM = "room_2"
SCENARIO_B_MODALITY = "temperature"

TS_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}T")
VALUE_REGEX = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def modality_from_filename(path: str) -> str:
    base = os.path.basename(path).lower()
    m = re.search(r"_sensor_([a-z0-9]+)\.txt$", base)
    if not m:
        raise ValueError(f"Cannot infer modality from filename: {base}")
    mod = m.group(1)
    return {
        "hum": "humidity",
        "temp": "temperature",
        "co2": "co2",
        "tvoc": "tvoc",
        "pir": "pir",
    }.get(mod, mod)

def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    np.random.seed(seed)

def room_from_filename(path: str) -> str:
    base = os.path.basename(path)
    m = re.match(r"(\d+)_sensor_", base)
    if not m:
        raise ValueError(f"Cannot infer room id from filename: {base}")
    return f"room_{int(m.group(1))}"


def parse_timestamp(ts_str: str) -> pd.Timestamp:
    s = ts_str.strip()
    s = s.replace(" Bucharest", "")
    ts_utc = pd.to_datetime(s, utc=True)
    ts_local = ts_utc.tz_convert(EUROPE_BUCHAREST).tz_localize(None)
    return ts_local


def parse_value(val_str: str) -> float:
    s = val_str.strip()
    if s == "":
        return np.nan
    if s.upper() == "T":
        return 1.0
    if s.upper() == "F":
        return 0.0
    m = VALUE_REGEX.search(s)
    return float(m.group(0)) if m else np.nan


def load_sensor_file_long(path: str) -> pd.DataFrame:
    room = room_from_filename(path)
    modality = modality_from_filename(path)

    ts_list, val_list = [], []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not TS_REGEX.match(line):
                continue  #header

            if "," not in line:
                continue

            ts_part, val_part = line.split(",", 1)
            try:
                ts = parse_timestamp(ts_part)
            except Exception:
                continue

            val = parse_value(val_part)
            ts_list.append(ts)
            val_list.append(val)

    if not ts_list:
        raise ValueError(f"No data rows parsed from: {path}")

    return pd.DataFrame({
        "timestamp": pd.to_datetime(ts_list),
        "room": room,
        "modality": modality,
        "value": val_list
    })


def load_all_data_wide(data_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(data_dir, "*_sensor_*.txt")))
    if not paths:
        raise FileNotFoundError(f"No *_sensor_*.txt files found in {data_dir}")

    parts = []
    for p in paths:
        try:
            parts.append(load_sensor_file_long(p))
        except Exception as e:
            print(f"Skipping {p}: {e}")

    if not parts:
        raise RuntimeError("No files successfully parsed.")

    long_df = pd.concat(parts, ignore_index=True)

    wide = long_df.pivot_table(
        index=["timestamp", "room"],
        columns="modality",
        values="value",
        aggfunc="mean"
    ).reset_index()

    wide.columns.name = None
    wide = wide.sort_values(["room", "timestamp"]).reset_index(drop=True)
    return wide


def regularize_and_resample(df: pd.DataFrame, rule: str, modalities: list[str]) -> pd.DataFrame:
    out = []
    for room, g in df.groupby("room"):
        g = g.sort_values("timestamp").set_index("timestamp")

        g_mean = g[modalities].resample(rule).mean()

        if "pir" in modalities and "pir" in g.columns:
            g_mean["pir"] = g[["pir"]].resample(rule).max()["pir"]

        g_mean["room"] = room
        out.append(g_mean.reset_index())

    return pd.concat(out, ignore_index=True).sort_values(["room", "timestamp"]).reset_index(drop=True)


def coverage_table(df_test_true: pd.DataFrame, modalities: list[str]) -> pd.DataFrame:
    rows = []
    for room, g in df_test_true.groupby("room"):
        row = {"room": room, "rows": len(g)}
        for m in modalities:
            row[f"{m}_non_nan"] = int(g[m].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rows", ascending=False)


def pick_best_rooms(cov: pd.DataFrame, modalities: list[str], min_points=50):
    score = []
    for _, r in cov.iterrows():
        ok = 0
        for m in modalities:
            if r.get(f"{m}_non_nan", 0) >= min_points:
                ok += 1
        score.append(ok)
    cov = cov.copy()
    cov["modality_score"] = score
    cov = cov.sort_values(["modality_score", "rows"], ascending=False)

    good = cov[cov["modality_score"] > 0]["room"].tolist()
    if not good:
        cov["total_non_nan"] = cov[[f"{m}_non_nan" for m in modalities]].sum(axis=1)
        return cov.sort_values("total_non_nan", ascending=False)["room"].tolist()
    return good


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    hour = d["timestamp"].dt.hour.values
    dow = d["timestamp"].dt.dayofweek.values
    d["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    d["cos_hour"] = np.cos(2 * np.pi * hour / 24)
    d["sin_dow"] = np.sin(2 * np.pi * dow / 7)
    d["cos_dow"] = np.cos(2 * np.pi * dow / 7)
    return d


def add_building_aggregate_features(df: pd.DataFrame, modalities: list[str]) -> pd.DataFrame:
    d = df.copy()
    n_rooms = d["room"].nunique()

    mean_all = d.groupby("timestamp")[modalities].mean(numeric_only=True)
    mean_all = mean_all.rename(columns={m: f"mean_all_{m}" for m in modalities})
    d = d.merge(mean_all, on="timestamp", how="left")

    for m in modalities:
        sum_all = d[f"mean_all_{m}"] * n_rooms
        other = (sum_all - d[m]) / max(n_rooms - 1, 1)
        d[f"mean_other_{m}"] = other.fillna(d[f"mean_all_{m}"])

    return d


def add_room_lags(df: pd.DataFrame, modalities: list[str], lags=(1, 2, 3)) -> pd.DataFrame:
    d = df.sort_values(["room", "timestamp"]).copy()
    for m in modalities:
        for L in lags:
            d[f"{m}_lag{L}"] = d.groupby("room")[m].shift(L)
    return d


def train_per_modality_models(train_df: pd.DataFrame, target_modalities: list[str]) -> dict:
    models = {}
    for target in target_modalities:
        drop_cols = {"timestamp", target}
        feature_cols = [c for c in train_df.columns if c not in drop_cols]

        X = train_df[feature_cols]
        y = train_df[target]

        cat_cols = ["room"]
        num_cols = [c for c in feature_cols if c not in cat_cols]

        pre = ColumnTransformer(
            transformers=[
                ("room", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("num", "passthrough", num_cols),
            ],
            remainder="drop",
        )

        reg = HistGradientBoostingRegressor(
            random_state=RANDOM_STATE,
            max_depth=6,
            learning_rate=0.06,
            max_iter=500,
        )

        pipe = Pipeline([("pre", pre), ("reg", reg)])

        mask = ~y.isna()
        pipe.fit(X.loc[mask], y.loc[mask])

        models[target] = (pipe, feature_cols)
    return models


def predict_with_models(models: dict, df: pd.DataFrame, target_modalities: list[str]) -> pd.DataFrame:
    out = df.copy()
    for target in target_modalities:
        model, feature_cols = models[target]
        out[f"pred_{target}"] = model.predict(out[feature_cols])
    return out


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def plot_window(
    df_window: pd.DataFrame,
    room: str,
    modalities: list[str],
    title: str,
    out_dir: str | None = None,

    gt_color: str = "#31403f",
    pred_color: str = "#b05ccf",
    use_markers: bool = False,
    gt_marker: str = "*",
    pred_marker: str = ">",
    markevery: int = 6,
    linewidth: float = 1.6,
):
    height_per_row = 1.5 if len(modalities) > 1 else 6
    fig, axes = plt.subplots(len(modalities), 1, figsize=(6, height_per_row * len(modalities)), sharex=True)
    if len(modalities) == 1:
        axes = [axes]

    for ax, m in zip(axes, modalities):
        # Ground truth line
        ax.plot(
            df_window["timestamp"],
            df_window[m],
            label=f"true {m}",
            color=gt_color,
            linewidth=linewidth,
            marker=(gt_marker if use_markers else None),
            markevery=(markevery if use_markers else None),
            markersize=6,
        )

        # Prediction line
        ax.plot(
            df_window["timestamp"],
            df_window[f"pred_{m}"],
            label=f"pred {m}",
            color=pred_color,
            linewidth=linewidth,
            linestyle="--",
            marker=(pred_marker if use_markers else None),
            markevery=(markevery if use_markers else None),
            markersize=5,
        )

        ax.set_ylabel(m)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("timestamp")
    fig.suptitle(f"{room} — {title}", y=0.99)
    plt.tight_layout()

    if out_dir is not None:
        fig_dir = os.path.join(out_dir, "figures")
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        fname = f"{safe_slug(room)}_{safe_slug(title)}_{safe_slug('_'.join(modalities))}.pdf"
        out_path = os.path.join(fig_dir, fname)
        plt.savefig(out_path, dpi=250, bbox_inches="tight")
        print(f"Saved figure {out_path}")
    else:
        fname = f"{safe_slug(room)}_{safe_slug(title)}_{safe_slug('_'.join(modalities))}.pdf"
        plt.savefig(fname, dpi=250, bbox_inches="tight")
        print(f"Saved figure {fname}")

    plt.close(fig)

def safe_slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", s).strip("_")


def save_run_artifacts(
    out_dir: str,
    results: pd.DataFrame,
    run_config: dict,
    plotA: pd.DataFrame | None,
    plotB: pd.DataFrame | None,
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    results_path = os.path.join(out_dir, "results.csv")
    results.to_csv(results_path, index=False)

    config_path = os.path.join(out_dir, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, default=str)

    if plotA is not None and len(plotA) > 0:
        plotA.to_csv(os.path.join(out_dir, "plot_scenario_A.csv"), index=False)
    if plotB is not None and len(plotB) > 0:
        plotB.to_csv(os.path.join(out_dir, "plot_scenario_B.csv"), index=False)

    print(f"\nSaved run artifacts to {out_dir}")
    print(f" - {results_path}")
    print(f" - {config_path}")
    if plotA is not None and len(plotA) > 0:
        print(f" - {os.path.join(out_dir, 'plot_scenario_A.parquet')}")
    if plotB is not None and len(plotB) > 0:
        print(f" - {os.path.join(out_dir, 'plot_scenario_B.parquet')}")


def load_run_artifacts(out_dir: str):
    results = pd.read_csv(os.path.join(out_dir, "results.csv"))
    with open(os.path.join(out_dir, "run_config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)

    plotA_path = os.path.join(out_dir, "plot_scenario_A.csv")
    plotB_path = os.path.join(out_dir, "plot_scenario_B.csv")

    plotA = pd.read_csv(plotA_path) if os.path.exists(plotA_path) else None
    plotB = pd.read_csv(plotB_path) if os.path.exists(plotB_path) else None

    for df in [plotA, plotB]:
        if df is not None and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

    return results, cfg, plotA, plotB


def main(out_dir="runs/latest"):
    seed_everything(RANDOM_STATE)

    df_raw = load_all_data_wide(DATA_DIR)
    print(f"Loaded wide rows: {len(df_raw):,}")
    print("Columns:", df_raw.columns.tolist())

    base_cols = {"timestamp", "room"}
    modalities = [c for c in df_raw.columns if c not in base_cols]
    if not modalities:
        raise RuntimeError("No modalities detected after pivot.")
    print("Detected modalities:", modalities)

    df = regularize_and_resample(df_raw, RESAMPLE_RULE, modalities)

    df = add_time_features(df)
    df = add_building_aggregate_features(df, modalities)
    df = add_room_lags(df, modalities, lags=(1, 2, 3))
    
    min_rooms = 2
    min_modalities = 3

    row_mod_count = df[modalities].notna().sum(axis=1)
    df["_mod_count"] = row_mod_count

    good_ts = (
        df[df["_mod_count"] >= min_modalities]
        .groupby("timestamp")["room"].nunique()
    )

    good_ts = good_ts[good_ts >= min_rooms].index.values
    df = df.drop(columns=["_mod_count"])

    if len(good_ts) < 500:
        print("Not many timestamps with good multi-room coverage. Consider lowering thresholds.")
    else:
        df = df[df["timestamp"].isin(good_ts)].copy()
    
    print(df.groupby("room")[modalities].apply(lambda x: x.notna().sum()).sort_values(by=modalities, ascending=False))

    unique_ts = np.sort(df["timestamp"].unique())
    if len(unique_ts) < 300:
        raise RuntimeError("Not enough time points after resampling.")

    steps_per_day = int(pd.Timedelta("1D") / pd.Timedelta(RESAMPLE_RULE))
    test_steps = min(TEST_DAYS * steps_per_day, len(unique_ts) // 3)
    cutoff = unique_ts[-test_steps]

    train_df = df[df["timestamp"] < cutoff].copy()
    test_df_obs = df[df["timestamp"] >= cutoff].copy()
    test_df_true = test_df_obs[["timestamp", "room"] + modalities].copy()

    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df_obs):,} | cutoff: {cutoff}")

    cov = coverage_table(test_df_true, modalities)
    print("\n=== Test coverage per room (non-NaN counts) ===")
    print(cov.to_string(index=False))

    best_rooms = pick_best_rooms(cov, modalities, min_points=50)
    if not best_rooms:
        raise RuntimeError("No rooms have enough data in the test period to evaluate.")

    scenario_a_room = best_rooms[0]
    scenario_b_room = best_rooms[1] if len(best_rooms) > 1 else best_rooms[0]

    print(f"\nChosen Scenario A room: {scenario_a_room}")
    print(f"Chosen Scenario B room: {scenario_b_room}")

    test_ts = np.sort(test_df_obs["timestamp"].unique())
    idx0 = min(10, len(test_ts) // 5)

    win_len_A = min(4 * steps_per_day, max(5, len(test_ts) // 3))
    win_len_B = min(3 * steps_per_day, max(5, len(test_ts) // 3))

    idxA_start = min(idx0, max(0, len(test_ts) - win_len_A - 1))
    idxA_end = idxA_start + win_len_A
    startA, endA = test_ts[idxA_start], test_ts[idxA_end - 1]

    idxB_start = min(idxA_end + steps_per_day, max(0, len(test_ts) - win_len_B - 1))
    idxB_end = min(idxB_start + win_len_B, len(test_ts))
    startB, endB = test_ts[idxB_start], test_ts[idxB_end - 1]

    # Scenario A: room fully missing
    maskA = (test_df_obs["room"] == scenario_a_room) & (test_df_obs["timestamp"] >= startA) & (test_df_obs["timestamp"] <= endA)
    for m in modalities:
        test_df_obs.loc[maskA, m] = np.nan

    # Scenario B: one modality missing
    maskB = None
    if SCENARIO_B_MODALITY in modalities:
        maskB = (test_df_obs["room"] == scenario_b_room) & (test_df_obs["timestamp"] >= startB) & (test_df_obs["timestamp"] <= endB)
        test_df_obs.loc[maskB, SCENARIO_B_MODALITY] = np.nan
    else:
        print(f"Scenario B modality '{SCENARIO_B_MODALITY}' not present; skipping.")

    models = train_per_modality_models(train_df, target_modalities=modalities)
    pred_test = predict_with_models(models, test_df_obs, target_modalities=modalities)

    merged = pred_test.merge(test_df_true, on=["timestamp", "room"], suffixes=("_obs", "_true"), how="left")

    rows = []

    winA = (merged["room"] == scenario_a_room) & (merged["timestamp"] >= startA) & (merged["timestamp"] <= endA)
    for m in modalities:
        y_true = merged.loc[winA, f"{m}_true"]
        y_pred = merged.loc[winA, f"pred_{m}"]
        valid = ~y_true.isna()
        if valid.sum() == 0:
            continue
        rows.append({
            "scenario": "A_room_missing_all_modalities",
            "room": scenario_a_room,
            "modality": m,
            "rmse": rmse(y_true[valid], y_pred[valid]),
            "mae": mae(y_true[valid], y_pred[valid]),
            "n": int(valid.sum()),
        })

    if maskB is not None:
        winB = (merged["room"] == scenario_b_room) & (merged["timestamp"] >= startB) & (merged["timestamp"] <= endB)
        m = SCENARIO_B_MODALITY
        y_true = merged.loc[winB, f"{m}_true"]
        y_pred = merged.loc[winB, f"pred_{m}"]
        valid = ~y_true.isna()
        if valid.sum() > 0:
            rows.append({
                "scenario": "B_single_modality_missing",
                "room": scenario_b_room,
                "modality": m,
                "rmse": rmse(y_true[valid], y_pred[valid]),
                "mae": mae(y_true[valid], y_pred[valid]),
                "n": int(valid.sum()),
            })

    results = pd.DataFrame(rows).sort_values(["scenario", "room", "modality"])
    print("\n=== Results ===")
    print(results.to_string(index=False) if len(results) else "No results computed (no GT in windows).")

    plotA = None
    if winA.any():
        plotA_tmp = merged.loc[winA, ["timestamp", "room"] + [f"{m}_true" for m in modalities] + [f"pred_{m}" for m in modalities]].copy()
        keepA = []
        for m in modalities:
            if plotA_tmp[f"{m}_true"].notna().any():
                plotA_tmp[m] = plotA_tmp[f"{m}_true"]
                keepA.append(m)
        cols = ["timestamp", "room"] + keepA + [f"pred_{m}" for m in keepA]
        plotA = plotA_tmp[cols].copy()

    plotB = None
    if maskB is not None:
        winB = (merged["room"] == scenario_b_room) & (merged["timestamp"] >= startB) & (merged["timestamp"] <= endB)
        if winB.any() and f"{SCENARIO_B_MODALITY}_true" in merged.columns:
            plotB_tmp = merged.loc[winB, ["timestamp", "room", f"{SCENARIO_B_MODALITY}_true", f"pred_{SCENARIO_B_MODALITY}"]].copy()
            if plotB_tmp[f"{SCENARIO_B_MODALITY}_true"].notna().any():
                plotB_tmp[SCENARIO_B_MODALITY] = plotB_tmp[f"{SCENARIO_B_MODALITY}_true"]
                plotB = plotB_tmp[["timestamp", "room", SCENARIO_B_MODALITY, f"pred_{SCENARIO_B_MODALITY}"]].copy()

    run_config = {
        "data_dir": DATA_DIR,
        "resample_rule": RESAMPLE_RULE,
        "cutoff": str(cutoff),
        "scenario_a_room": scenario_a_room,
        "scenario_b_room": scenario_b_room,
        "scenario_b_modality": SCENARIO_B_MODALITY,
        "startA": str(startA),
        "endA": str(endA),
        "startB": str(startB),
        "endB": str(endB),
        "modalities": modalities,
        "steps_per_day": steps_per_day,
    }
    save_run_artifacts(out_dir, results, run_config, plotA, plotB)

    if plotA is not None and len(plotA) > 0:
        modsA = [c for c in plotA.columns if c.startswith("pred_")]
        modsA = [m.replace("pred_", "") for m in modsA if m.replace("pred_", "") in plotA.columns]
        plot_window(plotA, scenario_a_room, modsA, "Scenario A: room fully missing window (GT vs Pred)")

    if plotB is not None and len(plotB) > 0:
        modsB = [c for c in plotB.columns if c.startswith("pred_")]
        modsB = [m.replace("pred_", "") for m in modsB if m.replace("pred_", "") in plotB.columns]
        plot_window(plotB, scenario_b_room, modsB, "Scenario B: one modality missing (GT vs Pred)")

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="runs/latest", help="Where to save/load artifacts")
    parser.add_argument("--plot-only", action="store_true", help="Skip training; only plot from saved artifacts")
    args = parser.parse_args()

    if args.plot_only:
        results, cfg, plotA, plotB = load_run_artifacts(args.out_dir)
        print("\n=== Loaded Results ===")
        print(results.to_string(index=False))

        if plotA is not None and len(plotA) > 0:
            modsA = [c.replace("pred_", "") for c in plotA.columns if c.startswith("pred_")]
            modsA = [m for m in modsA if m in plotA.columns]
            plot_window(plotA, cfg["scenario_a_room"], modsA,
                        "Scenario A: room fully missing window (GT vs Pred)",
                        out_dir=args.out_dir)

        if plotB is not None and len(plotB) > 0:
            modsB = [c.replace("pred_", "") for c in plotB.columns if c.startswith("pred_")]
            modsB = [m for m in modsB if m in plotB.columns]
            plot_window(plotB, cfg["scenario_b_room"], modsB,
                        "Scenario B: one modality missing (GT vs Pred)",
                        out_dir=args.out_dir)
        return

    main(out_dir=args.out_dir)

if __name__ == "__main__":
    cli()
