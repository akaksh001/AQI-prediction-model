"""
Phase 3b — Forecast Features (AQI + weather, future targets)
DSN2098 · Group 65 · Air Quality Analytics

Reads:
    data/processed/clean_aqi.csv       (Phase 1 output)
    data/raw/weather_delhi.csv         (from scripts/get_weather.py)

Saves (does NOT touch the nowcasting files):
    data/processed/forecast/train.csv, val.csv, test.csv
    data/processed/forecast/features.json   (feature groups used by phase 4b / 5b)

Task: at hour t, predict AQI at t+1, t+6 and t+24 hours.
Every feature is known at time t (current or past), so there is no leakage.
Because the target is in the future, current AQI itself is a legitimate feature.

Usage (run from the project folder):
    python scripts/phase3b_forecast_features.py
"""
import os
import sys
import json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

CLEAN_PATH   = "data/processed/clean_aqi.csv"
WEATHER_PATH = "data/raw/weather_delhi.csv"
OUT_DIR      = "data/processed/forecast"

HORIZONS   = [1, 6, 24]
GAP        = max(HORIZONS)      # rows purged at split edges (targets look GAP hours ahead)
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15

WEATHER_RAW = ["temperature_2m", "relative_humidity_2m", "dewpoint_2m",
               "surface_pressure", "precipitation", "wind_speed_10m",
               "wind_direction_10m"]
POLLUTANTS  = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2",
               "O3", "Benzene", "Toluene", "Xylene"]


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def main():
    for p in (CLEAN_PATH, WEATHER_PATH):
        if not os.path.exists(p):
            sys.exit(f"ERROR: {p} not found. Run the earlier phases / get_weather.py first.")
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load and join ─────────────────────────
    section("Load AQI + weather and join on datetime")
    aqi = pd.read_csv(CLEAN_PATH, parse_dates=["datetime"], index_col="datetime").sort_index()
    w   = pd.read_csv(WEATHER_PATH, parse_dates=["datetime"], index_col="datetime").sort_index()
    df  = aqi.join(w[WEATHER_RAW], how="left")

    n_missing = int(df[WEATHER_RAW].isna().any(axis=1).sum())
    print(f"  AQI rows: {len(aqi):,} | rows with missing weather after join: {n_missing}")
    df[WEATHER_RAW] = df[WEATHER_RAW].interpolate(limit=12, limit_direction="both")
    still = int(df[WEATHER_RAW].isna().any(axis=1).sum())
    if still:
        print(f"  WARNING: {still} rows still missing weather -> dropped")
        df = df.dropna(subset=WEATHER_RAW)

    # ── AQI history features ──────────────────
    section("Build features (all known at time t)")
    df["AQI_now"] = df["AQI"]
    aqi_feats = ["AQI_now"]
    for lag in [1, 3, 6, 24]:
        df[f"AQI_lag{lag}"] = df["AQI"].shift(lag)
        aqi_feats.append(f"AQI_lag{lag}")
    for win in [6, 24, 72]:
        df[f"AQI_roll{win}h_mean"] = df["AQI"].rolling(win, min_periods=1).mean()
        df[f"AQI_roll{win}h_std"]  = df["AQI"].rolling(win, min_periods=1).std().fillna(0)
        aqi_feats += [f"AQI_roll{win}h_mean", f"AQI_roll{win}h_std"]

    # ── Pollutant features ────────────────────
    poll_feats = [c for c in POLLUTANTS if c in df.columns]
    for lag in [1, 6, 24]:
        df[f"PM2.5_lag{lag}"] = df["PM2.5"].shift(lag)
        poll_feats.append(f"PM2.5_lag{lag}")
    for col in ["PM2.5", "PM10", "NO2"]:
        for win in [6, 24, 72]:
            df[f"{col}_roll{win}h_mean"] = df[col].rolling(win, min_periods=1).mean()
            df[f"{col}_roll{win}h_std"]  = df[col].rolling(win, min_periods=1).std().fillna(0)
            poll_feats += [f"{col}_roll{win}h_mean", f"{col}_roll{win}h_std"]

    # ── Weather features ──────────────────────
    wx = {"temperature_2m": "temp", "relative_humidity_2m": "rh", "dewpoint_2m": "dewpoint",
          "surface_pressure": "pressure", "precipitation": "precip", "wind_speed_10m": "wind_speed"}
    weather_feats = []
    for raw, short in wx.items():
        df[short] = df[raw]
        weather_feats.append(short)
    rad = np.deg2rad(df["wind_direction_10m"])
    df["wind_dir_sin"], df["wind_dir_cos"] = np.sin(rad), np.cos(rad)
    df["temp_dew_spread"]   = df["temp"] - df["dewpoint"]          # dryness / fog cue
    df["pressure_change_24h"] = df["pressure"] - df["pressure"].shift(24)
    df["precip_sum_24h"]    = df["precip"].rolling(24, min_periods=1).sum()
    weather_feats += ["wind_dir_sin", "wind_dir_cos", "temp_dew_spread",
                      "pressure_change_24h", "precip_sum_24h"]
    for short in ["temp", "rh", "wind_speed"]:
        for lag in [6, 24]:
            df[f"{short}_lag{lag}"] = df[short].shift(lag)
            weather_feats.append(f"{short}_lag{lag}")

    # ── Time features ─────────────────────────
    time_feats = []
    for name, vals, period in [("hour", df.index.hour, 24), ("month", df.index.month, 12),
                               ("day_of_week", df.index.dayofweek, 7),
                               ("day_of_year", df.index.dayofyear, 365)]:
        df[f"{name}_sin"] = np.sin(2 * np.pi * vals / period)
        df[f"{name}_cos"] = np.cos(2 * np.pi * vals / period)
        time_feats += [f"{name}_sin", f"{name}_cos"]
    df["is_winter"]  = df.index.month.isin([11, 12, 1, 2]).astype(int)
    df["is_weekend"] = df.index.dayofweek.isin([5, 6]).astype(int)
    df["is_night"]   = (df.index.hour.isin(range(22, 24)) | df.index.hour.isin(range(0, 6))).astype(int)
    time_feats += ["is_winter", "is_weekend", "is_night"]

    groups = {"AQI history": aqi_feats, "Pollutants": poll_feats,
              "Weather": weather_feats, "Time": time_feats}
    for g, cols in groups.items():
        print(f"  {g:<12}: {len(cols)} features")
    all_feats = [c for cols in groups.values() for c in cols]
    print(f"  TOTAL (with weather)   : {len(all_feats)}")
    print(f"  TOTAL (without weather): {len(all_feats) - len(weather_feats)}")

    # ── Future targets ────────────────────────
    section("Future targets")
    targets = []
    for h in HORIZONS:
        df[f"AQI_t{h}"]    = df["AQI"].shift(-h)
        df[f"hazard_t{h}"] = (df["AQI_Bucket"].shift(-h) == "Severe").astype(int)
        targets += [f"AQI_t{h}", f"hazard_t{h}"]
        print(f"  Added: AQI_t{h}, hazard_t{h}  (AQI {h} hour(s) ahead)")

    # Same rows for every horizon so the comparison is fair
    before = len(df)
    df = df.dropna(subset=all_feats + [f"AQI_t{GAP}"])
    print(f"\n  Rows before: {before:,} | after dropping NaN edges: {len(df):,}")

    # ── Time-ordered split with purge gap ─────
    section("Time-ordered split (70/15/15) with purge gap")
    n, n_tr, n_val = len(df), int(len(df) * TRAIN_FRAC), int(len(df) * VAL_FRAC)
    train = df.iloc[: n_tr - GAP]                       # purge: last targets would peek into val
    val   = df.iloc[n_tr : n_tr + n_val - GAP]          # purge: last targets would peek into test
    test  = df.iloc[n_tr + n_val :]
    for name, part in [("Train", train), ("Val", val), ("Test", test)]:
        print(f"  {name:<6}: {len(part):,}  ({part.index.min().date()} -> {part.index.max().date()})"
              f" | Severe rate (24h target): {part['hazard_t24'].mean()*100:.1f}%")

    keep = all_feats + targets + ["AQI", "AQI_Bucket"]
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part[keep].to_csv(os.path.join(OUT_DIR, f"{name}.csv"))
    with open(os.path.join(OUT_DIR, "features.json"), "w") as f:
        json.dump({"groups": groups, "horizons": HORIZONS, "gap": GAP}, f, indent=2)
    print(f"\n  Saved train/val/test.csv and features.json -> {OUT_DIR}")


if __name__ == "__main__":
    main()