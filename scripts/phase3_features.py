"""
Phase 3 — Feature Engineering
DSN2098 · Group 65 · Air Quality Analytics
Owner: Akaksh Samdani

Reads:  data/processed/clean_aqi.csv          (Phase 1 output)
Saves:
    data/processed/features.csv               <- full feature matrix
    data/processed/train.csv
    data/processed/val.csv
    data/processed/test.csv
    outputs/eda/08_stl_decomposition.png
    outputs/eda/09_lag_correlation.png
    outputs/eda/10_feature_summary.png

What this script builds:
    A) STL decomposition  -> trend, seasonal, residual per pollutant
    B) Lag features       -> AQI_lag1, AQI_lag3, AQI_lag6, AQI_lag24
    C) Rolling stats      -> 6h / 24h / 72h rolling mean & std for AQI + top pollutants
    D) Cyclical encoding  -> hour, month, day_of_week as (sin, cos) pairs
    E) Calendar flags     -> is_winter, is_weekend, is_night
    F) Ordinal target     -> AQI_ord (0=Good ... 5=Severe) for multi-class head
    G) Binary target      -> hazard (1 if Severe, else 0) for spike-detection head
    H) Time-ordered split -> 70% train / 15% val / 15% test (NO shuffle)

Usage:
    python scripts/phase3_features.py
    (run from D:/projectexhibit-aiml/)
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
CLEAN_PATH  = "data/processed/clean_aqi.csv"
FEAT_PATH   = "data/processed/features.csv"
TRAIN_PATH  = "data/processed/train.csv"
VAL_PATH    = "data/processed/val.csv"
TEST_PATH   = "data/processed/test.csv"
PLOT_DIR    = "outputs/eda"

TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15
# TEST_FRAC  = 0.15  (remainder)

# Pollutants we run STL on (top 4 by AQI correlation from Phase 2)
STL_COLS    = ["PM2.5", "PM10", "NO2", "Benzene"]

# Lag windows (hours)
LAG_WINDOWS = [1, 3, 6, 24]

# Rolling windows (hours)
ROLL_WINDOWS = [6, 24, 72]

# AQI category -> ordinal
BUCKET_ORD = {
    "Good": 0, "Satisfactory": 1, "Moderate": 2,
    "Poor": 3, "Very Poor": 4, "Severe": 5
}

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def check_file(path):
    if not os.path.exists(path):
        print(f"\nERROR: {path} not found. Run previous phases first.\n")
        sys.exit(1)

def save_plot(fig, filename):
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # ── Load ──────────────────────────────────
    section("Load clean_aqi.csv")
    check_file(CLEAN_PATH)
    df = pd.read_csv(CLEAN_PATH, parse_dates=["datetime"], index_col="datetime")
    df.sort_index(inplace=True)
    print(f"  Shape  : {df.shape}")
    print(f"  Range  : {df.index.min()} -> {df.index.max()}")

    # ── A. STL Decomposition ──────────────────
    section("A — STL Decomposition (trend + seasonal + residual)")
    stl_fig, axes = plt.subplots(len(STL_COLS), 3,
                                  figsize=(15, len(STL_COLS) * 2.5))
    stl_fig.suptitle("STL Decomposition — Trend / Seasonal / Residual",
                      fontsize=12, fontweight="bold")

    for i, col in enumerate(STL_COLS):
        series = df[col].copy()
        # Fill any remaining gaps with forward-fill for STL (needs complete series)
        series = series.ffill().bfill()

        stl = STL(series, period=24, robust=True).fit()   # 24h = 1 day period
        df[f"{col}_trend"]    = stl.trend
        df[f"{col}_seasonal"] = stl.seasonal
        df[f"{col}_resid"]    = stl.resid

        # Plot
        axes[i, 0].plot(stl.trend,    color="#4472c4", linewidth=0.4)
        axes[i, 0].set_title(f"{col} — Trend",    fontsize=8)
        axes[i, 1].plot(stl.seasonal, color="#ed7d31", linewidth=0.4)
        axes[i, 1].set_title(f"{col} — Seasonal", fontsize=8)
        axes[i, 2].plot(stl.resid,    color="#a9d18e", linewidth=0.4)
        axes[i, 2].set_title(f"{col} — Residual", fontsize=8)
        for ax in axes[i]:
            ax.set_xticklabels([])
            ax.tick_params(axis="y", labelsize=6)

        print(f"  {col:<12} -> {col}_trend, {col}_seasonal, {col}_resid  added")

    plt.tight_layout()
    save_plot(stl_fig, "08_stl_decomposition.png")

    # Also run STL on AQI itself
    aqi_series = df["AQI"].ffill().bfill()
    stl_aqi    = STL(aqi_series, period=24, robust=True).fit()
    df["AQI_trend"]    = stl_aqi.trend
    df["AQI_seasonal"] = stl_aqi.seasonal
    df["AQI_resid"]    = stl_aqi.resid
    print(f"  {'AQI':<12} -> AQI_trend, AQI_seasonal, AQI_resid  added")

    stl_features = (
        [f"{c}_{t}" for c in STL_COLS for t in ["trend","seasonal","resid"]] +
        ["AQI_trend", "AQI_seasonal", "AQI_resid"]
    )
    print(f"\n  STL features added: {len(stl_features)}")

    # ── B. Lag Features ───────────────────────
    section("B — Lag Features (AQI lags)")
    lag_features = []
    for lag in LAG_WINDOWS:
        col_name = f"AQI_lag{lag}"
        df[col_name] = df["AQI"].shift(lag)
        lag_features.append(col_name)
        print(f"  Added: {col_name}  (shift={lag}h)")

    # Also lag PM2.5 (strongest predictor)
    for lag in [1, 6, 24]:
        col_name = f"PM2.5_lag{lag}"
        df[col_name] = df["PM2.5"].shift(lag)
        lag_features.append(col_name)
        print(f"  Added: {col_name}")

    # ── Lag Correlation Plot ──────────────────
    aqi_lag_cols = [f"AQI_lag{l}" for l in LAG_WINDOWS]
    lag_corr = df[["AQI"] + aqi_lag_cols].corr()["AQI"].drop("AQI")
    fig, ax = plt.subplots(figsize=(7, 3))
    colors = ["#4472c4" if v > 0 else "#ed7d31" for v in lag_corr.values]
    bars = ax.bar(lag_corr.index, lag_corr.values, color=colors, edgecolor="white")
    for bar, val in zip(bars, lag_corr.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + 0.005, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("Lag Feature Correlation with Current AQI", fontsize=11, fontweight="bold")
    ax.set_xlabel("Lag Feature"); ax.set_ylabel("Pearson r")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.8, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label="r=0.8 threshold")
    ax.legend(fontsize=8)
    save_plot(fig, "09_lag_correlation.png")
    print(f"\n  Lag correlation with AQI:")
    for k, v in lag_corr.items():
        bar = "#" * int(v * 40)
        print(f"    {k:<14} r = {v:.3f}  {bar}")

    # ── C. Rolling Statistics ─────────────────
    section("C — Rolling Statistics")
    roll_cols   = ["AQI", "PM2.5", "PM10", "NO2"]
    roll_features = []
    for col in roll_cols:
        for w in ROLL_WINDOWS:
            mean_col = f"{col}_roll{w}h_mean"
            std_col  = f"{col}_roll{w}h_std"
            df[mean_col] = df[col].rolling(window=w, min_periods=1).mean()
            df[std_col]  = df[col].rolling(window=w, min_periods=1).std().fillna(0)
            roll_features += [mean_col, std_col]
            print(f"  Added: {mean_col}, {std_col}")

    print(f"\n  Rolling features added: {len(roll_features)}")

    # ── D. Cyclical Encoding ──────────────────
    section("D — Cyclical Encoding (sin/cos)")
    cycles = {
        "hour"        : (df.index.hour,        24),
        "month"       : (df.index.month,        12),
        "day_of_week" : (df.index.dayofweek,     7),
        "day_of_year" : (df.index.dayofyear,   365),
    }
    cyclic_features = []
    for name, (values, period) in cycles.items():
        df[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
        df[f"{name}_cos"] = np.cos(2 * np.pi * values / period)
        cyclic_features  += [f"{name}_sin", f"{name}_cos"]
        print(f"  Added: {name}_sin, {name}_cos  (period={period})")

    # ── E. Calendar Flags ─────────────────────
    section("E — Calendar Flags")
    df["is_winter"]  = df.index.month.isin([11, 12, 1, 2]).astype(int)
    df["is_weekend"] = df.index.dayofweek.isin([5, 6]).astype(int)
    df["is_night"]   = df.index.hour.isin(range(22, 24)).astype(int) | \
                       df.index.hour.isin(range(0, 6)).astype(int)
    calendar_features = ["is_winter", "is_weekend", "is_night"]
    for f in calendar_features:
        pct = df[f].mean() * 100
        print(f"  Added: {f:<15} ({pct:.1f}% of rows = 1)")

    # ── F+G. Target Variables ─────────────────
    section("F+G — Target Variables")
    df["AQI_ord"] = df["AQI_Bucket"].map(BUCKET_ORD)
    df["hazard"]  = (df["AQI_Bucket"] == "Severe").astype(int)
    print(f"  AQI_ord   : 0 (Good) -> 5 (Severe)")
    print(f"  hazard    : {df['hazard'].sum():,} Severe rows = {df['hazard'].mean()*100:.1f}% of data")
    print(f"\n  AQI_ord distribution:")
    for bucket, ord_val in BUCKET_ORD.items():
        count = (df["AQI_ord"] == ord_val).sum()
        print(f"    {ord_val} - {bucket:<14} : {count:>6,} rows")

    # ── Drop rows with NaN (from lags) ────────
    section("Drop NaN Rows from Lags")
    rows_before = len(df)
    df.dropna(subset=lag_features, inplace=True)
    rows_after  = len(df)
    print(f"  Rows before: {rows_before:,}")
    print(f"  Rows dropped (lag NaN): {rows_before - rows_after:,}  (first 24 hours)")
    print(f"  Rows after : {rows_after:,}")

    # ── Build Feature List ────────────────────
    section("Feature Matrix Summary")

    # Raw pollutant columns
    raw_poll = [c for c in ["PM2.5","PM10","NO","NO2","NOx","NH3",
                             "CO","SO2","O3","Benzene","Toluene","Xylene"]
                if c in df.columns]

    all_features = (
        raw_poll +
        stl_features +
        lag_features +
        roll_features +
        cyclic_features +
        calendar_features
    )
    # Keep only features that actually exist in df
    all_features = [f for f in all_features if f in df.columns]
    targets      = ["AQI", "AQI_ord", "hazard", "AQI_Bucket"]

    print(f"\n  Feature groups:")
    print(f"    Raw pollutants   : {len(raw_poll)}")
    print(f"    STL components   : {len(stl_features)}")
    print(f"    Lag features     : {len(lag_features)}")
    print(f"    Rolling stats    : {len(roll_features)}")
    print(f"    Cyclical         : {len(cyclic_features)}")
    print(f"    Calendar flags   : {len(calendar_features)}")
    print(f"    {'─'*30}")
    print(f"    TOTAL FEATURES   : {len(all_features)}")
    print(f"    TARGETS          : {targets}")

    # ── Feature Importance Preview Plot ───────
    # Show variance of each feature group as a bar
    group_labels = ["Raw\nPollutants", "STL\nComponents",
                    "Lag\nFeatures", "Rolling\nStats",
                    "Cyclical", "Calendar\nFlags"]
    group_counts = [len(raw_poll), len(stl_features), len(lag_features),
                    len(roll_features), len(cyclic_features), len(calendar_features)]
    group_colors = ["#4472c4","#ed7d31","#a9d18e","#ffc000","#9b59b6","#e74c3c"]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(group_labels, group_counts, color=group_colors,
                  edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, group_counts):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3, str(val),
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title(f"Feature Engineering Summary — {len(all_features)} Total Features",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Features")
    ax.set_ylim(0, max(group_counts) * 1.2)
    save_plot(fig, "10_feature_summary.png")

    # ── H. Time-Ordered Train/Val/Test Split ──
    section("H — Time-Ordered Train / Val / Test Split")
    n     = len(df)
    n_tr  = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    df_train = df.iloc[:n_tr]
    df_val   = df.iloc[n_tr : n_tr + n_val]
    df_test  = df.iloc[n_tr + n_val :]

    print(f"\n  Total rows : {n:,}")
    print(f"  Train      : {len(df_train):,}  "
          f"({df_train.index.min().date()} -> {df_train.index.max().date()})")
    print(f"  Val        : {len(df_val):,}  "
          f"({df_val.index.min().date()} -> {df_val.index.max().date()})")
    print(f"  Test       : {len(df_test):,}  "
          f"({df_test.index.min().date()} -> {df_test.index.max().date()})")

    print(f"\n  Hazard rate per split:")
    for name, split in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        rate = split["hazard"].mean() * 100
        print(f"    {name:<6} : {rate:.1f}% Severe")

    # ── Save ──────────────────────────────────
    section("Save Outputs")
    df[all_features + targets].to_csv(FEAT_PATH)
    print(f"  features.csv -> {FEAT_PATH}  ({os.path.getsize(FEAT_PATH)/1024/1024:.1f} MB)")

    df_train[all_features + targets].to_csv(TRAIN_PATH)
    df_val  [all_features + targets].to_csv(VAL_PATH)
    df_test [all_features + targets].to_csv(TEST_PATH)

    for path in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
        print(f"  {os.path.basename(path):<14} -> {path}  ({os.path.getsize(path)/1024:.0f} KB)")

    print(f"\n  Phase 3 complete.")
    print(f"  Next: Phase 4 (Anshu) reads train.csv / val.csv / test.csv")
    print(f"  Feature columns to use: all_features (listed below)")
    print(f"\n  all_features = {all_features}\n")


if __name__ == "__main__":
    main()
