"""
Phase 1 — Data Collection & Cleaning
DSN2098 · Group 65 · Air Quality Analytics
Owner: Aryavardhan

Steps:
  1. Load city_hour.csv from Kaggle dataset
  2. Filter to Delhi only
  3. Validate schema & parse datetime
  4. Report missingness
  5. MICE imputation (IterativeImputer)
  6. IQR outlier clipping
  7. Re-derive AQI_Bucket from cleaned AQI
  8. Validate output
  9. Save → data/processed/clean_aqi.csv

Usage:
  python scripts/phase1_clean.py
  (run from D:/projectexhibit-aiml/)
"""

import os
import sys
# Force UTF-8 output so emoji / arrows render on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.experimental import enable_iterative_imputer   # noqa
from sklearn.impute import IterativeImputer

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
RAW_PATH       = "data/raw/city_hour.csv"
OUT_PATH       = "data/processed/clean_aqi.csv"
PLOT_DIR       = "outputs/eda"
TARGET_CITY    = "Delhi"

# Pollutant + meteorological columns present in city_hour.csv
POLLUTANT_COLS = ["PM2.5", "PM10", "NO", "NO2", "NOx",
                  "NH3", "CO", "SO2", "O3", "Benzene",
                  "Toluene", "Xylene"]
# NOTE: city_hour.csv does NOT include met. variables (temp/humidity/etc.)
# Those will be added in Phase 3 (Feature Engineering) via an auxiliary dataset.
# For Phase 1 we clean what we have.

NUMERIC_COLS   = POLLUTANT_COLS + ["AQI"]

IQR_MULTIPLIER = 1.5   # standard Tukey fences

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check_file(path: str):
    if not os.path.exists(path):
        print(f"\n❌  File not found: {path}")
        print(f"    Please copy city_hour.csv into  data/raw/  and re-run.\n")
        sys.exit(1)


def missingness_report(df: pd.DataFrame, label: str):
    section(f"Missingness Report — {label}")
    miss = df[NUMERIC_COLS].isnull().sum()
    pct  = (miss / len(df) * 100).round(2)
    report = pd.DataFrame({"Missing": miss, "Pct (%)": pct})
    report = report[report["Missing"] > 0].sort_values("Pct (%)", ascending=False)
    if report.empty:
        print("  ✅  No missing values.")
    else:
        print(report.to_string())
    return report


def plot_missingness(df: pd.DataFrame, filename: str, title: str):
    """Heatmap of missing values across columns."""
    fig, ax = plt.subplots(figsize=(12, 4))
    mask = df[NUMERIC_COLS].isnull()
    sns.heatmap(mask.T, cbar=False, yticklabels=True,
                xticklabels=False, ax=ax,
                cmap=["#e8f5e9", "#e53935"])
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Column")
    ax.set_xlabel("Time →")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=120)
    plt.close()
    print(f"  📊  Saved: outputs/eda/{filename}")


def plot_boxplots(before: pd.DataFrame, after: pd.DataFrame, filename: str):
    """Side-by-side boxplots before and after IQR clipping."""
    cols = [c for c in NUMERIC_COLS if c in before.columns]
    n = len(cols)
    fig, axes = plt.subplots(2, n, figsize=(n * 2.5, 6), sharey=False)
    for i, col in enumerate(cols):
        axes[0, i].boxplot(before[col].dropna(), vert=True, patch_artist=True,
                           boxprops=dict(facecolor="#ef9a9a"))
        axes[0, i].set_title(col, fontsize=7)
        axes[0, i].tick_params(axis="both", labelsize=6)

        axes[1, i].boxplot(after[col].dropna(), vert=True, patch_artist=True,
                           boxprops=dict(facecolor="#a5d6a7"))
        axes[1, i].tick_params(axis="both", labelsize=6)

    axes[0, 0].set_ylabel("Before clipping", fontsize=8)
    axes[1, 0].set_ylabel("After clipping",  fontsize=8)
    fig.suptitle("IQR Outlier Clipping — Before vs After", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=120)
    plt.close()
    print(f"  📊  Saved: outputs/eda/{filename}")


def aqi_to_category(aqi: float) -> str:
    """CPCB AQI bucket mapping."""
    if pd.isna(aqi):        return "Unknown"
    if aqi <= 50:           return "Good"
    if aqi <= 100:          return "Satisfactory"
    if aqi <= 200:          return "Moderate"
    if aqi <= 300:          return "Poor"
    if aqi <= 400:          return "Very Poor"
    return "Severe"


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    # ── 1. Load ──────────────────────────────
    section("Step 1 — Load city_hour.csv")
    check_file(RAW_PATH)
    df_all = pd.read_csv(RAW_PATH)
    print(f"  Full dataset shape : {df_all.shape}")
    print(f"  Cities available   : {sorted(df_all['City'].unique())}")

    # ── 2. Filter to Delhi ───────────────────
    section(f"Step 2 — Filter to {TARGET_CITY}")
    df = df_all[df_all["City"] == TARGET_CITY].copy()
    print(f"  {TARGET_CITY} rows : {len(df):,}")

    # ── 3. Parse datetime & set index ────────
    section("Step 3 — Parse Datetime & Schema Validation")
    df["Datetime"] = pd.to_datetime(df["Datetime"], dayfirst=False)
    df.sort_values("Datetime", inplace=True)
    df.set_index("Datetime", inplace=True)
    df.index.name = "datetime"

    # Keep only relevant numeric columns that exist
    existing_num = [c for c in NUMERIC_COLS if c in df.columns]
    missing_cols  = [c for c in NUMERIC_COLS if c not in df.columns]
    if missing_cols:
        print(f"  ⚠️  Columns not found (skipped): {missing_cols}")

    print(f"  Date range  : {df.index.min()} -> {df.index.max()}")
    print(f"  Total rows  : {len(df):,}")
    print(f"  Columns     : {list(df.columns)}")

    # ── 4. Missingness (before) ───────────────
    plot_missingness(df, "miss_before.png", f"Missing Values — {TARGET_CITY} (Before Imputation)")
    missingness_report(df, "Before Imputation")

    # ── 5. MICE Imputation ────────────────────
    section("Step 4 — MICE Imputation (IterativeImputer)")
    imp = IterativeImputer(max_iter=10, random_state=42, verbose=0)
    arr_imputed = imp.fit_transform(df[existing_num])
    df_imputed  = pd.DataFrame(arr_imputed, columns=existing_num, index=df.index)

    # Merge imputed numeric back into df
    df[existing_num] = df_imputed[existing_num]

    # Ensure no negatives introduced by imputer (physical constraint)
    for col in existing_num:
        df[col] = df[col].clip(lower=0)

    missingness_report(df, "After Imputation")

    # ── 6. IQR Outlier Clipping ───────────────
    section("Step 5 — IQR Outlier Clipping")
    df_before_clip = df[existing_num].copy()

    clip_log = []
    for col in existing_num:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - IQR_MULTIPLIER * IQR
        upper = Q3 + IQR_MULTIPLIER * IQR
        n_clipped = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        clip_log.append({"Column": col, "Lower_fence": round(lower, 2),
                         "Upper_fence": round(upper, 2), "Rows_clipped": n_clipped})

    clip_df = pd.DataFrame(clip_log)
    print(clip_df.to_string(index=False))

    plot_boxplots(df_before_clip, df[existing_num], "boxplot_before_after.png")

    # Hard domain clamp: AQI is physically bounded [0, 500] by CPCB definition.
    # IQR upper fence (626.5) allowed values > 500 to remain — cap them here.
    if "AQI" in df.columns:
        n_above = (df["AQI"] > 500).sum()
        df["AQI"] = df["AQI"].clip(lower=0, upper=500)
        if n_above:
            print(f"  ℹ️   Clamped {n_above} AQI rows above 500 → 500 (CPCB scale ceiling)")

    # ── 7. Re-derive AQI_Bucket ───────────────
    section("Step 6 — Re-derive AQI_Bucket")
    df["AQI_Bucket"] = df["AQI"].apply(aqi_to_category)
    print(df["AQI_Bucket"].value_counts().to_string())

    # ── 8. Validation ─────────────────────────
    section("Step 7 — Output Validation")

    checks = {
        "Zero NaN values"           : df[existing_num].isnull().sum().sum() == 0,
        "AQI in [0, 500]"           : df["AQI"].between(0, 500).all(),
        "No negative pollutants"    : (df[[c for c in existing_num if c != "AQI"]] >= 0).all().all(),
        "Datetime index sorted"     : df.index.is_monotonic_increasing,
    }
    all_pass = True
    for check, result in checks.items():
        status = "✅" if result else "❌"
        if not result:
            all_pass = False
        print(f"  {status}  {check}")

    if not all_pass:
        print("\n  ⚠️  Some checks failed — review before proceeding to Phase 2.")
    else:
        print("\n  🎉  All checks passed!")

    # ── 9. Save ───────────────────────────────
    section("Step 8 — Save clean_aqi.csv")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH)
    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"  Saved  → {OUT_PATH}")
    print(f"  Shape  : {df.shape}")
    print(f"  Size   : {size_kb:.1f} KB")
    print(f"\n  🏁  Phase 1 complete. All downstream phases now read:  {OUT_PATH}\n")


if __name__ == "__main__":
    main()
