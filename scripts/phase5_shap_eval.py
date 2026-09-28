"""
Phase 5 — Interpretability (SHAP) & Evaluation
DSN2098 · Group 65 · Air Quality Analytics
Owner: Amrit Raj

Reads:
    models/xgb_reg.pkl
    models/xgb_clf_hazard.pkl
    data/processed/train.csv / val.csv / test.csv

Saves:
    outputs/model/15_shap_summary_bar.png
    outputs/model/16_shap_summary_dot.png
    outputs/model/17_shap_attribution_split.png
    outputs/model/18_shap_waterfall_single.png
    outputs/model/19_rolling_origin_eval.png
    outputs/model/shap_metrics.csv

What this script does:
    1. Compute SHAP values using shap.TreeExplainer on XGBoost models
    2. Generate Global Feature Importance (SHAP summary bar & dot plots)
    3. Separate SHAP contributions into Meteorological vs. Anthropogenic categories
    4. Produce single-prediction Waterfall / Force attribution plot
    5. Perform Time-Ordered / Rolling-Origin Evaluation over expanding time windows
    6. Export comprehensive SHAP metrics and summary plots

Usage:
    python scripts/phase5_shap_eval.py
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
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, recall_score, precision_score
import xgboost as xgb

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
TRAIN_PATH  = "data/processed/train.csv"
VAL_PATH    = "data/processed/val.csv"
TEST_PATH   = "data/processed/test.csv"
MODEL_DIR   = "models"
PLOT_DIR    = "outputs/model"

METEOROLOGICAL_COLS = [
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "day_of_week_sin", "day_of_week_cos", "day_of_year_sin", "day_of_year_cos",
    "is_winter", "is_weekend", "is_night"
]

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def check_file(path):
    if not os.path.exists(path):
        print(f"\nERROR: {path} not found. Ensure previous phases ran successfully.\n")
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

    # 1. Load Data & Models
    section("Load Splits & Pre-trained XGBoost Models")
    for p in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
        check_file(p)

    train = pd.read_csv(TRAIN_PATH, parse_dates=["datetime"], index_col="datetime")
    val   = pd.read_csv(VAL_PATH,   parse_dates=["datetime"], index_col="datetime")
    test  = pd.read_csv(TEST_PATH,  parse_dates=["datetime"], index_col="datetime")

    target_cols = ["AQI", "AQI_ord", "hazard", "AQI_Bucket"]
    feature_cols = [c for c in train.columns if c not in target_cols]

    reg_path = os.path.join(MODEL_DIR, "xgb_reg.pkl")
    check_file(reg_path)
    reg = joblib.load(reg_path)

    clf_path = os.path.join(MODEL_DIR, "xgb_clf_hazard.pkl")
    check_file(clf_path)
    clf_h, hazard_threshold = joblib.load(clf_path)

    print(f"  Test samples : {len(test):,}")
    print(f"  Features     : {len(feature_cols)}")

    # Sample test set for SHAP to speed up evaluation
    sample_size = min(2000, len(test))
    test_sample = test[feature_cols].sample(sample_size, random_state=42)

    # 2. SHAP Explanation for XGBoost Regressor
    section("Compute SHAP Values for XGBoost Regressor")
    explainer_reg = shap.TreeExplainer(reg)
    shap_vals_reg = explainer_reg.shap_values(test_sample)

    # Global SHAP Bar Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_vals_reg, test_sample, plot_type="bar", show=False, max_display=15)
    plt.title("SHAP Global Feature Importance (Regressor)", fontsize=12, fontweight="bold", pad=15)
    save_plot(plt.gcf(), "15_shap_summary_bar.png")

    # Global SHAP Summary Dot Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_vals_reg, test_sample, show=False, max_display=15)
    plt.title("SHAP Summary Plot (Feature Impact & Value Distribution)", fontsize=12, fontweight="bold", pad=15)
    save_plot(plt.gcf(), "16_shap_summary_dot.png")

    # 3. Categorical Attribution: Meteorological vs. Anthropogenic
    section("Meteorological vs. Anthropogenic Attribution Split")
    
    met_indices = [i for i, c in enumerate(feature_cols) if c in METEOROLOGICAL_COLS]
    anth_indices = [i for i, c in enumerate(feature_cols) if c not in METEOROLOGICAL_COLS]

    abs_shap = np.abs(shap_vals_reg)
    mean_met_shap = abs_shap[:, met_indices].sum(axis=1).mean()
    mean_anth_shap = abs_shap[:, anth_indices].sum(axis=1).mean()

    total_impact = mean_met_shap + mean_anth_shap
    pct_met = (mean_met_shap / total_impact) * 100
    pct_anth = (mean_anth_shap / total_impact) * 100

    print(f"  Meteorological Catalyst Contribution  : {mean_met_shap:.2f} ({pct_met:.1f}%)")
    print(f"  Anthropogenic & Temporal Drivers      : {mean_anth_shap:.2f} ({pct_anth:.1f}%)")

    fig, ax = plt.subplots(figsize=(6, 5))
    categories = ["Meteorological Catalysts", "Anthropogenic & Temporal"]
    values = [pct_met, pct_anth]
    colors = ["#5b9bd5", "#ed7d31"]

    bars = ax.bar(categories, values, color=colors, width=0.5, edgecolor="black", linewidth=0.8)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1, f"{height:.1f}%", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Mean |SHAP Value| Share (%)", fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_title("Overall Driver Attribution Split", fontsize=11, fontweight="bold")
    save_plot(fig, "17_shap_attribution_split.png")

    # 4. Single-Prediction Waterfall Plot
    section("Single Prediction Explanation (Waterfall Plot)")
    # Pick a high pollution sample from test set
    high_aqi_idx = test["AQI"].idxmax()
    sample_single = test.loc[[high_aqi_idx], feature_cols]
    
    explainer_exp = shap.Explanation(
        values=explainer_reg.shap_values(sample_single)[0],
        base_values=explainer_reg.expected_value,
        data=sample_single.iloc[0].values,
        feature_names=feature_cols
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.waterfall_plot(explainer_exp, max_display=10, show=False)
    plt.title(f"SHAP Waterfall Plot for Peak AQI Event ({high_aqi_idx.date()})", fontsize=11, fontweight="bold", pad=15)
    save_plot(plt.gcf(), "18_shap_waterfall_single.png")

    # 5. Time-Ordered / Rolling-Origin Validation
    section("Rolling-Origin Backtesting Validation")
    
    # Combine train + val + test for sequential backtest
    full_df = pd.concat([train, val, test]).sort_index()
    X_full, y_full = full_df[feature_cols], full_df["AQI"]

    # Create 4 expanding window folds across time
    n_total = len(full_df)
    window_step = int(n_total * 0.15)
    min_train_size = int(n_total * 0.55)

    rolling_results = []
    
    for fold in range(4):
        train_end_idx = min_train_size + fold * window_step
        test_end_idx = min(train_end_idx + window_step, n_total)

        X_tr_fold = X_full.iloc[:train_end_idx]
        y_tr_fold = y_full.iloc[:train_end_idx]
        X_te_fold = X_full.iloc[train_end_idx:test_end_idx]
        y_te_fold = y_full.iloc[train_end_idx:test_end_idx]

        fold_reg = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
        fold_reg.fit(X_tr_fold, y_tr_fold, verbose=False)

        preds = fold_reg.predict(X_te_fold)
        rmse = np.sqrt(mean_squared_error(y_te_fold, preds))
        mae = mean_absolute_error(y_te_fold, preds)
        r2 = r2_score(y_te_fold, preds)

        start_date = X_te_fold.index.min().date()
        end_date = X_te_fold.index.max().date()
        print(f"  Fold {fold+1} [{start_date} -> {end_date}]: RMSE={rmse:.2f} | MAE={mae:.2f} | R2={r2:.4f}")

        rolling_results.append({
            "Fold": fold + 1,
            "Period": f"{start_date} to {end_date}",
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
            "R2": round(r2, 4)
        })

    rolling_df = pd.DataFrame(rolling_results)
    
    # Plot Rolling-Origin RMSE
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rolling_df["Fold"], rolling_df["RMSE"], marker="o", linewidth=2, color="#4472c4", label="RMSE")
    ax.set_xticks(rolling_df["Fold"])
    ax.set_xticklabels([f"Fold {f}\n({p})" for f, p in zip(rolling_df["Fold"], rolling_df["Period"])], fontsize=8)
    ax.set_ylabel("RMSE", fontsize=10)
    ax.set_title("Rolling-Origin Backtest RMSE Across Expanding Time Windows", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    for idx, row in rolling_df.iterrows():
        ax.annotate(f"{row['RMSE']:.2f}", (row["Fold"], row["RMSE"]), textcoords="offset points", xytext=(0,8), ha="center", fontweight="bold")
    save_plot(fig, "19_rolling_origin_eval.png")

    # 6. Save Summary CSV
    metrics_path = os.path.join(PLOT_DIR, "shap_metrics.csv")
    rolling_df.to_csv(metrics_path, index=False)
    print(f"\n  Saved -> {metrics_path}")

    section("Phase 5 Complete — Summary")
    print("""
  Interpretability & Evaluation Outputs:
    outputs/model/15_shap_summary_bar.png
    outputs/model/16_shap_summary_dot.png
    outputs/model/17_shap_attribution_split.png
    outputs/model/18_shap_waterfall_single.png
    outputs/model/19_rolling_origin_eval.png
    outputs/model/shap_metrics.csv

  Next: Phase 6 (Priyam) — Interactive Dashboard & Deliverable Documentation
    """)

if __name__ == "__main__":
    main()
