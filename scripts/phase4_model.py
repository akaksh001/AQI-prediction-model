"""
Phase 4 — XGBoost Model Training & Benchmarking
DSN2098 · Group 65 · Air Quality Analytics
Owner: Anshu Sharma

Reads:  data/processed/train.csv / val.csv / test.csv   (Phase 3 output)
Saves:
    models/xgb_reg.pkl          <- XGBoost regressor  (AQI continuous)
    models/xgb_clf_hazard.pkl   <- XGBoost classifier (hazard 0/1)
    models/xgb_clf_cat.pkl      <- XGBoost classifier (AQI category 0-5)
    outputs/model/11_reg_pred_vs_actual.png
    outputs/model/12_clf_confusion.png
    outputs/model/13_benchmark.png
    outputs/model/14_xgb_feature_importance.png
    outputs/model/metrics_summary.csv

What this script does:
    1. Load splits from Phase 3 (train / val / test)
    2. Train XGBoost Regressor   -> predict AQI value (MSE loss)
    3. Train XGBoost Hazard Clf  -> predict Severe spike (binary, recall-optimised)
    4. Train XGBoost Category Clf-> predict AQI_ord 0-5 (multiclass)
    5. Benchmark Reg against RandomForest + LightGBM on val set
    6. Evaluate all models on held-out test set
    7. Save models + plots + metrics CSV

Usage:
    python scripts/phase4_model.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.ensemble       import RandomForestRegressor
from sklearn.metrics        import (mean_squared_error, mean_absolute_error,
                                    r2_score, classification_report,
                                    confusion_matrix, ConfusionMatrixDisplay,
                                    f1_score, recall_score, precision_score)
import xgboost as xgb

# LightGBM is optional — skip gracefully if not installed
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("  [INFO] LightGBM not installed — skipping LGB benchmark.")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
TRAIN_PATH  = "data/processed/train.csv"
VAL_PATH    = "data/processed/val.csv"
TEST_PATH   = "data/processed/test.csv"
MODEL_DIR   = "models"
PLOT_DIR    = "outputs/model"

BUCKET_ORDER = ["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"]
BUCKET_ORD   = {b: i for i, b in enumerate(BUCKET_ORDER)}

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def check_file(path):
    if not os.path.exists(path):
        print(f"\nERROR: {path} not found. Run phase3_features.py first.\n")
        sys.exit(1)

def save_plot(fig, filename):
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")

def reg_metrics(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  {label:<25}  RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.4f}")
    return {"RMSE": round(rmse,4), "MAE": round(mae,4), "R2": round(r2,4)}

def clf_metrics(y_true, y_pred, label=""):
    rec  = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1   = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    print(f"  {label:<25}  Recall={rec:.4f}  Prec={prec:.4f}  F1={f1:.4f}")
    return {"Recall": round(rec,4), "Precision": round(prec,4), "F1": round(f1,4)}

# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
def load_splits():
    section("Load Train / Val / Test Splits")
    for p in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
        check_file(p)

    def load(path):
        df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
        return df

    train = load(TRAIN_PATH)
    val   = load(VAL_PATH)
    test  = load(TEST_PATH)

    # Identify feature columns (everything except target columns)
    target_cols = ["AQI", "AQI_ord", "hazard", "AQI_Bucket"]
    feature_cols = [c for c in train.columns if c not in target_cols]

    print(f"  Train  : {train.shape}  ({train.index.min().date()} -> {train.index.max().date()})")
    print(f"  Val    : {val.shape}")
    print(f"  Test   : {test.shape}")
    print(f"  Features : {len(feature_cols)}")
    print(f"  Targets  : {target_cols}")

    return train, val, test, feature_cols

# ──────────────────────────────────────────────
# MODEL 1 — XGBoost Regressor
# ──────────────────────────────────────────────
def train_xgb_regressor(train, val, test, feature_cols):
    section("Model 1 — XGBoost Regressor (predict AQI value)")

    X_tr, y_tr = train[feature_cols], train["AQI"]
    X_val,y_val= val[feature_cols],   val["AQI"]
    X_te, y_te = test[feature_cols],  test["AQI"]

    reg = xgb.XGBRegressor(
        n_estimators      = 800,
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 3,
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        random_state      = 42,
        n_jobs            = -1,
        verbosity         = 0,
        early_stopping_rounds = 30,
        eval_metric       = "rmse",
    )
    reg.fit(X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False)

    best_iter = reg.best_iteration
    print(f"  Best iteration : {best_iter}")

    # Metrics
    print("\n  --- Validation ---")
    val_metrics  = reg_metrics(y_val, reg.predict(X_val),  "XGB Regressor (val)")
    print("\n  --- Test ---")
    test_metrics = reg_metrics(y_te,  reg.predict(X_te),   "XGB Regressor (test)")

    # Plot: predicted vs actual
    y_pred_te = reg.predict(X_te)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Scatter
    ax = axes[0]
    ax.scatter(y_te, y_pred_te, alpha=0.15, s=6, color="#4472c4")
    lo, hi = min(y_te.min(), y_pred_te.min()), max(y_te.max(), y_pred_te.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual AQI",    fontsize=10)
    ax.set_ylabel("Predicted AQI", fontsize=10)
    ax.set_title(f"XGB Regressor — Predicted vs Actual\n"
                 f"RMSE={test_metrics['RMSE']}  MAE={test_metrics['MAE']}  R²={test_metrics['R2']}",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)

    # Residuals
    residuals = y_pred_te - y_te.values
    ax = axes[1]
    ax.hist(residuals, bins=60, color="#ed7d31", edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Residual (Predicted - Actual)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Residual Distribution", fontsize=10, fontweight="bold")
    ax.text(0.97, 0.95, f"Mean={residuals.mean():.2f}\nStd={residuals.std():.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    plt.tight_layout()
    save_plot(fig, "11_reg_pred_vs_actual.png")

    # Save model
    model_path = os.path.join(MODEL_DIR, "xgb_reg.pkl")
    joblib.dump(reg, model_path)
    print(f"  Model saved -> {model_path}")

    return reg, feature_cols, val_metrics, test_metrics

# ──────────────────────────────────────────────
# MODEL 2 — XGBoost Hazard Classifier
# ──────────────────────────────────────────────
def train_xgb_hazard_clf(train, val, test, feature_cols):
    section("Model 2 — XGBoost Hazard Classifier (predict Severe spike)")

    X_tr, y_tr = train[feature_cols], train["hazard"]
    X_val,y_val= val[feature_cols],   val["hazard"]
    X_te, y_te = test[feature_cols],  test["hazard"]

    # Compute class weight to upweight Severe (recall-optimised)
    n_neg  = (y_tr == 0).sum()
    n_pos  = (y_tr == 1).sum()
    spw    = round(n_neg / n_pos, 2)
    print(f"  scale_pos_weight = {spw}  (n_neg={n_neg:,} / n_pos={n_pos:,})")

    clf_h = xgb.XGBClassifier(
        n_estimators          = 800,
        max_depth             = 5,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        scale_pos_weight      = spw,
        eval_metric           = "auc",
        use_label_encoder     = False,
        random_state          = 42,
        n_jobs                = -1,
        verbosity             = 0,
        early_stopping_rounds = 30,
    )
    clf_h.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              verbose=False)

    print(f"  Best iteration : {clf_h.best_iteration}")

    # Use 0.4 threshold (lower = higher recall)
    THRESHOLD = 0.40
    prob_val  = clf_h.predict_proba(X_val)[:, 1]
    prob_te   = clf_h.predict_proba(X_te)[:, 1]
    pred_val  = (prob_val >= THRESHOLD).astype(int)
    pred_te   = (prob_te  >= THRESHOLD).astype(int)

    print(f"\n  Threshold = {THRESHOLD}  (tuned for recall)")
    print("\n  --- Validation ---")
    val_clf_m  = clf_metrics(y_val, pred_val, "Hazard Clf (val)")
    print("\n  --- Test (HEADLINE METRIC) ---")
    test_clf_m = clf_metrics(y_te,  pred_te,  "Hazard Clf (test)")

    print("\n  Full test classification report:")
    print(classification_report(y_te, pred_te,
                                 target_names=["Normal","Severe"],
                                 zero_division=0))

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_te, pred_te)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal","Severe"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Hazard Classifier — Test Confusion Matrix\n"
                 f"Recall={test_clf_m['Recall']}  F1={test_clf_m['F1']}",
                 fontsize=10, fontweight="bold")
    save_plot(fig, "12_clf_confusion.png")

    model_path = os.path.join(MODEL_DIR, "xgb_clf_hazard.pkl")
    joblib.dump((clf_h, THRESHOLD), model_path)
    print(f"  Model saved -> {model_path}")

    return clf_h, THRESHOLD, val_clf_m, test_clf_m

# ──────────────────────────────────────────────
# MODEL 3 — XGBoost Multi-class Category Clf
# ──────────────────────────────────────────────
def train_xgb_cat_clf(train, val, test, feature_cols):
    section("Model 3 — XGBoost Category Classifier (AQI_ord 0-5)")

    X_tr, y_tr = train[feature_cols], train["AQI_ord"]
    X_val,y_val= val[feature_cols],   val["AQI_ord"]
    X_te, y_te = test[feature_cols],  test["AQI_ord"]

    clf_c = xgb.XGBClassifier(
        n_estimators          = 800,
        max_depth             = 6,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        objective             = "multi:softprob",
        num_class             = 6,
        eval_metric           = "mlogloss",
        use_label_encoder     = False,
        random_state          = 42,
        n_jobs                = -1,
        verbosity             = 0,
        early_stopping_rounds = 30,
    )
    clf_c.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              verbose=False)

    pred_te = clf_c.predict(X_te)
    f1_mac  = f1_score(y_te, pred_te, average="macro",  zero_division=0)
    f1_wt   = f1_score(y_te, pred_te, average="weighted", zero_division=0)
    print(f"  Best iteration : {clf_c.best_iteration}")
    print(f"  Test F1 (macro)    : {f1_mac:.4f}")
    print(f"  Test F1 (weighted) : {f1_wt:.4f}")
    print("\n  Per-class report:")
    print(classification_report(y_te, pred_te,
                                 target_names=BUCKET_ORDER,
                                 zero_division=0))

    model_path = os.path.join(MODEL_DIR, "xgb_clf_cat.pkl")
    joblib.dump(clf_c, model_path)
    print(f"  Model saved -> {model_path}")

    return clf_c, f1_mac, f1_wt

# ──────────────────────────────────────────────
# BENCHMARK vs RF + LGB
# ──────────────────────────────────────────────
def benchmark(train, val, test, feature_cols, xgb_val_m, xgb_test_m):
    section("Benchmark — XGBoost vs RandomForest vs LightGBM")

    X_tr, y_tr = train[feature_cols], train["AQI"]
    X_val,y_val= val[feature_cols],   val["AQI"]
    X_te, y_te = test[feature_cols],  test["AQI"]

    results = {}

    # XGBoost (already trained — re-use metrics)
    results["XGBoost"] = {"val": xgb_val_m, "test": xgb_test_m}
    print(f"  XGBoost (pre-trained): val RMSE={xgb_val_m['RMSE']}  test RMSE={xgb_test_m['RMSE']}")

    # RandomForest
    print("\n  Training RandomForest (n=200)...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=12,
                                n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)
    rf_val_m  = reg_metrics(y_val, rf.predict(X_val),  "RandomForest (val)")
    rf_test_m = reg_metrics(y_te,  rf.predict(X_te),   "RandomForest (test)")
    results["RandomForest"] = {"val": rf_val_m, "test": rf_test_m}

    # LightGBM
    if HAS_LGB:
        print("\n  Training LightGBM...")
        lgb_m = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.05,
                                    max_depth=6, random_state=42,
                                    n_jobs=-1, verbose=-1)
        lgb_m.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(30, verbose=False),
                             lgb.log_evaluation(-1)])
        lgb_val_m  = reg_metrics(y_val, lgb_m.predict(X_val),  "LightGBM (val)")
        lgb_test_m = reg_metrics(y_te,  lgb_m.predict(X_te),   "LightGBM (test)")
        results["LightGBM"] = {"val": lgb_val_m, "test": lgb_test_m}

    # Benchmark plot
    models  = list(results.keys())
    rmse_val  = [results[m]["val"]["RMSE"]  for m in models]
    rmse_test = [results[m]["test"]["RMSE"] for m in models]
    r2_test   = [results[m]["test"]["R2"]   for m in models]

    colors = ["#4472c4", "#ed7d31", "#70ad47"][:len(models)]
    x      = np.arange(len(models))
    width  = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # RMSE comparison
    ax = axes[0]
    b1 = ax.bar(x - width/2, rmse_val,  width, label="Val RMSE",  color=[c+"99" for c in colors])
    b2 = ax.bar(x + width/2, rmse_test, width, label="Test RMSE", color=colors)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("RMSE (lower is better)"); ax.set_title("RMSE Comparison", fontweight="bold")
    ax.legend(fontsize=9)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    # R² comparison
    ax = axes[1]
    bars = ax.bar(models, r2_test, color=colors, edgecolor="white")
    ax.set_ylabel("R² (higher is better)"); ax.set_title("R² on Test Set", fontweight="bold")
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, r2_test):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Model Benchmark — AQI Regression", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_plot(fig, "13_benchmark.png")

    return results

# ──────────────────────────────────────────────
# FEATURE IMPORTANCE PLOT
# ──────────────────────────────────────────────
def plot_feature_importance(reg, feature_cols):
    section("Feature Importance — XGBoost Regressor")

    imp = pd.Series(reg.feature_importances_, index=feature_cols)
    top20 = imp.sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = []
    for feat in top20.index:
        if "lag" in feat or "roll" in feat or "trend" in feat or "seasonal" in feat or "resid" in feat:
            colors.append("#4472c4")  # blue = engineered temporal
        elif any(p in feat for p in ["PM","NO","CO","SO","O3","NH","Benz","Tol","Xyl"]):
            colors.append("#ed7d31")  # orange = raw pollutant
        else:
            colors.append("#70ad47")  # green = cyclical / calendar

    bars = ax.barh(top20.index[::-1], top20.values[::-1],
                   color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Feature Importance Score", fontsize=10)
    ax.set_title("Top 20 Features — XGBoost Regressor\n"
                 "(blue=temporal/engineered  orange=raw pollutant  green=calendar)",
                 fontsize=11, fontweight="bold")

    for bar, val in zip(bars, top20.values[::-1]):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=7.5)

    plt.tight_layout()
    save_plot(fig, "14_xgb_feature_importance.png")

    print("\n  Top 10 features:")
    for feat, score in top20.head(10).items():
        bar = "#" * int(score * 500)
        print(f"    {feat:<28} {score:.4f}  {bar}")

# ──────────────────────────────────────────────
# SAVE METRICS SUMMARY
# ──────────────────────────────────────────────
def save_metrics(bench_results, test_clf_m, f1_mac, f1_wt):
    section("Metrics Summary")

    rows = []
    for model, res in bench_results.items():
        rows.append({
            "Model": model, "Task": "Regression",
            "Val_RMSE": res["val"]["RMSE"], "Val_MAE": res["val"]["MAE"], "Val_R2": res["val"]["R2"],
            "Test_RMSE": res["test"]["RMSE"],"Test_MAE": res["test"]["MAE"],"Test_R2": res["test"]["R2"],
        })

    rows.append({
        "Model": "XGBoost_Hazard", "Task": "Binary Classification (Severe)",
        "Test_Recall": test_clf_m["Recall"],
        "Test_Precision": test_clf_m["Precision"],
        "Test_F1": test_clf_m["F1"],
    })
    rows.append({
        "Model": "XGBoost_Category", "Task": "Multiclass (AQI_ord 0-5)",
        "Test_F1_macro": f1_mac,
        "Test_F1_weighted": f1_wt,
    })

    metrics_df = pd.DataFrame(rows)
    metrics_path = os.path.join(PLOT_DIR, "metrics_summary.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n  Saved -> {metrics_path}")
    print(metrics_df.to_string(index=False))

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR,  exist_ok=True)

    train, val, test, feature_cols = load_splits()

    reg,   feature_cols, xgb_val_m,  xgb_test_m  = train_xgb_regressor (train, val, test, feature_cols)
    clf_h, threshold,    val_clf_m,   test_clf_m  = train_xgb_hazard_clf(train, val, test, feature_cols)
    clf_c, f1_mac,       f1_wt                    = train_xgb_cat_clf   (train, val, test, feature_cols)

    bench_results = benchmark(train, val, test, feature_cols, xgb_val_m, xgb_test_m)

    plot_feature_importance(reg, feature_cols)
    save_metrics(bench_results, test_clf_m, f1_mac, f1_wt)

    section("Phase 4 Complete")
    print(f"""
  Models saved to:  models/
    xgb_reg.pkl          <- regression head
    xgb_clf_hazard.pkl   <- hazard spike classifier
    xgb_clf_cat.pkl      <- category classifier (0-5)

  Plots saved to:  outputs/model/
    11_reg_pred_vs_actual.png
    12_clf_confusion.png
    13_benchmark.png
    14_xgb_feature_importance.png
    metrics_summary.csv

  Next: Phase 5 (Amrit) — SHAP attribution & rolling-origin evaluation
    """)


if __name__ == "__main__":
    main()
