"""
Phase 4b — Forecast Models (1 h, 6 h, 24 h ahead) with a weather ablation
DSN2098 · Group 65 · Air Quality Analytics

Reads : data/processed/forecast/train.csv, val.csv, test.csv, features.json
Saves : models/forecast/xgb_reg_h{H}_{variant}.pkl
        models/forecast/xgb_clf_h{H}_{variant}.pkl   (model, threshold)
        outputs/forecast/forecast_comparison.csv
        outputs/forecast/forecast_rmse_by_horizon.png

The regressor predicts the change from current AQI (AQI_t+H minus AQI_now).
For each horizon it trains XGBoost twice:
    noweather = AQI history + pollutants + time
    weather   = the above + weather features
and compares both with the persistence baseline ("AQI stays as it is now").

Usage:
    python scripts/phase4b_forecast_models.py
"""
import os
import sys
import json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             recall_score, precision_score, f1_score)

DATA_DIR  = "data/processed/forecast"
MODEL_DIR = "models/forecast"
OUT_DIR   = "outputs/forecast"
THRESHOLD = 0.40          # hazard threshold, tuned for recall


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def feature_cols(groups, variant):
    names = ["AQI history", "Pollutants", "Time"] + (["Weather"] if variant == "weather" else [])
    return [c for g in names for c in groups[g]]


def reg_metrics(y, p):
    return {"RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": float(mean_absolute_error(y, p)), "R2": float(r2_score(y, p))}


def clf_metrics(y, p):
    return {"Recall": float(recall_score(y, p, zero_division=0)),
            "Precision": float(precision_score(y, p, zero_division=0)),
            "F1": float(f1_score(y, p, zero_division=0))}


def main():
    if not os.path.exists(os.path.join(DATA_DIR, "features.json")):
        sys.exit("ERROR: run scripts/phase3b_forecast_features.py first.")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    meta = json.load(open(os.path.join(DATA_DIR, "features.json")))
    groups, horizons = meta["groups"], meta["horizons"]
    rd = lambda n: pd.read_csv(os.path.join(DATA_DIR, f"{n}.csv"), parse_dates=["datetime"], index_col="datetime")
    train, val, test = rd("train"), rd("val"), rd("test")
    print(f"  Train {len(train):,} | Val {len(val):,} | Test {len(test):,}")

    rows = []
    for h in horizons:
        section(f"Horizon = {h} hour(s) ahead")
        y_tr, y_va, y_te = train[f"AQI_t{h}"], val[f"AQI_t{h}"], test[f"AQI_t{h}"]
        h_tr, h_va, h_te = train[f"hazard_t{h}"], val[f"hazard_t{h}"], test[f"hazard_t{h}"]

        # Persistence baselines: "AQI stays as it is now"
        m = reg_metrics(y_te, test["AQI"])
        rows.append({"Horizon_h": h, "Model": "Persistence", "Task": "AQI", **m})
        print(f"  Persistence (AQI)     RMSE={m['RMSE']:.2f}  R2={m['R2']:.4f}")
        now_severe = (test["AQI_Bucket"] == "Severe").astype(int)
        c = clf_metrics(h_te, now_severe)
        rows.append({"Horizon_h": h, "Model": "Persistence", "Task": "Hazard", **c})
        print(f"  Persistence (hazard)  Recall={c['Recall']:.4f}  F1={c['F1']:.4f}")

        for variant in ["noweather", "weather"]:
            cols = feature_cols(groups, variant)

            reg = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=4,
                                   subsample=0.8, colsample_bytree=0.8, random_state=42,
                                   n_jobs=-1, early_stopping_rounds=50)
            # Predict the CHANGE from the current AQI (delta), then add it back.
            # This starts from persistence and lets trees learn only the correction.
            reg.fit(train[cols], y_tr - train["AQI"],
                    eval_set=[(val[cols], y_va - val["AQI"])], verbose=False)
            m = reg_metrics(y_te, test["AQI"] + reg.predict(test[cols]))
            rows.append({"Horizon_h": h, "Model": f"XGBoost_{variant}", "Task": "AQI", **m})
            print(f"  XGB {variant:<10} AQI  RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}  R2={m['R2']:.4f}"
                  f"  ({len(cols)} features)")
            joblib.dump(reg, os.path.join(MODEL_DIR, f"xgb_reg_h{h}_{variant}.pkl"))

            spw = float((h_tr == 0).sum() / max((h_tr == 1).sum(), 1))
            clf = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6,
                                    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                                    eval_metric="logloss", random_state=42, n_jobs=-1,
                                    early_stopping_rounds=30)
            clf.fit(train[cols], h_tr, eval_set=[(val[cols], h_va)], verbose=False)
            pred = (clf.predict_proba(test[cols])[:, 1] >= THRESHOLD).astype(int)
            c = clf_metrics(h_te, pred)
            rows.append({"Horizon_h": h, "Model": f"XGBoost_{variant}", "Task": "Hazard", **c})
            print(f"  XGB {variant:<10} Hazard Recall={c['Recall']:.4f}  Prec={c['Precision']:.4f}  F1={c['F1']:.4f}")
            joblib.dump((clf, THRESHOLD), os.path.join(MODEL_DIR, f"xgb_clf_h{h}_{variant}.pkl"))

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "forecast_comparison.csv"), index=False)

    # ── Summary table ─────────────────────────
    section("Summary — AQI regression RMSE (lower is better)")
    reg = res[res["Task"] == "AQI"].pivot(index="Horizon_h", columns="Model", values="RMSE")
    reg = reg[["Persistence", "XGBoost_noweather", "XGBoost_weather"]]
    reg["Gain_vs_persistence_%"]   = (1 - reg["XGBoost_weather"] / reg["Persistence"]) * 100
    reg["Weather_gain_%"]          = (1 - reg["XGBoost_weather"] / reg["XGBoost_noweather"]) * 100
    print(reg.round(2).to_string())

    # ── Plot ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(reg))
    for i, (col, lab, colr) in enumerate([("Persistence", "Persistence (no ML)", "#9aa5b1"),
                                          ("XGBoost_noweather", "XGBoost, no weather", "#4472c4"),
                                          ("XGBoost_weather", "XGBoost + weather", "#ed7d31")]):
        bars = ax.bar(x + (i - 1) * 0.27, reg[col], 0.27, label=lab, color=colr)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{b.get_height():.1f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} h ahead" for h in reg.index])
    ax.set_ylabel("Test RMSE (AQI points)")
    ax.set_title("Forecast error by horizon", fontweight="bold")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(OUT_DIR, "forecast_rmse_by_horizon.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved -> {OUT_DIR}/forecast_comparison.csv and forecast_rmse_by_horizon.png")
    print(f"  Models saved -> {MODEL_DIR}/")


if __name__ == "__main__":
    main()