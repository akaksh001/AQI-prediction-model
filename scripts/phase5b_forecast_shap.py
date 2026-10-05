"""
Phase 5b — SHAP (24 h forecast model) + rolling-origin backtest
DSN2098 · Group 65 · Air Quality Analytics

Reads : data/processed/forecast/*.csv, features.json
        models/forecast/xgb_reg_h24_weather.pkl
Saves : outputs/forecast/shap_group_share.png
        outputs/forecast/shap_top_features.png
        outputs/forecast/backtest_h24.csv
        outputs/forecast/backtest_h24.png

SHAP groups: AQI history / Pollutants / Weather / Time.
Backtest: 3 expanding-window folds for the 24 h horizon, with and without weather,
with a purge gap so training targets never reach into the test fold.

Usage:
    python scripts/phase5b_forecast_shap.py
"""
import os
import sys
import json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

DATA_DIR  = "data/processed/forecast"
MODEL_DIR = "models/forecast"
OUT_DIR   = "outputs/forecast"
H = 24


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def feature_cols(groups, variant):
    names = ["AQI history", "Pollutants", "Time"] + (["Weather"] if variant == "weather" else [])
    return [c for g in names for c in groups[g]]


def main():
    model_path = os.path.join(MODEL_DIR, f"xgb_reg_h{H}_weather.pkl")
    for p in (os.path.join(DATA_DIR, "features.json"), model_path):
        if not os.path.exists(p):
            sys.exit(f"ERROR: {p} not found. Run phase3b and phase4b first.")
    os.makedirs(OUT_DIR, exist_ok=True)

    meta = json.load(open(os.path.join(DATA_DIR, "features.json")))
    groups, gap = meta["groups"], meta["gap"]
    rd = lambda n: pd.read_csv(os.path.join(DATA_DIR, f"{n}.csv"), parse_dates=["datetime"], index_col="datetime")
    train, val, test = rd("train"), rd("val"), rd("test")
    cols = feature_cols(groups, "weather")
    feat_to_group = {c: g for g, cs in groups.items() for c in cs}

    # ── SHAP ──────────────────────────────────
    section(f"SHAP — {H} h regressor (with weather)")
    reg = joblib.load(model_path)
    sample = test[cols].sample(min(2000, len(test)), random_state=42)
    sv = shap.TreeExplainer(reg).shap_values(sample)
    abs_sv = np.abs(sv)

    shares = {}
    for g in groups:
        idx = [i for i, c in enumerate(cols) if feat_to_group[c] == g]
        shares[g] = abs_sv[:, idx].sum(axis=1).mean() if idx else 0.0
    total = sum(shares.values())
    for g, v in sorted(shares.items(), key=lambda kv: -kv[1]):
        print(f"  {g:<12}: {v / total * 100:5.1f}%")

    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(shares)
    vals = [shares[g] / total * 100 for g in names]
    bars = ax.bar(names, vals, color=["#4472c4", "#ed7d31", "#2e9e8f", "#9b59b6"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_ylabel("Share of mean |SHAP| (%)")
    ax.set_ylim(0, max(vals) * 1.2)
    ax.set_title(f"What drives the {H}-hour-ahead forecast", fontweight="bold")
    fig.savefig(os.path.join(OUT_DIR, "shap_group_share.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(9, 6))
    shap.summary_plot(sv, sample, plot_type="bar", show=False, max_display=15)
    plt.title(f"Top features — {H} h forecast", fontweight="bold")
    plt.savefig(os.path.join(OUT_DIR, "shap_top_features.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Backtest ──────────────────────────────
    section(f"Rolling-origin backtest — {H} h horizon (3 folds, purge gap)")
    full = pd.concat([train, val, test]).sort_index()
    n = len(full)
    step, start = int(n * 0.15), int(n * 0.55)
    y = full[f"AQI_t{H}"]
    rows = []
    for fold in range(3):
        tr_end = start + fold * step
        te_end = min(tr_end + step, n)
        tr = slice(0, tr_end - gap)            # purge gap
        te = slice(tr_end, te_end)
        period = f"{full.index[tr_end].date()} to {full.index[te_end - 1].date()}"
        persist = float(np.sqrt(mean_squared_error(y.iloc[te], full["AQI"].iloc[te])))
        row = {"Fold": fold + 1, "Period": period, "Persistence_RMSE": round(persist, 2)}
        for variant in ["noweather", "weather"]:
            c = feature_cols(groups, variant)
            m = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
            m.fit(full[c].iloc[tr], y.iloc[tr], verbose=False)
            p = m.predict(full[c].iloc[te])
            row[f"{variant}_RMSE"] = round(float(np.sqrt(mean_squared_error(y.iloc[te], p))), 2)
            row[f"{variant}_R2"] = round(float(r2_score(y.iloc[te], p)), 4)
        rows.append(row)
        print(f"  Fold {fold+1} [{period}]  persistence={row['Persistence_RMSE']:.2f}  "
              f"no-weather={row['noweather_RMSE']:.2f}  weather={row['weather_RMSE']:.2f}")

    bt = pd.DataFrame(rows)
    bt.to_csv(os.path.join(OUT_DIR, f"backtest_h{H}.csv"), index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    for col, lab, colr in [("Persistence_RMSE", "Persistence", "#9aa5b1"),
                           ("noweather_RMSE", "XGBoost, no weather", "#4472c4"),
                           ("weather_RMSE", "XGBoost + weather", "#ed7d31")]:
        ax.plot(bt["Fold"], bt[col], marker="o", label=lab, color=colr)
    ax.set_xticks(bt["Fold"])
    ax.set_xticklabels([f"Fold {f}\n{p}" for f, p in zip(bt["Fold"], bt["Period"])], fontsize=7)
    ax.set_ylabel("RMSE")
    ax.set_title(f"Rolling-origin backtest, {H} h ahead", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.savefig(os.path.join(OUT_DIR, f"backtest_h{H}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved outputs -> {OUT_DIR}/")


if __name__ == "__main__":
    main()