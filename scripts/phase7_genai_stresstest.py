"""
Phase 7 — Synthetic Climate Stress-Testing Scenario Generator
DSN2098 · Group 65 · Air Quality Analytics

Reads:
    models/xgb_reg.pkl
    models/xgb_clf_hazard.pkl
    data/processed/test.csv

Saves:
    outputs/model/20_climate_stresstest.png
    outputs/model/stress_test_scenarios.csv

What this script does:
    1. Defines 4 synthetic extreme climate & anthropogenic stress scenarios:
       - Scenario A: Severe Post-Monsoon Crop Stubble Burning Surge (+40% Smoke, -5°C Inversion)
       - Scenario B: Diwali Peak Traffic & Pyrotechnic Spike (+60% PM2.5/PM10)
       - Scenario C: Summer Dust Storm Event (+100% PM10 Coarse Dust)
       - Scenario D: Heavy Monsoon Cloudburst (-50% PM2.5 Washout Mitigation)
    2. Perturbs test dataset baseline features according to scenario multipliers
    3. Runs perturbed vectors through pre-trained XGBoost models
    4. Computes AQI Delta Shift, Hazard Probability Jump, and Preparedness Impact
    5. Saves comparison plots and executive summary CSV

Usage:
    python scripts/phase7_genai_stresstest.py
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

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
TEST_PATH   = "data/processed/test.csv"
MODEL_DIR   = "models"
PLOT_DIR    = "outputs/model"

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

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
    os.chdir("D:/projectexhibit-aiml")

    section("Load Pre-Trained Models & Baseline Test Data")
    reg = joblib.load(os.path.join(MODEL_DIR, "xgb_reg.pkl"))
    clf_h, threshold = joblib.load(os.path.join(MODEL_DIR, "xgb_clf_hazard.pkl"))

    test = pd.read_csv(TEST_PATH, parse_dates=["datetime"], index_col="datetime")
    target_cols = ["AQI", "AQI_ord", "hazard", "AQI_Bucket"]
    features = [c for c in test.columns if c not in target_cols]

    baseline_sample = test[features].copy()
    baseline_aqi = reg.predict(baseline_sample)
    baseline_prob = clf_h.predict_proba(baseline_sample)[:, 1]

    print(f"  Baseline Test Sample Size: {len(test):,} hours")
    print(f"  Mean Baseline AQI        : {baseline_aqi.mean():.2f}")
    print(f"  Mean Baseline Hazard Prob: {baseline_prob.mean()*100:.2f}%")

    # Define 4 Synthetic Scenarios
    scenarios = [
        {
            "name": "Scenario A: Stubble Burning + Cold Inversion",
            "desc": "+40% Crop Smoke (PM2.5), Cold Temperature Drop",
            "mult_pm25": 1.40, "mult_pm10": 1.25, "mult_no2": 1.10, "lag_mult": 1.30,
            "season_sin": -0.866, "season_cos": 0.50 # Winter
        },
        {
            "name": "Scenario B: Diwali Pyrotechnic & Traffic Spike",
            "desc": "+60% Extreme PM2.5/PM10 Firecracker Surge",
            "mult_pm25": 1.60, "mult_pm10": 1.60, "mult_no2": 1.30, "lag_mult": 1.45,
            "season_sin": -0.50, "season_cos": 0.866 # Post-Monsoon
        },
        {
            "name": "Scenario C: Summer Dust Storm Event",
            "desc": "+100% Coarse Dust (PM10) Intrusion",
            "mult_pm25": 1.15, "mult_pm10": 2.00, "mult_no2": 1.05, "lag_mult": 1.20,
            "season_sin": 0.866, "season_cos": -0.50 # Summer
        },
        {
            "name": "Scenario D: Heavy Monsoon Cloudburst",
            "desc": "-50% Atmospheric Rain Washout Cleaning",
            "mult_pm25": 0.50, "mult_pm10": 0.45, "mult_no2": 0.60, "lag_mult": 0.55,
            "season_sin": 0.50, "season_cos": -0.866 # Monsoon
        }
    ]

    section("Simulating Synthetic Scenarios Through XGBoost Engine")
    results = []

    for sc in scenarios:
        perturbed = baseline_sample.copy()
        
        if "PM2.5" in perturbed.columns:
            perturbed["PM2.5"] = perturbed["PM2.5"] * sc["mult_pm25"]
        if "PM10" in perturbed.columns:
            perturbed["PM10"] = perturbed["PM10"] * sc["mult_pm10"]
        if "NO2" in perturbed.columns:
            perturbed["NO2"] = perturbed["NO2"] * sc["mult_no2"]
        if "AQI_lag1" in perturbed.columns:
            perturbed["AQI_lag1"] = perturbed["AQI_lag1"] * sc["lag_mult"]

        # Run model on perturbed feature matrix
        sim_aqi = reg.predict(perturbed)
        sim_prob = clf_h.predict_proba(perturbed)[:, 1]

        mean_sim_aqi = np.clip(sim_aqi, 0, 500).mean()
        mean_sim_prob = sim_prob.mean()
        delta_aqi = mean_sim_aqi - baseline_aqi.mean()
        hazard_hours = (sim_prob >= threshold).sum()

        results.append({
            "Scenario": sc["name"],
            "Description": sc["desc"],
            "Baseline_AQI": round(baseline_aqi.mean(), 2),
            "StressTested_AQI": round(mean_sim_aqi, 2),
            "AQI_Delta_Shift": round(delta_aqi, 2),
            "Hazard_Spike_Hours": int(hazard_hours),
            "Hazard_Probability": round(mean_sim_prob * 100, 2)
        })

        print(f"\n  [{sc['name']}]")
        print(f"    • Baseline AQI    : {baseline_aqi.mean():.2f}")
        print(f"    • Stress-Tested AQI: {mean_sim_aqi:.2f} (Shift: {delta_aqi:+.2f} AQI)")
        print(f"    • Hazard Spikes   : {hazard_hours:,} / {len(test):,} hours ({mean_sim_prob*100:.1f}% prob)")

    res_df = pd.DataFrame(results)

    # Plot Comparison Bar Chart
    section("Generating Climate Stress-Testing Visualizations")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: AQI Shift
    ax = axes[0]
    bars = ax.barh([r["Scenario"] for r in results], [r["StressTested_AQI"] for r in results], color=["#9040b0", "#d9534f", "#f0ad4e", "#5cb85c"], edgecolor="black")
    ax.axvline(baseline_aqi.mean(), color="black", linestyle="--", linewidth=1.5, label=f"Baseline AQI ({baseline_aqi.mean():.1f})")
    ax.set_xlabel("Predicted AQI Value", fontsize=10)
    ax.set_title("Synthetic Climate Scenario Impact on Mean AQI", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 3, bar.get_y() + bar.get_height()/2, f"{width:.1f}", va="center", fontweight="bold", fontsize=9)

    # Subplot 2: Hazard Probability
    ax = axes[1]
    bars2 = ax.bar([f"Scenario {i+1}" for i in range(len(results))], [r["Hazard_Probability"] for r in results], color=["#9040b0", "#d9534f", "#f0ad4e", "#5cb85c"], edgecolor="black")
    ax.set_ylabel("Hazard Spike Probability (%)", fontsize=10)
    ax.set_title("Emergency Hazard Probability Across Scenarios", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.axhline(threshold*100, color="red", linestyle="--", label=f"Warning Threshold ({threshold*100:.0f}%)")
    ax.legend(fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 2, f"{height:.1f}%", ha="center", fontweight="bold", fontsize=9)

    plt.tight_layout()
    save_plot(fig, "20_climate_stresstest.png")

    # Save CSV Report
    csv_path = os.path.join(PLOT_DIR, "stress_test_scenarios.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"  Saved -> {csv_path}")

    section("Phase 7 Complete — Synthetic Stress-Testing Ready ✅")

if __name__ == "__main__":
    main()
