# 🌫️ AIR QUALITY ANALYTICS & PREDICTION SYSTEM

### DSN2098 · Group 65 · Final Project Repository

**Team Members:**

- **Aryavardhan** — Phase 1: Data Collection & MICE Imputation
- **Yugank Shakya** — Phase 2: Exploratory Data Analysis (EDA)
- **Akaksh Samdani** — Phase 3: Feature Engineering (lags, rolling stats, cyclical encoding; STL for EDA)
- **Anshu Sharma** — Phase 4: XGBoost Modeling & Benchmarking
- **Amrit Raj** — Phase 5: SHAP Interpretability & Backtesting
- **Priyam Kumar** — Phase 6: Interactive Dashboard & Documentation

---

## 📌 Project Overview

Standard AQI monitoring reports a number but not a reason. This project delivers an **end-to-end Explainable AI Air Quality Analytics System** built on about 5.5 years of hourly CPCB data (Jan 2015 – Jul 2020, 48,192 hourly records) for Delhi.

The system estimates the AQI of the current hour from pollutant readings and recent AQI history, flags severe-pollution hours, classifies the CPCB category, and explains each prediction with SHAP.

> **Scope note:** the task is **AQI estimation (nowcasting)**, not multi-hour forecasting. Same-hour pollutant readings are model inputs, and AQI is calculated from them. The dataset contains **no meteorological variables** (temperature, wind, humidity), so weather effects were not modelled.

---

## 🚀 Key Features & Capabilities

1. **Continuous AQI Estimation (`XGBoost Regressor`)**
   - Estimates the AQI value for each hour.
   - **Test RMSE:** `2.95 AQI units` | **Test MAE:** `2.11` | **Test R²:** `0.9994`
   - Beats the "copy the previous hour" baseline (RMSE 3.65) by about 19%.

2. **Severe Hazard Detection (`XGBoost Hazard Classifier`)**
   - Recall-optimized classifier (`scale_pos_weight = 6.58`, threshold = `0.40`).
   - **Severe recall:** `99.84%` (630 of 631 severe hours in the test data) | **Precision:** `97.98%` | **F1:** `0.9890`

3. **6-Class AQI Category Classification (`XGBoost Multiclass Head`)**
   - CPCB categories: *Good, Satisfactory, Moderate, Poor, Very Poor, Severe*.
   - **Accuracy:** `98%` | **Weighted F1:** `0.9819` | **Macro F1:** `0.9146`
   - The *Good* class has only 9 test samples, so its metrics are unreliable and pull the macro F1 down.

4. **SHAP Feature Attribution (`shap.TreeExplainer`)**
   - Shows which features drive each prediction.
   - Share of total |SHAP|: AQI history `88.8%` · pollutant readings `10.8%` · time/calendar features `0.4%`.

5. **Interactive Delhi Map & Health Advisory Portal**
   - `outputs/aqi_demo_dashboard.html`: select Delhi/NCR stations, adjust sensor values, view colour-coded AQI and health advisories.
   - **Note:** the dashboard uses a simplified estimator for illustration. It is not connected to the trained models in `/models`.

---

## 📊 Evaluation Summary

| Model | Task | Test Metric | Performance |
| ----- | ---- | ----------- | ----------- |
| Persistence baseline (copy previous hour) | AQI value | RMSE / R² | 3.65 / 0.9990 |
| **XGBoost Regressor** | AQI value | RMSE / MAE / R² | **2.95 / 2.11 / 0.9994** |
| RandomForest (benchmark) | AQI value | RMSE / R² | 5.15 / 0.9981 |
| **XGBoost Hazard Classifier** | Severe hour (binary) | Recall / Precision / F1 | **99.84% / 97.98% / 0.9890** |
| **XGBoost Category Head** | 6 CPCB categories | Weighted F1 / Macro F1 | **0.9819 / 0.9146** |

**Rolling-origin backtest (expanding window, 3 folds):**

| Fold | Test period | RMSE | R² |
| ---- | ----------- | ---- | -- |
| 1 | 2018-01 → 2018-11 | 5.00 | 0.9975 |
| 2 | 2018-11 → 2019-09 | 3.38 | 0.9991 |
| 3 | 2019-09 → 2020-06 | 3.67 | 0.9990 |

**Top features (XGBoost importance):** `AQI_lag1` (57%), `AQI_lag3` (22%), `AQI_roll6h_mean` (18%).

**Why R² is so high:** AQI changes slowly from hour to hour, so even the simple baseline reaches R² = 0.999. RMSE against the baseline is the more informative comparison.

---

## 🔒 Data Leakage Prevention

- **Time-ordered split** (70% / 15% / 15%), no shuffling: train 2015-01 → 2018-11, validation 2018-11 → 2019-09, test 2019-09 → 2020-07.
- **Rolling-window features use `shift(1)`**, so they only see past hours, never the current one.
- **STL components are not model inputs.** Trend + seasonal + residual sums exactly to the target; STL is used for exploratory plots only (`outputs/eda/08_stl_decomposition.png`).
- Lag features are shifted into the past. Target columns (`AQI`, `AQI_ord`, `hazard`, `AQI_Bucket`) are excluded from the feature list.

---

## ⚠️ Limitations & Future Work

- The task is nowcasting: same-hour pollutant readings are inputs, and AQI is computed from them.
- The model depends heavily on recent AQI (`AQI_lag1`, `AQI_lag3`, `AQI_roll6h_mean`).
- No meteorological data, so weather effects were not tested.
- The *Good* category is very rare (9 test samples), so its metrics are unstable.
- The test period includes the 2020 lockdown (Severe rate 8.7% vs 13.2% in training), so the test set differs from the training distribution.
- Missing-value imputation (MICE) and outlier clipping were fitted on the full dataset before splitting.
- The dashboard is a demo estimator, not connected to the trained models.
- **Future work:** add weather data, forecast 24–72 hours ahead, serve the models through an API, extend to other cities.

---

## 📂 Repository Structure

```
AQI-prediction-model/
├── data/
│   ├── raw/               ← Drop raw CPCB city_hour.csv here
│   └── processed/         ← clean_aqi.csv, train/val/test splits
├── models/
│   ├── xgb_reg.pkl        ← Trained XGBoost Regressor
│   ├── xgb_clf_hazard.pkl ← Hazard classifier (model, threshold)
│   └── xgb_clf_cat.pkl    ← 6-class category classifier
├── notebooks/
│   └── aqi_analytics_dashboard.ipynb
├── outputs/
│   ├── aqi_demo_dashboard.html   ← Interactive Delhi map & health portal (demo)
│   ├── eda/                      ← EDA & STL plots
│   └── model/                    ← Confusion matrix, SHAP & backtest plots
├── scripts/
│   ├── phase1_clean.py    ← MICE imputation & IQR outlier clipping
│   ├── phase2_eda.py      ← Exploratory data analysis
│   ├── phase3_features.py ← Lag, rolling, cyclical features; time-ordered split
│   ├── phase4_model.py    ← XGBoost training & benchmarking
│   ├── phase5_shap_eval.py← SHAP attribution & rolling-origin backtest
│   └── audit_model.py     ← Terminal accuracy audit
└── requirements.txt
```

**Features used by the models: 54** = 12 raw pollutants + 7 lag features + 24 rolling statistics + 8 cyclical (sin/cos) + 3 calendar flags.

---

## 🛠️ Quick Start & Setup

### 1. Installation

```
git clone https://github.com/akaksh001/AQI-prediction-model.git
cd AQI-prediction-model
pip install -r requirements.txt
```

### 2. Verify Trained Models in Terminal

```
python scripts/audit_model.py
```

> `audit_model.py` currently contains `os.chdir("D:/projectexhibit-aiml")`. Edit that path to your local project folder, or remove the line and run the script from the repository root.

### 3. Launch the Demo Portal

Open `outputs/aqi_demo_dashboard.html` in any web browser.

### 4. Reproduce the Full Pipeline (Phases 1–5)

```
python scripts/phase1_clean.py
python scripts/phase2_eda.py
python scripts/phase3_features.py
python scripts/phase4_model.py
python scripts/phase5_shap_eval.py
```