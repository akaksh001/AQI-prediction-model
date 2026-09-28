# 🌫️ AIR QUALITY ANALYTICS & PREDICTION SYSTEM
### DSN2098 · Group 65 · Final Project Repository

**Team Members:**
- **Aryavardhan** — Phase 1: Data Collection & MICE Imputation
- **Yugank Shakya** — Phase 2: Exploratory Data Analysis (EDA)
- **Akaksh Samdani** — Phase 3: Feature Engineering & STL Decomposition
- **Anshu Sharma** — Phase 4: XGBoost Modeling & Benchmarking
- **Amrit Raj** — Phase 5: SHAP Interpretability & Backtesting
- **Priyam Kumar** — Phase 6: Interactive Dashboard & Documentation

---

## 📌 Project Overview
Standard AQI monitoring reports a number but not a reason. Hazardous pollution spikes emerge from a mix of meteorological catalysts (temperature inversion, humidity, wind) and anthropogenic drivers (traffic, industry, construction dust) that often go unexplained.

This project delivers an **end-to-end Explainable AI Air Quality Analytics System** built on 5.5 years of hourly CPCB data (2015–2020) for Delhi NCR.

---

## 🚀 Key Features & Capabilities

1. **Continuous AQI Forecasting (`XGBoost Regressor`)**
   - Predicts exact future AQI numerical values.
   - **Test RMSE:** `2.28 AQI units` | **Test R²:** `0.9996` (99.96% variance explained).

2. **Severe Hazard Spike Early Warning (`XGBoost Hazard Classifier`)**
   - Recall-optimized classifier (`scale_pos_weight = 6.58`, threshold = `0.40`).
   - **Severe Spike Recall:** `99.68%` (Catches 629 of 631 severe pollution events in test data).

3. **6-Class AQI Category Categorization (`XGBoost Multiclass Head`)**
   - Classifies air quality into CPCB categories (*Good, Satisfactory, Moderate, Poor, Very Poor, Severe*).
   - **Weighted F1-Score:** `0.9851` | **Accuracy:** `98.51%`.

4. **SHAP Feature Driver Attribution (`shap.TreeExplainer`)**
   - Quantifies individual feature contributions to explain *why* an AQI spike occurred.
   - Separates **Meteorological Catalysts** from **Anthropogenic Drivers**.

5. **Interactive Delhi Regional Map & Health Advisory Portal**
   - Interactive web interface (`outputs/aqi_demo_dashboard.html`) to select Delhi regions (*Anand Vihar, R.K. Puram, ITO, Dwarka, etc.*), confirm sensor parameters, and display **Public Health Hazard Advisories**.

---

## 📂 Repository Structure
```
AQI-prediction-model/
├── data/
│   ├── raw/               ← Drop raw CPCB city_hour.csv here
│   └── processed/         ← Clean dataset, validation splits (clean_aqi.csv, test.csv)
├── models/
│   ├── xgb_reg.pkl        ← Trained XGBoost Regressor model
│   ├── xgb_clf_hazard.pkl ← Trained Hazard Spike Classifier model
│   └── xgb_clf_cat.pkl    ← Trained 6-Class Category Classifier model
├── notebooks/
│   └── aqi_analytics_dashboard.ipynb  ← Interactive Jupyter notebook dashboard
├── outputs/
│   ├── aqi_demo_dashboard.html        ← Interactive Delhi Map & Health Portal
│   ├── eda/                           ← Phase 2 & 3 EDA & STL plots
│   └── model/                         ← Model confusion matrix, SHAP & backtest plots
├── scripts/
│   ├── phase1_clean.py    ← Phase 1: MICE Imputation & IQR Outlier Clipping
│   ├── phase2_eda.py      ← Phase 2: Exploratory Data Analysis & Plots
│   ├── phase3_features.py ← Phase 3: STL Decomposition & Temporal Lag Engineering
│   ├── phase4_model.py    ← Phase 4: XGBoost Model Training & Benchmarking
│   ├── phase5_shap_eval.py← Phase 5: SHAP Attribution & Rolling-Origin Backtest
│   └── audit_model.py     ← Instant Terminal Verification & Accuracy Audit
└── requirements.txt
```

---

## 🛠️ Quick Start & Setup

### 1. Installation
Clone the repository and install required Python packages:
```bash
git clone https://github.com/akaksh001/AQI-prediction-model.git
cd AQI-prediction-model
pip install -r requirements.txt
```

### 2. Verify Trained Models & Accuracy in Terminal
Run the audit script to evaluate accuracy on held-out test data:
```bash
python scripts/audit_model.py
```

### 3. Launch Interactive Demo Portal
Open `outputs/aqi_demo_dashboard.html` in any web browser to test regional map selection, confirmed predictions, and health advisories.

### 4. Run Complete Pipeline (Phases 1–5)
To reproduce the full pipeline from raw data:
```bash
python scripts/phase1_clean.py
python scripts/phase2_eda.py
python scripts/phase3_features.py
python scripts/phase4_model.py
python scripts/phase5_shap_eval.py
```

---

## 📊 Evaluation Summary

| Model | Task | Test Metric | Performance |
|---|---|---|---|
| **XGBoost Regressor** | Continuous AQI Value | Test RMSE / R² | **2.28 RMSE / R² = 0.9996** |
| **XGBoost Hazard Classifier** | Binary Severe Spike | Severe Recall / F1 | **99.68% Recall / 0.9882 F1** |
| **XGBoost Category Head** | 6 CPCB Categories | Categorical F1 | **0.9851 Weighted F1** |
