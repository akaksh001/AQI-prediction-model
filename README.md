# 🌫️ AIR QUALITY ANALYTICS & PREDICTION SYSTEM

### DSN2098 · Group 65 · Final Project Repository

---

## 👥 Team Members

| Team Member | Contribution |
|---|---|
| **Aryavardhan** | Phase 1: Data Collection & MICE Imputation |
| **Yugank Shakya** | Phase 2: Exploratory Data Analysis (EDA) |
| **Akaksh Samdani** | Phase 3: Feature Engineering, Forecast Features & STL-based EDA |
| **Anshu Sharma** | Phase 4: XGBoost Modeling & Benchmarking |
| **Amrit Raj** | Phase 5: SHAP Interpretability & Backtesting |

---

# 📌 Project Overview

This project is an **end-to-end Explainable AI Air Quality Analytics System** developed using approximately **5.5 years of hourly CPCB air-quality data for Delhi**, covering January 2015 to June/July 2020 with **48,192 hourly records**.

The system supports two complementary tasks:

### 1. AQI Nowcasting
Estimates the AQI of the current hour using pollutant measurements and recent AQI history.

### 2. Short-Term AQI Forecasting
Predicts future AQI at:

- **1 hour ahead**
- **6 hours ahead**
- **24 hours ahead**

The forecasting module additionally incorporates meteorological variables such as temperature, humidity, pressure, precipitation and wind.

The project also provides:

- Severe pollution hazard detection
- Six-class CPCB AQI category classification
- SHAP-based model interpretability
- Rolling-origin backtesting
- Interactive Delhi/NCR map and health advisory dashboard

> **Scope note:** The interactive HTML dashboard currently demonstrates the AQI nowcasting workflow using a simplified estimator. The trained forecasting models are evaluated separately through the Python forecasting pipeline.

---

# 🚀 Key Features & Capabilities

## 1. 📊 Continuous AQI Estimation — Nowcasting

An **XGBoost Regressor** estimates the AQI for the current hour.

### Performance

- **Test RMSE:** `2.95 AQI units`
- **Test MAE:** `2.11`
- **Test R²:** `0.9994`

The model improves upon the persistence baseline:

> "Assume the current AQI remains the same as the previous hour."

| Model | RMSE |
|---|---:|
| Persistence Baseline | 3.65 |
| **XGBoost** | **2.95** |

This represents approximately **19% improvement** over the persistence baseline.

---

# 2. 🚨 Severe Pollution Hazard Detection

An **XGBoost binary classifier** identifies whether an observation belongs to the **Severe** AQI category.

The classifier uses a recall-oriented configuration:

- `scale_pos_weight = 6.58`
- Classification threshold = `0.40`

### Performance

- **Recall:** `99.84%`
- **Precision:** `97.98%`
- **F1:** `0.9890`

The model detected **630 of 631 severe hours** in the test data.

---

# 3. 🏷️ Six-Class CPCB AQI Classification

An XGBoost multiclass model classifies AQI into six CPCB categories:

1. Good
2. Satisfactory
3. Moderate
4. Poor
5. Very Poor
6. Severe

### Performance

- **Accuracy:** `98%`
- **Weighted F1:** `0.9819`
- **Macro F1:** `0.9146`

The **Good** class contains only 9 test samples, making its class-specific metrics less reliable and lowering the macro F1 score.

---

# 4. 🔮 Short-Term AQI Forecasting

A separate forecasting pipeline predicts AQI at three future horizons:

- **1 hour ahead**
- **6 hours ahead**
- **24 hours ahead**

The forecasting model uses **72 weather-enabled features**, consisting of:

- AQI history and lag features
- AQI rolling statistics
- Pollutant measurements
- Weather variables
- Cyclical time features
- Calendar indicators

The forecasting model is compared against a **persistence baseline**, where future AQI is assumed to remain equal to the current AQI.

---

## 📈 Forecasting Performance

| Horizon | Persistence RMSE | XGBoost + Weather RMSE | Improvement | R² |
|---|---:|---:|---:|---:|
| **1 hour** | 3.65 | **2.84** | **22.32%** | **0.9994** |
| **6 hours** | 18.43 | **15.34** | **16.78%** | **0.9831** |
| **24 hours** | 49.83 | **48.73** | **2.21%** | **0.8294** |

### Interpretation

The forecasting results show that:

- The **1-hour forecast** provides the strongest improvement over persistence.
- The **6-hour forecast** also provides a substantial improvement.
- The **24-hour forecast** is considerably more difficult and provides only a modest improvement over the persistence baseline.

This is expected because AQI becomes harder to predict as the forecast horizon increases due to changing meteorological conditions, pollution emissions and atmospheric dynamics.

> **Key observation:** Persistence is already a strong baseline for short-term AQI prediction because AQI is highly correlated with its recent values.

---

# 5. 🌦️ Weather Ablation Study

To evaluate the contribution of meteorological information, each forecasting horizon was trained with and without weather features.

| Horizon | XGBoost Without Weather | XGBoost + Weather | Weather Improvement |
|---|---:|---:|---:|
| **1 hour** | 2.99 | **2.84** | **5.05%** |
| **6 hours** | 15.94 | **15.34** | **3.77%** |
| **24 hours** | 49.51 | **48.73** | **1.58%** |

Weather features improved forecasting performance at **all three horizons**.

The benefit is strongest at the 1-hour horizon and becomes smaller for the 24-hour forecast.

---

# 6. 🚨 Forecast Hazard Detection

The forecasting pipeline also predicts whether the future AQI will reach the **Severe** category.

### Weather-enabled XGBoost classifier

| Horizon | Recall | Precision | F1 |
|---|---:|---:|---:|
| **1 hour** | **99.84%** | 98.90% | **0.9937** |
| **6 hours** | **95.72%** | 93.21% | **0.9445** |
| **24 hours** | **78.29%** | 67.49% | **0.7249** |

The results show that severe-event detection also becomes more difficult as the forecast horizon increases.

---

# 7. 🔍 SHAP Explainability

SHAP (`TreeExplainer`) is used to understand which feature groups contribute most strongly to the forecasting model.

For the **24-hour weather-enabled forecast model**:

| Feature Group | Mean |SHAP| Share |
|---|---:|
| **AQI History** | **41.5%** |
| **Pollutants** | **24.5%** |
| **Weather** | **21.7%** |
| **Time** | **12.4%** |

### Interpretation

The model relies most heavily on recent AQI history, followed by pollutant measurements and meteorological information.

This demonstrates that future AQI depends on a combination of:

- Existing pollution levels
- Recent pollution trends
- Pollutant concentrations
- Weather conditions
- Temporal patterns

---

# 8. 🔄 Rolling-Origin Forecast Backtesting

A three-fold expanding-window backtest was performed for the **24-hour forecasting task**.

| Fold | Test Period | Persistence RMSE | No Weather RMSE | Weather RMSE |
|---|---|---:|---:|---:|
| **1** | 2018-01-08 → 2018-11-04 | 52.34 | 58.59 | 53.53 |
| **2** | 2018-11-04 → 2019-09-03 | 61.01 | 53.92 | **51.59** |
| **3** | 2019-09-03 → 2020-06-29 | 49.84 | 52.82 | **46.13** |

The rolling-origin evaluation demonstrates that forecasting performance varies across historical periods.

The weather-enabled model outperforms persistence in **Fold 2 and Fold 3**, while performance in Fold 1 is slightly worse than the persistence baseline.

This provides a more realistic assessment of model robustness across time.

---

# 📊 Overall Evaluation Summary

## Nowcasting

| Model | Task | Test Metric | Performance |
|---|---|---|---|
| Persistence Baseline | AQI value | RMSE / R² | 3.65 / 0.9990 |
| **XGBoost Regressor** | AQI value | RMSE / MAE / R² | **2.95 / 2.11 / 0.9994** |
| RandomForest | AQI value | RMSE / R² | 5.15 / 0.9981 |
| **XGBoost Hazard Classifier** | Severe hour | Recall / Precision / F1 | **99.84% / 97.98% / 0.9890** |
| **XGBoost Category Head** | 6 CPCB categories | Weighted F1 / Macro F1 | **0.9819 / 0.9146** |

## Forecasting

| Horizon | Persistence RMSE | XGBoost Weather RMSE | Improvement |
|---|---:|---:|---:|
| **1h** | 3.65 | **2.84** | **22.32%** |
| **6h** | 18.43 | **15.34** | **16.78%** |
| **24h** | 49.83 | **48.73** | **2.21%** |

---

# 🧠 Why Is R² So High?

The AQI time series is highly persistent, meaning that AQI usually does not change drastically between consecutive hours.

Consequently, even the simple persistence baseline achieves:

```text
R² = 0.9990