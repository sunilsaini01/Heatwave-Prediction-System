<div align="center">

# Heatwave Prediction System

### District-Level, 10-Day Maximum-Temperature Forecasting for Uttar Pradesh, India

**A hybrid LSTM + XGBoost residual-correction pipeline for climate-risk early warning.**
Built as part of the **Climate Resilience Observatory (CRO)** initiative.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow%20%2F%20Keras-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-GPU%20hist-EC6B23?style=flat)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Kaggle](https://img.shields.io/badge/Trained%20on-Kaggle%202%C3%97T4-20BEFF?style=flat&logo=kaggle&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

</div>

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Key Results](#3-key-results)
4. [Data](#4-data)
5. [Exploratory & Statistical Analysis](#5-exploratory--statistical-analysis)
6. [Feature Engineering & Selection](#6-feature-engineering--selection)
7. [Validation Strategy](#7-validation-strategy)
8. [Modelling Approach](#8-modelling-approach)
9. [Evaluation](#9-evaluation)
10. [Repository Structure](#10-repository-structure)
11. [Reproducing the Results](#11-reproducing-the-results)
12. [Limitations & Known Caveats](#12-limitations--known-caveats)
13. [Roadmap](#13-roadmap)
14. [Tech Stack](#14-tech-stack)
15. [Author](#15-author)
16. [License](#16-license)

---

## 1. Executive Summary

Extreme heat events across India are becoming more frequent, starting earlier, and getting more severe. This project builds a scalable, **district-level forecasting system** that predicts daily maximum temperature (`T2M_MAX`) **up to 10 days ahead** for **125 districts** in and around Uttar Pradesh. The system is designed to feed heatwave early-warning and decision-support workflows.

| | |
|---|---|
| **Task** | Direct multi-step regression: 30 days of history → next 10 days of `T2M_MAX` |
| **Coverage** | 125 districts (UP + bordering Rajasthan, Delhi-NCR/Haryana, MP, Bihar, Jharkhand, Chhattisgarh, Uttarakhand, HP, Nepal) |
| **Data** | ~1.92 M district-day records, daily, **1984 – 2025** |
| **Architecture** | Stage 1: LSTM sequence forecaster → Stage 2: per-horizon XGBoost residual correctors |
| **Headline result** | **1.689 °C** mean test MAE across the 10-day horizon (vs 1.717 °C LSTM-only) |
| **Leakage control** | Strict chronological splits, train-only scaler fitting, per-district sequence windows |

**Intended uses:** early warning systems, climate-resilience planning, disaster-response preparation, and heatwave risk analysis.

---

## 2. Problem Statement

Heatwaves are among India's deadliest natural hazards. They hit outdoor workers, the elderly, and economically vulnerable people hardest. Uttar Pradesh is especially exposed because of:

- **Geography**: a large landlocked plain with no orographic cooling.
- **Loo winds**: hot, dry westerlies from the Thar Desert that sweep western UP during the pre-monsoon (April – June).
- **Population density**: more than 200 million residents, many working outdoors.
- **Late monsoon onset**: the monsoon arrives later than in coastal states, which extends heat exposure.

The **March – April 2022** event was one of the most severe and unusually early heatwaves ever recorded in UP, with temperatures above 44 °C weeks before the normal season. It falls inside the held-out test period and serves as the main real-world benchmark.

### Operational heatwave definition (IMD)

The India Meteorological Department declares a heatwave at a plains station when **both** of the following hold:

| Condition | Threshold |
|---|---|
| Absolute temperature | `T_max ≥ 40 °C` |
| Departure from normal | `T_max ≥ 4.5 °C` above the district climatological normal |

The climatological normal is the mean `T_max` per district and day-of-year over the **1984 – 2015** baseline. That is 32 years, longer than the WMO's 30-year standard.

> The current models forecast continuous `T2M_MAX`. Heatwave flags are derived downstream by applying the IMD rule above. Direct probabilistic heatwave classification is on the [roadmap](#13-roadmap).

---

## 3. Key Results

### Test-set MAE by forecast horizon (2019 – 2025, 314,625 sequences)

| Horizon | LSTM baseline (°C) | **Hybrid LSTM + XGBoost (°C)** | Δ (°C) | Relative gain |
|:---:|:---:|:---:|:---:|:---:|
| Day 1  | 1.4135 | **1.3868** | −0.0267 | 1.9 % |
| Day 2  | 1.5417 | **1.5314** | −0.0104 | 0.7 % |
| Day 3  | 1.6240 | **1.6178** | −0.0062 | 0.4 % |
| Day 4  | 1.6860 | **1.6739** | −0.0121 | 0.7 % |
| Day 5  | 1.7357 | **1.7158** | −0.0200 | 1.2 % |
| Day 6  | 1.7714 | **1.7456** | −0.0258 | 1.5 % |
| Day 7  | 1.8106 | **1.7710** | −0.0395 | 2.2 % |
| Day 8  | 1.8403 | **1.7949** | −0.0454 | 2.5 % |
| Day 9  | 1.8576 | **1.8193** | −0.0383 | 2.1 % |
| Day 10 | 1.8893 | **1.8362** | −0.0531 | 2.8 % |
| **Mean** | **1.7170** | **1.6893** | **−0.0277** | **1.6 %** |

**Takeaways**

- The hybrid model **improves on the baseline at every one of the 10 horizons**. No horizon gets worse, so the corrector adds signal rather than noise.
- Gains **grow with lead time**: about 0.5 – 1.9 % on days 1 – 6 and 2.1 – 2.8 % on days 7 – 10. Those later days are where standalone sequence models drift most, which is exactly where residual correction was expected to help.
- Error rises smoothly and monotonically with horizon, from 1.39 °C to 1.84 °C. There is no error blow-up, because the direct (non-recursive) formulation avoids compounding errors.

---

## 4. Data

### 4.1 Sources

| Property | Details |
|---|---|
| Provenance | IMD + NASA POWER (variable naming follows the NASA POWER parameter convention) |
| Hosting | Kaggle: [`tarunraii/uttar-pradesh-heatwave-dataset-imd-1984-2025`](https://www.kaggle.com/datasets/tarunraii/uttar-pradesh-heatwave-dataset-imd-1984-2025) (raw) and [`tarunraii/heatwave-dataset-uttar-pradesh-1984-2025`](https://www.kaggle.com/datasets/tarunraii/heatwave-dataset-uttar-pradesh-1984-2025) (LSTM-ready features) |
| Geography | 125 districts: Uttar Pradesh plus neighbouring Rajasthan, Delhi, Haryana, Madhya Pradesh, Bihar, Jharkhand, Chhattisgarh, Uttarakhand, Himachal Pradesh and Nepal |
| Spatial extent | Lat 23.2° – 30.6° N, Lon 76.7° – 84.9° E |
| Temporal resolution | Daily |
| Period | 1984-01-01 → 2025-12-30 |
| Volume | **1,917,500** raw rows (21 columns) / **1,916,625** model-ready rows (30 columns) |
| Missing values | None |
| Target | Daily maximum 2 m air temperature, `T2M_MAX` (°C) |

**Why include neighbouring regions?** Heatwaves ignore administrative boundaries. Heat builds over Rajasthan and the Chambal basin and moves eastward into UP. Neighbouring districts therefore give the upstream context needed to forecast that propagation, and they set up the planned graph-based models.

### 4.2 Raw meteorological variables

| Variable | Description | Treatment |
|---|---|---|
| `T2M_MAX` | Daily maximum temperature (°C) | **Target** |
| `T2M_MIN`, `T2M` | Daily minimum / mean temperature (°C) | Predictor |
| `RH2M` | Relative humidity at 2 m (%) | Predictor |
| `QV2M` | Specific humidity at 2 m | Predictor |
| `TSOIL1` | Soil temperature, layer 1 (°C) | Predictor |
| `ALLSKY_SFC_SW_DWN` | All-sky surface shortwave radiation | Predictor |
| `CLOUD_AMT` | Cloud amount (%) | Predictor |
| `WS10M` | Wind speed at 10 m (m/s) | Predictor |
| `WD10M` | Wind direction at 10 m (°) | Encoded → `WD10M_SIN`, `WD10M_COS` |
| `PS` | Surface pressure (kPa) | Predictor |
| `PRECTOTCORR` | Bias-corrected precipitation (mm/day) | Predictor |
| `EVLAND` | Evapotranspiration over land (mm/day) | Predictor |
| `TS` | Surface skin temperature (°C) | **Dropped**, VIF = 42.3 |
| `CLRSKY_SFC_SW_DWN` | Clear-sky surface shortwave radiation | **Dropped**, VIF = 63.3 |
| `GWETTOP` | Surface soil wetness | **Dropped**, VIF = 25.9 |
| `GWETROOT` | Root-zone soil wetness | **Dropped**, VIF = 56.6 |
| `city`, `lat`, `lon` | District identifier and coordinates | Metadata / static spatial features |

---

## 5. Exploratory & Statistical Analysis

| Analysis | Finding |
|---|---|
| **Target distribution** | Most `T2M_MAX` values fall between 25 and 35 °C, peaking at 30 – 32 °C. The distribution is right-skewed, with a tail reaching 45 – 50 °C, which is the heatwave regime. |
| **Seasonality** | Monthly mean Tmax rises from about 24 °C in January to a peak of about 42 °C in May, then falls sharply with the monsoon (about 35 °C in July). **April – June is the primary heatwave window**, consistent with Indo-Gangetic Plain climatology. |
| **Pre-monsoon distribution** | In April – June, most days fall between 38 and 45 °C, with a mode around 42 – 43 °C. A large share sits above the 40 °C IMD threshold, so there are plenty of extreme-event samples to learn from. |
| **Season breakdown** | The pre-monsoon distribution extends well past 40 °C and contains most of the outliers. Monsoon days peak at 32 – 33 °C, hot but humid, so the heat-stress profile is different. Winter has the widest downward spread, with some districts below 10 °C. |
| **Inter-annual trend** | The all-district yearly mean Tmax varies between about 31 and 34 °C with no strong monotonic trend. This variability reflects monsoon behaviour and large-scale circulation. |
| **Hottest districts** | Bharatpur (33.90 °C), Dholpur, Bhind, Morena, Agra, Mathura, Jalaun, Firozabad, Etawah and Hamirpur lead. They sit in the **Rajasthan border, Chambal basin, western UP and Bundelkhand**, which are semi-arid regions with low vegetation cover and high insolation. This supports the west-to-east propagation hypothesis. |
| **Correlation structure** | `T2M_MAX_lag1/3/7` and `T2M_MAX_roll7_mean` correlate very strongly with the target (temperature persistence). There are also strong collinear blocks within the soil-dryness, atmospheric-dryness and radiation feature groups. |
| **Mann–Kendall trend test** (pre-monsoon, Mar – Jun, per district) | **107 of 125 districts show a statistically significant *decreasing* trend** in seasonal mean Tmax. The other 18 show no significant trend and **none show an increasing trend**. Sen's slopes range from −0.016 to −0.067 °C/yr. The strongest declines are in the Terai and Nepal border belt (Nawalparasi, Rupandehi, Kapilvastu, West Champaran), and the weakest are in eastern UP and Bihar (Kaimur, Sonbhadra, Varanasi). |

> **Interpretation note:** A falling *seasonal mean* does not mean falling *extremes*. Pre-monsoon means can drop because of rising aerosol loading, irrigation-driven cooling, or more pre-monsoon showers, while upper-tail events such as 2022 still get more intense. Extreme-value analysis (e.g. trends in annual maxima, or exceedance counts above 40 °C) is a planned follow-up.

---

## 6. Feature Engineering & Selection

### 6.1 Engineered features

| Category | Features | Rationale |
|---|---|---|
| Temperature persistence | `T2M_MAX_lag1`, `T2M_MAX_lag3`, `T2M_MAX_lag7` | Autoregressive signal at short, medium and weekly lags |
| Rolling statistics | `T2M_MAX_roll7_mean`, `T2M_MAX_roll7_std` | Recent level and volatility |
| Diurnal / surface energy | `T2M_RANGE_lag1`, `skin_air_delta_lag1` | Diurnal temperature range; skin-air temperature gradient (sensible-heat flux proxy) |
| Radiation & cloud | `RADIATION_EFFECTIVE_lag1`, `CLOUD_AMT_lag1` | Effective incoming solar forcing |
| Dryness | `ATM_DRYNESS_lag1`, `SOIL_DRYNESS_lag1` | Atmospheric vapour deficit; soil-moisture feedback on heating |
| Heat stress | `HEAT_STRESS_lag1` | Combined temperature-humidity load |
| Moisture & hydrology | `QV2M_lag1`, `PRECTOTCORR_lag1`, `GWETROOT_lag1`, `EVLAND_lag1` | Moisture availability and evaporative cooling |
| Dynamics | `WS10M_lag1`, `PS_lag1`, `WD10M_SIN/COS` | Advection and synoptic state; cyclical wind-direction encoding |
| Seasonality | `sin_doy`, `cos_doy` | Cyclical day-of-year encoding (no discontinuity at Dec 31 → Jan 1) |
| Spatial | `lat`, `lon` | Lets one global model learn location-specific behaviour |

All dynamic predictors are **lagged by at least one day**, so no same-day information from the target leaks into the inputs.

### 6.2 Multicollinearity-driven selection (VIF)

Variance Inflation Factor analysis found heavy collinearity among physically related variables. Rather than mechanically dropping every high-VIF column, the selection used **domain knowledge**:

- **Kept** the lagged temperature terms despite their high VIF, because they are essential to autoregressive forecasting.
- **Dropped** redundant lags and near-duplicate signals: `SOIL_DRYNESS_lag2/3`, `ATM_DRYNESS_lag2`, `ALLSKY_lag1` and `CLRSKY_lag1` (already captured by `RADIATION_EFFECTIVE_lag1`).
- **Dropped** `month` and `year`, since sine/cosine day-of-year already encodes seasonality, and `year` would invite extrapolation on the test set.
- On the raw dataset: dropped `CLRSKY_SFC_SW_DWN` (≈ `ALLSKY` on the clear-sky days that dominate UP summers), `GWETTOP` (moves with `GWETROOT`) and `TS` (near-duplicate of `T2M_MAX`).

**Final LSTM input: 22 features per timestep.**

---

## 7. Validation Strategy

Random splits leak future information into training in time-series problems, so **all splits are strictly chronological**.

### Forecasting pipeline (LSTM + Hybrid)

| Split | Period | Rows | Sequences | Purpose |
|---|---|---:|---:|---|
| Train | 1984 – 2010 | 1,231,875 | 1,227,000 | LSTM fitting |
| Validation | 2011 – 2018 | 365,250 | 360,375 | LSTM early stopping / LR scheduling **and** XGBoost residual training |
| Test | 2019 – 2025 | 319,500 | 314,625 | Final held-out evaluation (includes the 2022 heatwave) |

### CRO statistical pipeline

| Split | Period | Years | Rows | Purpose |
|---|---|---:|---:|---|
| Train | 1984 – 2015 | 32 | 1,461,000 | Model training + climatological-normal baseline |
| Validation | 2016 – 2019 | 4 | 182,625 | Hyperparameter tuning |
| Test | 2020 – 2025 | 6 | 273,875 | Final evaluation, including the 2022 UP heatwave |

**Further leakage safeguards**

- `StandardScaler` is **fit on training data only** and then applied to validation and test.
- Sliding-window sequences are built **separately per district**, so no window crosses a district boundary.
- Targets `[t, …, t+9]` lie strictly after inputs `[t−30, …, t−1]`.

---

## 8. Modelling Approach

### 8.1 Architecture

```text
             ┌──────────────────────────────────────────┐
             │  Daily meteorological time series         │
             │  125 districts × 22 features × 30 days    │
             └────────────────────┬─────────────────────┘
                                  │
                                  ▼
             ┌──────────────────────────────────────────┐
  Stage 1    │  LSTM direct multi-step forecaster        │
             │  LSTM(64) → Dropout(0.2) → Dense(32,relu) │
             │           → Dense(10)                     │
             └────────────────────┬─────────────────────┘
                                  │  ŷ_lstm  (10 values)
                                  ▼
             ┌──────────────────────────────────────────┐
  Stage 2    │  10 × XGBoost residual correctors         │
             │  input = last-timestep features (22)      │
             │        + ŷ_lstm[d]               (1)      │
             │  target = y − ŷ_lstm[d]                   │
             └────────────────────┬─────────────────────┘
                                  │  r̂[d]
                                  ▼
             ┌──────────────────────────────────────────┐
             │  ŷ_final[d] = ŷ_lstm[d] + r̂[d]           │
             │  d = 1 … 10                               │
             └──────────────────────────────────────────┘
```

### 8.2 Why a residual hybrid?

| Component | Strengths it contributes |
|---|---|
| **LSTM** | Temporal dependencies, seasonality, persistence, long-range weather patterns |
| **XGBoost** | Systematic (bias-like) forecast errors, nonlinear tabular corrections, horizon-specific residual behaviour |

The LSTM handles sequence modelling but drifts at longer leads. XGBoost cannot read sequences directly, but it is very good at structured tabular correction. Stacking them as **base forecaster + residual learner** makes long-range forecasts more stable and accurate.

### 8.3 Stage 1: LSTM baseline (`LSTM-baseline-model.ipynb`)

| Setting | Value |
|---|---|
| Formulation | **Direct** multi-step, with all 10 days in one forward pass (non-recursive, so errors do not propagate) |
| Input window / horizon | 30 days → 10 days |
| Input tensor | `(samples, 30, 22)` |
| Layers | `LSTM(64)` → `Dropout(0.2)` → `Dense(32, relu)` → `Dense(10, float32)` |
| Parameters | 24,682 (≈ 96 KB) |
| Loss / metric | MSE / MAE |
| Optimiser | Adam (lr = 1e-3) |
| Batch size / max epochs | 128 / 30 |
| Callbacks | `EarlyStopping(val_loss, patience=15, restore_best_weights=True)`, `ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6)` |
| Precision | `mixed_float16` global policy (Tesla T4 tensor cores) |
| Hardware | Kaggle, 2 × NVIDIA Tesla T4 (≈ 70 s / epoch) |

### 8.4 Stage 2: XGBoost residual correction (`Hybrid-Model.ipynb`)

**Design decisions**

- **One model per horizon day (10 total).** Error structure differs by lead time, and a single shared model would blur it.
- **Trained on validation-set residuals, not training residuals.** The LSTM has partly memorised its training sequences, so training residuals are artificially small and unrepresentative. Validation residuals reflect real out-of-sample weaknesses.
- **Last-timestep features only (22) plus the LSTM's own prediction for that day (1), giving 23 inputs.** Flattening all 30 timesteps would create 660 highly redundant features and invite overfitting. Day *t−1* is the most informative snapshot of the current atmospheric state, and the LSTM's forecast tells the corrector *where* the LSTM tends to over- or under-predict.
- **Heavily regularised shallow trees**, so corrections generalise beyond the residuals they were fit on.

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 100 |
| `max_depth` | 3 |
| `learning_rate` | 0.05 |
| `subsample` / `colsample_bytree` | 0.5 / 0.5 |
| `min_child_weight` | 50 |
| `reg_alpha` (L1) / `reg_lambda` (L2) | 1.0 / 5.0 |
| `tree_method` / `device` | `hist` / `cuda` |
| `random_state` | 42 |

Validation residuals before correction: mean **−0.301 °C** (the LSTM over-predicts on average), std **2.198 °C**. Most of the corrector's early-horizon gain comes from removing this systematic bias.

---

## 9. Evaluation

### 9.1 Metric

**Mean Absolute Error (°C)**, computed per horizon day and then averaged across the 10 days. MAE is reported in physical units, so operational users can read it directly, and it is robust to occasional extreme residuals.

### 9.2 LSTM baseline: validation vs test

| Horizon | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Val MAE (°C)  | 1.357 | 1.488 | 1.564 | 1.614 | 1.657 | 1.686 | 1.716 | 1.741 | 1.757 | 1.783 |
| Test MAE (°C) | 1.413 | 1.542 | 1.624 | 1.686 | 1.736 | 1.771 | 1.811 | 1.840 | 1.858 | 1.889 |

The val-to-test gap is small, about 0.06 – 0.11 °C, so generalisation to the more recent (and hotter-extreme) 2019 – 2025 period holds up.

### 9.3 Hybrid: in-sample validation gain

| Horizon | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| LSTM val MAE   | 1.357 | 1.488 | 1.564 | 1.614 | 1.657 | 1.686 | 1.716 | 1.741 | 1.757 | 1.783 |
| Hybrid val MAE | 1.295 | 1.435 | 1.508 | 1.551 | 1.580 | 1.598 | 1.617 | 1.633 | 1.649 | 1.660 |

> These validation figures are **in-sample** for XGBoost, since the correctors were fit on these same residuals. The held-out **test** results in [§3](#3-key-results) are the authoritative performance numbers. The drop from about 0.06 – 0.12 °C of gain in-sample to about 0.01 – 0.05 °C out-of-sample is the expected shrinkage and confirms that the regularisation is doing its job.

### 9.4 Artefacts produced

| Artefact | Produced by | Description |
|---|---|---|
| `lstm_baseline_final.keras` | Notebook 1 | Trained LSTM |
| `y_lstm_val.npy`, `y_lstm_test.npy` | Notebook 1 | LSTM predictions, shapes `(360375, 10)` / `(314625, 10)` |
| `y_val_seq.npy`, `y_test_seq.npy` | Notebook 1 | Ground-truth target windows |
| `X_val_last.npy`, `X_test_last.npy` | Notebook 1 | Last-timestep features, shapes `(·, 22)` |
| `xgb_day1.pkl … xgb_day10.pkl` | Notebook 2 | Per-horizon XGBoost correctors (≈ 118 KB each) |
| `y_hybrid_test.npy` | Notebook 2 | Final hybrid test predictions, `(314625, 10)` |
| `y_xgb_residual_test.npy` | Notebook 2 | Raw XGBoost corrections on test |
| `mae_results.pkl` | Notebook 2 | Per-day and overall MAE for both models |
| `mae_comparison.png` | Notebook 2 | LSTM vs Hybrid MAE-by-horizon plot |

---

## 10. Repository Structure

```text
Heatwave-Prediction-System/
├── CRO-Heatwave-Prediction.ipynb   # Problem framing, IMD definition, raw-data audit, cyclical
│                                   # encoding, VIF, seasonal EDA, Mann–Kendall trend tests
├── LSTM-baseline-model.ipynb       # Notebook 1: EDA, feature selection, chronological split,
│                                   # scaling, sequence building, LSTM training & evaluation,
│                                   # export of predictions for Stage 2
├── Hybrid-Model.ipynb              # Notebook 2: residual computation, 10 × XGBoost correctors,
│                                   # hybrid inference, LSTM-vs-Hybrid evaluation, model export
├── LICENSE                         # MIT
└── README.md
```

**Execution order:** `CRO-Heatwave-Prediction.ipynb` (analysis) → `LSTM-baseline-model.ipynb` → `Hybrid-Model.ipynb`

---

## 11. Reproducing the Results

The notebooks were developed and run on **Kaggle** (Python 3.12, 2 × Tesla T4 GPUs), and they read from `/kaggle/input/...` and write to `/kaggle/working/`.

### Option A: Kaggle (recommended)

1. Create a Kaggle notebook and attach both datasets listed in [§4.1](#41-sources).
2. Turn on the **GPU T4 × 2** accelerator.
3. Run `LSTM-baseline-model.ipynb` end to end. It saves the LSTM and the `.npy` hand-off files.
4. In a second notebook, attach the output of step 3 as an input, then run `Hybrid-Model.ipynb`.

### Option B: Local

```bash
git clone https://github.com/sunilsaini01/Heatwave-Prediction-System.git
cd Heatwave-Prediction-System

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install pandas numpy matplotlib seaborn scikit-learn \
            tensorflow xgboost statsmodels pymannkendall joblib jupyter
```

Download the datasets from Kaggle, then update the `pd.read_csv(...)` and `BASE = ...` paths, plus any `/kaggle/working/` output paths, to point at local directories. A CUDA-capable GPU is strongly recommended: the LSTM trains on about 1.2 M sequences per epoch. For CPU-only XGBoost, set `device='cpu'`.

---

## 12. Limitations & Known Caveats

Transparency about what the current results do **not** show:

1. **LSTM converges very early.** Validation loss is lowest after **epoch 1** (val MAE 1.637). Training loss keeps falling while validation loss rises, so `restore_best_weights` effectively returns the epoch-1 model. The causes are likely high temporal persistence plus a train/validation distribution shift (1984 – 2010 vs 2011 – 2018). Next steps: lower learning rate, more dropout or recurrent dropout, and weight decay.
2. **The validation set has two jobs.** It drives LSTM early stopping *and* serves as XGBoost's training set. Test results stay clean, but a dedicated calibration split would remove this coupling.
3. **Point forecasts only.** There are no prediction intervals yet, which an operational warning service needs.
4. **Regression, not classification.** Heatwave flags come from thresholding `T2M_MAX`. Skill on the *event* (POD, FAR, CSI on IMD heatwave days) has not been reported yet, and a 1.7 °C MAE averaged over all days can hide weaker skill in the extreme tail.
5. **No explicit spatial coupling.** Districts are modelled independently, with `lat`/`lon` as their only spatial signal. Upstream heat advection (Rajasthan → UP) is not modelled explicitly.
6. **Baseline comparisons.** Benchmarks against persistence, climatology, and NWP (e.g. IMD GFS/ECMWF) would put the absolute MAE in context.
7. **Two split schemes.** The CRO analysis notebook and the forecasting notebooks use different chronological boundaries, as documented in [§7](#7-validation-strategy).

---

## 13. Roadmap

- [ ] **Heatwave probability classification**: district-level heatwave probability for each of the next 10 – 15 days
- [ ] **Severity tiers**: Moderate / Severe / Extreme classification
- [ ] **Uncertainty quantification**: confidence intervals via quantile regression, MC dropout, or conformal prediction
- [ ] **Graph Neural Networks**: model heat propagation across neighbouring districts as a spatio-temporal graph
- [ ] **Attention-based interpretability**: attention maps showing which districts and variables drive each forecast
- [ ] **Extend the horizon** to 15 days
- [ ] **Event-based verification**: POD / FAR / CSI on IMD-defined heatwave days, with a case study of the 2022 event
- [ ] **Operational deployment**: scheduled ingestion → inference → alerting pipeline
- [ ] **Real-time dashboard**: district risk map for government health, disaster-response and labour departments

---

## 14. Tech Stack

**Machine Learning & Deep Learning**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EC6B23?style=flat)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

**Data, Statistics & Research**

![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-4051B5?style=flat)
![pymannkendall](https://img.shields.io/badge/pymannkendall-6A5ACD?style=flat)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)

### Research & Engineering Focus

Climate AI · Time-Series Forecasting · Deep Learning · Environmental Intelligence · Spatiotemporal Modelling · AI for Public Infrastructure

---

## 15. Author

**Sunil Kumar**

Focus areas: Climate Intelligence · Forecasting Systems · Applied Deep Learning · Data Science & Statistical Modelling

I build research-oriented AI systems that pair real-world impact with scalable engineering.

- GitHub: [@sunilsaini01](https://github.com/sunilsaini01)
- Kaggle: [kaggle.com](https://www.kaggle.com/)

### Project Vision

This work is part of ongoing exploration into **AI-driven climate-resilience systems** for large-scale public forecasting and decision support. The long-term aim is an operational, interpretable, district-level heat early-warning service for Uttar Pradesh and the wider Indo-Gangetic Plain.

### Citation

```bibtex
@misc{kumar2026heatwave,
  author       = {Sunil, Kumar},
  title        = {Heatwave Prediction System: Hybrid LSTM--XGBoost District-Level
                  Maximum-Temperature Forecasting for Uttar Pradesh, India},
  year         = {2026},
  howpublished = {\url{https://github.com/sunilsaini01/Heatwave-Prediction-System}},
  note         = {Climate Resilience Observatory (CRO)}
}
```

---

## 16. License

Released under the [MIT License](LICENSE). © 2026 Sunil Kumar.
