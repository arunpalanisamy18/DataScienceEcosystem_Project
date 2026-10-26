# Predictive Maintenance Intelligence (IA3 Data Science Project)

An end-to-end predictive-maintenance system: it **learns to predict machine failure from sensor readings**, then drives a **multi-machine factory simulation and Streamlit dashboard** that raises risk alerts and explains them.

## 1. Problem statement

Unplanned machine failure causes costly downtime, while over-servicing wastes money. Given six readings from a machine (type, air temperature, process temperature, rotational speed, torque, tool wear), the goal is to **estimate the probability that the machine is about to fail** so maintenance can be scheduled in time.

Because only ~3.4% of records are failures, the problem is *highly imbalanced*: a model that always says "no failure" is 96.6% accurate and completely useless. Success is therefore measured with **precision, recall, F1 and PR-AUC** on the failure class, not accuracy.

## 2. Dataset

| Item | Detail |
|---|---|
| Name | AI4I 2020 Predictive Maintenance Dataset (UCI ML Repository, id 601) |
| Size | 10,000 records, 12 columns, no missing values, no duplicates |
| Inputs used | `Type`, `Air temperature`, `Process temperature`, `Rotational speed`, `Torque`, `Tool wear` |
| Target | `Machine failure` (1 = failure): 339 failures (3.39%) |
| Excluded | `TWF, HDF, PWF, OSF, RNF` are the *causes* of the failure label. Using them would be **target leakage**, so they are never model inputs (a unit test enforces this). |

Link: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
The dataset is **synthetic** (generated to mimic real machines), which limits how far results can be generalised.

## 3. Method

1. **EDA** (`eda.py`): class imbalance, sensor distributions by outcome, correlations, failure modes, failure rate by machine type.
2. **Physics-based feature engineering** (`features.py`). The dataset documentation describes how failures arise, so three combined features were added to the raw sensors:
   `Temp difference = process - air`, `Power = torque x angular speed`, `Overstrain = tool wear x torque`.
3. **Leak-free protocol** (`rigorous_evaluation.py`). Stratified **60% train / 20% validation / 20% test**:
   - model comparison and hyper-parameter tuning use **5-fold cross-validation on the training split only**;
   - the **decision threshold is chosen on the validation split**;
   - the **test split is used exactly once** for the final numbers.
4. **Models compared**: dummy baseline, Logistic Regression, Random Forest, Gradient Boosting, each with and without the engineered features.
5. **Tuning**: randomised search (16 candidates x 5 folds) on PR-AUC.
6. **Uncertainty and diagnostics**: 1,000-sample bootstrap 95% confidence intervals, calibration curve, permutation importance, recall by failure mode.

## 4. Results

### Cross-validated model comparison (training split, mean +/- std)

| Model | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|
| Dummy baseline | 0.034 | 0.000 | 0.000 | 0.000 |
| Logistic Regression (raw) | 0.451 +/- 0.089 | 0.803 | 0.141 | 0.240 |
| Random Forest (raw) | 0.726 +/- 0.071 | 0.503 | 0.761 | 0.600 |
| Gradient Boosting (raw) | 0.801 +/- 0.052 | 0.592 | 0.875 | 0.704 |
| Logistic Regression (+features) | 0.490 +/- 0.063 | 0.838 | 0.167 | 0.278 |
| Random Forest (+features) | 0.898 +/- 0.042 | 0.794 | 0.943 | 0.860 |
| **Gradient Boosting (+features)** | **0.904 +/- 0.070** | **0.833** | **0.971** | **0.896** |

Engineered features lift Gradient Boosting PR-AUC from 0.80 to 0.90 and F1 from 0.70 to 0.90.

### Final held-out test set (2,000 records, 68 failures)

| Model / mode | Threshold | Precision | Recall | F1 | PR-AUC | ROC-AUC | TN / FP / FN / TP |
|---|---|---|---|---|---|---|---|
| **Final model: Balanced** | 0.32 | **96.6%** | **83.8%** | **89.8%** | **0.901** | 0.970 | 1930 / 2 / 11 / 57 |
| Final model: High-recall | 0.05 | 56.7% | 86.8% | 68.6% | 0.901 | 0.970 | 1887 / 45 / 9 / 59 |
| Baseline (untuned GB, raw features) | 0.35 | 76.9% | 73.5% | 75.2% | 0.782 | 0.969 | 1917 / 15 / 18 / 50 |

Bootstrap 95% confidence intervals (balanced mode): precision 0.911-1.000, recall 0.741-0.921, F1 0.835-0.947, PR-AUC 0.830-0.956. With only 68 test failures the intervals are wide, so the figures should be read as "about 0.90 F1", not as exact values.

### Where the model succeeds and fails (recall by failure mode, balanced threshold)

| Mode | Detected / total | Recall |
|---|---|---|
| PWF (power) | 13 / 13 | 100% |
| OSF (overstrain) | 16 / 16 | 100% |
| HDF (heat dissipation) | 28 / 29 | 97% |
| TWF (tool wear) | 2 / 10 | 20% |
| RNF (random) | 0 / 4 | 0% |

The model catches almost every failure that has a physical signature. **RNF failures are random by construction and TWF failures occur at a random wear level between 200 and 240 min**, so they are inherently hard to predict from sensors. This is a property of the data, not only of the model.

### Threshold trade-off
The decision threshold is a business choice. **Balanced** maximises F1 on validation. **High-recall** is the highest threshold that still catches at least 85% of validation failures. It misses fewer failures (9 vs 11) but raises 45 false alarms instead of 2. The dashboard lets the user switch between them. Under an assumed 10:1 cost of a missed failure vs a false alarm, the balanced mode actually has the lowest test cost (112 vs 135); a pure cost-minimising threshold on validation was unstable (0.03) and was rejected.

## 5. Project structure

```
PredictiveMaintenanceIA3/
├── app.py                    Streamlit dashboard (4 pages)
├── dashboard_logic.py        Table / history helpers used by the dashboard
├── features.py               Physics-based feature engineering (shared)
├── inference.py              Cached model loading + prediction (single & batch)
├── explain.py                Failure-mode warnings ("why is this machine flagged?")
├── eda.py                    Exploratory data analysis and charts
├── rigorous_evaluation.py    CV comparison, tuning, thresholds, final test, charts, model export
├── validate_dataset.py       (optional) re-download and validate the UCI dataset
├── ai4i2020_raw.csv          Local copy of the dataset
├── requirements.txt          Dependencies
├── models/                   gradient_boosting_pipeline.joblib, model_config.json
├── reports/                  All charts (PNG) and metrics (CSV)
├── simulation/               machine_engine.py, factory_controller.py, factory_monitor.py, scenarios.py
├── tests/                    35 automated tests (pytest)
├── docs/                     Project report (Word + PDF) and viva preparation notes
└── .streamlit/config.toml    Dashboard theme
```

## 6. How to run (Windows PowerShell)

```powershell
cd "C:\Users\hp\Documents\PredictiveMaintenanceIA3"
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 1. (Recommended once) regenerate analysis, charts and the saved model on YOUR machine
python eda.py
python rigorous_evaluation.py      # about 1-2 minutes

# 2. Run the automated tests
python -m pytest -q

# 3. Launch the dashboard
streamlit run app.py
```

If the dashboard reports that the saved model cannot be loaded (a scikit-learn version difference), run `python rigorous_evaluation.py` once; it rebuilds `models/`. Results are deterministic (fixed random seeds).

## 7. Dashboard guide

| Page | What it does |
|---|---|
| **Factory Overview** | Five simulated machines with live risk cards, advance by 1 / 10 / 50 steps, risk-over-time chart, comparison chart, readings table |
| **Simulation Controls** | Pick a machine, apply a one-click scenario (*Worn tool under load*, *Heavy overload*, *Overheating*, *Light duty*) or set sliders, advance, **replace the tool**, reset; shows the prediction, a per-machine risk trajectory and **why the machine is flagged** |
| **Analytics** | Dataset overview, EDA charts, model comparison, final test metrics with confidence intervals, confusion matrix, PR curve, threshold selection, calibration, importance, simulation history + CSV download |
| **About & Limitations** | Problem, method, colour legend, limitations |

Risk colours: **LOW RISK** (below half the threshold), **WATCH** (between half the threshold and the threshold), **HIGH RISK** (at or above the threshold). A sidebar switch changes the alert sensitivity.

**Quick demo (about 20 seconds):** *Simulation Controls* -> choose MCH-001 -> click **Worn tool under load** -> click **Advance 50 steps**. The machine turns HIGH RISK after roughly 40 steps (some machine types pass through WATCH first) and the dashboard lists the overstrain warning. Click **Replace tool** and the risk falls.

## 8. Limitations (please read)

1. **Simulated live data.** The sensor stream is generated by a simple simulator, not real equipment. The simulator is clamped to the sensor ranges present in the training data so the model is never asked to extrapolate.
2. **Synthetic dataset.** AI4I 2020 is synthetic, so real-world performance is likely lower than the numbers above.
3. **Small test set.** 68 test failures give wide confidence intervals (see section 4).
4. **Unpredictable failures.** RNF and much of TWF are random by design and are largely missed (section 4).
5. **Calibration.** Predicted probabilities are model scores, not confirmed real-world failure probabilities.
6. **Explanations.** The "why flagged" warnings use the dataset's published failure-mode rules for the operator display; they are not model inputs.
7. **Single dataset, single split.** Cross-validation is used for model choice, but the final test is one split.

## 9. Possible future work

Real sensor streams, survival / remaining-useful-life models, cost-sensitive learning with real maintenance costs, SHAP-based per-prediction explanations, drift monitoring.

## 10. Academic use

Created for academic purposes (IA3 Data Science Project). Provided as-is without warranty for production use.
