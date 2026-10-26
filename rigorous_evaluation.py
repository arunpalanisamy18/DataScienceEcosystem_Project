"""Train, tune, validate and test the predictive-maintenance model.

Data protocol (the test set is touched exactly once, at the very end):

    100% data
      |- 20%  FINAL TEST  (never used for any choice)
      '- 80%  development
            |- 75% of dev (= 60% overall) -> TRAIN       (fit + cross-validated tuning)
            '- 25% of dev (= 20% overall) -> VALIDATION  (decision-threshold selection)

Outputs: models/*.joblib, models/model_config.json, reports/*.csv, reports/*.png
"""
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from features import (
    CATEGORICAL, ENGINEERED, NUMERIC_ALL, RAW_FEATURES, RAW_NUMERIC,
    add_engineered_features,
)

warnings.filterwarnings("ignore")
SEED = 42
COST_FN, COST_FP = 10.0, 1.0     # assumed: a missed failure costs 10x a false alarm
N_BOOT = 1000

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports"
MODELS = ROOT / "models"
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")

df = pd.read_csv(ROOT / "ai4i2020_raw.csv")
X, y = df[RAW_FEATURES], df["Machine failure"]

# ------------------------------------------------------------------ splits
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.25, random_state=SEED, stratify=y_dev)

split_table = pd.DataFrame({
    "Split": ["Train", "Validation", "Test"],
    "Records": [len(X_train), len(X_val), len(X_test)],
    "Failures": [int(y_train.sum()), int(y_val.sum()), int(y_test.sum())],
})
split_table["Failure rate (%)"] = (split_table["Failures"] / split_table["Records"] * 100).round(2)
split_table.to_csv(OUT / "data_split_summary.csv", index=False)
print(split_table.to_string(index=False))


# ------------------------------------------------------------- pipelines
def make_pipeline(model, engineered=True):
    numeric = NUMERIC_ALL if engineered else RAW_NUMERIC
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encode", OneHotEncoder(handle_unknown="ignore"))]),
         CATEGORICAL),
    ])
    steps = []
    if engineered:
        steps.append(("engineer", FunctionTransformer(add_engineered_features)))
    steps += [("preprocess", pre), ("model", model)]
    return Pipeline(steps)


candidates = {
    "Dummy baseline": make_pipeline(DummyClassifier(strategy="most_frequent"), False),
    "Logistic Regression (raw)": make_pipeline(
        LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED), False),
    "Random Forest (raw)": make_pipeline(
        RandomForestClassifier(n_estimators=300, class_weight="balanced",
                               min_samples_leaf=2, random_state=SEED, n_jobs=-1), False),
    "Gradient Boosting (raw)": make_pipeline(
        GradientBoostingClassifier(random_state=SEED), False),
    "Logistic Regression (+features)": make_pipeline(
        LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED)),
    "Random Forest (+features)": make_pipeline(
        RandomForestClassifier(n_estimators=300, class_weight="balanced",
                               min_samples_leaf=2, random_state=SEED, n_jobs=-1)),
    "Gradient Boosting (+features)": make_pipeline(
        GradientBoostingClassifier(random_state=SEED)),
}

# ------------------------------------- 1) cross-validated model comparison
# Uses the TRAIN split only, so validation/test stay untouched.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = {"pr_auc": "average_precision", "roc_auc": "roc_auc",
           "precision": "precision", "recall": "recall", "f1": "f1"}
rows = []
print("\n=== 5-fold cross-validation on the training split ===")
for name, pipe in candidates.items():
    t0 = time.time()
    res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    row = {"Model": name}
    for key in scoring:
        vals = res[f"test_{key}"]
        row[f"{key} mean"] = vals.mean()
        row[f"{key} std"] = vals.std()
    rows.append(row)
    print(f"{name:34s} PR-AUC {row['pr_auc mean']:.3f} +/- {row['pr_auc std']:.3f}"
          f"  ({time.time() - t0:.0f}s)")
cv_df = pd.DataFrame(rows).sort_values("pr_auc mean", ascending=False)
cv_df.to_csv(OUT / "cv_model_comparison.csv", index=False)

# ----------------------------------- 2) hyper-parameter tuning (train only)
print("\n=== Randomised hyper-parameter search (Gradient Boosting + features) ===")
param_dist = {
    "model__n_estimators": [100, 200, 300, 400],
    "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
    "model__max_depth": [2, 3, 4, 5],
    "model__subsample": [0.7, 0.85, 1.0],
    "model__min_samples_leaf": [1, 3, 5, 10],
}
search = RandomizedSearchCV(
    make_pipeline(GradientBoostingClassifier(random_state=SEED)),
    param_distributions=param_dist, n_iter=16, scoring="average_precision",
    cv=cv, random_state=SEED, n_jobs=1, refit=True)
search.fit(X_train, y_train)
model = search.best_estimator_
best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
print("Best parameters:", best_params)
print(f"Best CV PR-AUC: {search.best_score_:.4f}")
pd.DataFrame(search.cv_results_)[
    ["mean_test_score", "std_test_score", "rank_test_score"]
    + [f"param_{k}" for k in param_dist]
].sort_values("rank_test_score").to_csv(OUT / "tuning_results.csv", index=False)

# ---------------------------------- 3) thresholds chosen on VALIDATION only
val_prob = model.predict_proba(X_val)[:, 1]
grid = np.round(np.arange(0.02, 0.96, 0.01), 2)
rows = []
for t in grid:
    pred = (val_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
    rows.append({
        "Threshold": t,
        "Precision": precision_score(y_val, pred, zero_division=0),
        "Recall": recall_score(y_val, pred, zero_division=0),
        "F1": f1_score(y_val, pred, zero_division=0),
        "FP": fp, "FN": fn,
        "Cost": COST_FN * fn + COST_FP * fp,
    })
val_search = pd.DataFrame(rows)
val_search.to_csv(OUT / "validation_threshold_search.csv", index=False)
thr_f1 = float(val_search.loc[val_search["F1"].idxmax(), "Threshold"])
# High-recall mode: the highest threshold that still catches >= 85% of
# validation failures. (Minimising the 10:1 cost directly picked an unstable
# threshold of 0.03 on this small validation set, so a recall target is used.)
RECALL_TARGET = 0.85
thr_cost = float(val_search.loc[val_search["Recall"] >= RECALL_TARGET, "Threshold"].max())
print(f"\nValidation-selected thresholds: balanced(F1)={thr_f1:.2f}  "
      f"high-recall(recall>={RECALL_TARGET:.0%})={thr_cost:.2f}")

# ------------------------------------------------- 4) FINAL TEST (once)
test_prob = model.predict_proba(X_test)[:, 1]


def evaluate(prob, threshold, truth):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(truth, pred),
        "Precision": precision_score(truth, pred, zero_division=0),
        "Recall": recall_score(truth, pred, zero_division=0),
        "F1": f1_score(truth, pred, zero_division=0),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Cost": COST_FN * fn + COST_FP * fp,
    }


final_rows = []
for mode, thr in [("Balanced (max F1)", thr_f1), ("High-recall (validation recall >= 85%)", thr_cost)]:
    final_rows.append({"Operating mode": mode, "Threshold": thr,
                       **evaluate(test_prob, thr, y_test),
                       "PR-AUC": average_precision_score(y_test, test_prob),
                       "ROC-AUC": roc_auc_score(y_test, test_prob)})

# baseline = untuned raw-feature Gradient Boosting under the same protocol
base = candidates["Gradient Boosting (raw)"].fit(X_train, y_train)
base_val = base.predict_proba(X_val)[:, 1]
base_thr = float(max(grid, key=lambda t: f1_score(y_val, (base_val >= t).astype(int), zero_division=0)))
base_test = base.predict_proba(X_test)[:, 1]
final_rows.append({"Operating mode": "BASELINE: untuned GB, raw features",
                   "Threshold": base_thr, **evaluate(base_test, base_thr, y_test),
                   "PR-AUC": average_precision_score(y_test, base_test),
                   "ROC-AUC": roc_auc_score(y_test, base_test)})
final_df = pd.DataFrame(final_rows)
final_df.to_csv(OUT / "final_test_metrics.csv", index=False)
print("\n=== FINAL HELD-OUT TEST RESULTS ===")
print(final_df.round(4).to_string(index=False))

# ---------------------------------------- bootstrap confidence intervals
rng = np.random.default_rng(SEED)
yt = y_test.to_numpy()
boot = {"Precision": [], "Recall": [], "F1": [], "PR-AUC": []}
for _ in range(N_BOOT):
    idx = rng.integers(0, len(yt), len(yt))
    if yt[idx].sum() == 0:
        continue
    pred = (test_prob[idx] >= thr_f1).astype(int)
    boot["Precision"].append(precision_score(yt[idx], pred, zero_division=0))
    boot["Recall"].append(recall_score(yt[idx], pred, zero_division=0))
    boot["F1"].append(f1_score(yt[idx], pred, zero_division=0))
    boot["PR-AUC"].append(average_precision_score(yt[idx], test_prob[idx]))
ci = pd.DataFrame({
    "Metric": list(boot),
    "Estimate": [final_df.loc[0, m] for m in boot],
    "CI 2.5%": [np.percentile(v, 2.5) for v in boot.values()],
    "CI 97.5%": [np.percentile(v, 97.5) for v in boot.values()],
})
ci.to_csv(OUT / "bootstrap_confidence_intervals.csv", index=False)
print("\nBootstrap 95% CIs (balanced threshold):")
print(ci.round(3).to_string(index=False))

# ---------------------------------------------------------- error analysis
modes = df.loc[X_test.index, ["TWF", "HDF", "PWF", "OSF", "RNF"]]
pred_bal = (test_prob >= thr_f1).astype(int)
err_rows = []
for m in modes.columns:
    mask = (modes[m] == 1).to_numpy()
    n = int(mask.sum())
    caught = int(pred_bal[mask].sum()) if n else 0
    err_rows.append({"Failure mode": m, "Failures in test set": n,
                     "Detected": caught,
                     "Recall": caught / n if n else np.nan})
err_df = pd.DataFrame(err_rows)
err_df.to_csv(OUT / "recall_by_failure_mode.csv", index=False)
print("\nRecall by failure mode (balanced threshold):")
print(err_df.round(3).to_string(index=False))

pd.DataFrame({"Actual failure": yt, "Predicted probability": test_prob,
              "Predicted failure (balanced)": pred_bal}
             ).to_csv(OUT / "final_test_predictions.csv", index=False)

# ------------------------------------------------------ permutation importance
perm = permutation_importance(model, X_val, y_val, scoring="average_precision",
                              n_repeats=15, random_state=SEED, n_jobs=1)
imp = pd.DataFrame({"Feature": X_val.columns, "Importance": perm.importances_mean,
                    "Std": perm.importances_std}).sort_values("Importance", ascending=False)
imp.to_csv(OUT / "permutation_importance.csv", index=False)

pre = model.named_steps["preprocess"]
names = [n.replace("num__", "").replace("cat__", "") for n in pre.get_feature_names_out()]
tree_imp = pd.DataFrame({"Feature": names,
                         "Importance": model.named_steps["model"].feature_importances_}
                        ).sort_values("Importance", ascending=False)
tree_imp.to_csv(OUT / "feature_importance.csv", index=False)

# ----------------------------------------------------------------- charts
GREEN, RED, BLUE = "#2E8B57", "#C0392B", "#2C6FBB"

# 05 model comparison (CV)
plot_df = cv_df.sort_values("pr_auc mean")
plt.figure(figsize=(9, 5.5))
plt.barh(plot_df["Model"], plot_df["pr_auc mean"], xerr=plot_df["pr_auc std"],
         color=[GREEN if "+features" in m else BLUE for m in plot_df["Model"]], capsize=3)
plt.xlabel("Cross-validated PR-AUC (mean +/- std, 5 folds, training split)")
plt.title("Model Comparison  (green = with engineered features)")
plt.tight_layout(); plt.savefig(OUT / "05_model_comparison.png", dpi=200); plt.close()

# 06 confusion matrix
cm = confusion_matrix(y_test, pred_bal, labels=[0, 1])
plt.figure(figsize=(5.5, 4.8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["No failure", "Failure"], yticklabels=["No failure", "Failure"])
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title(f"Test Confusion Matrix (threshold {thr_f1:.2f})")
plt.tight_layout(); plt.savefig(OUT / "06_confusion_matrix.png", dpi=200); plt.close()

# 07 precision-recall: baseline vs final
plt.figure(figsize=(7, 5.5))
for label, p, c in [("Final model (+features, tuned)", test_prob, GREEN),
                    ("Baseline (raw features, untuned)", base_test, BLUE)]:
    pr, rc, _ = precision_recall_curve(y_test, p)
    plt.plot(rc, pr, color=c, lw=2, label=f"{label}  AP={average_precision_score(y_test, p):.3f}")
plt.axhline(y_test.mean(), color="grey", ls="--", label="No-skill")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(loc="lower left")
plt.title("Precision-Recall Curves (held-out test set)")
plt.tight_layout(); plt.savefig(OUT / "07_precision_recall.png", dpi=200); plt.close()

# 11 ROC
plt.figure(figsize=(6, 5.5))
for label, p, c in [("Final model", test_prob, GREEN), ("Baseline", base_test, BLUE)]:
    fpr, tpr, _ = roc_curve(y_test, p)
    plt.plot(fpr, tpr, color=c, lw=2, label=f"{label}  AUC={roc_auc_score(y_test, p):.3f}")
plt.plot([0, 1], [0, 1], "--", color="grey")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend()
plt.title("ROC Curves (held-out test set)")
plt.tight_layout(); plt.savefig(OUT / "11_roc_curve.png", dpi=200); plt.close()

# 12 threshold trade-off (validation)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(val_search["Threshold"], val_search["Precision"], label="Precision", color=BLUE)
ax.plot(val_search["Threshold"], val_search["Recall"], label="Recall", color=RED)
ax.plot(val_search["Threshold"], val_search["F1"], label="F1", color=GREEN)
ax.axvline(thr_f1, color=GREEN, ls="--", label=f"Balanced = {thr_f1:.2f}")
ax.axvline(thr_cost, color=RED, ls="--", label=f"High-recall = {thr_cost:.2f}")
ax.set_xlabel("Decision threshold"); ax.set_ylabel("Score (validation set)")
ax.set_title("Threshold Selection on the Validation Set"); ax.legend()
plt.tight_layout(); plt.savefig(OUT / "12_threshold_selection.png", dpi=200); plt.close()

# 13 calibration
frac, mean_pred = calibration_curve(y_test, test_prob, n_bins=8, strategy="quantile")
plt.figure(figsize=(5.8, 5.5))
plt.plot(mean_pred, frac, "o-", color=GREEN, label="Final model")
plt.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
plt.xlabel("Mean predicted probability"); plt.ylabel("Observed failure fraction")
plt.title("Calibration Curve (test set)"); plt.legend()
plt.tight_layout(); plt.savefig(OUT / "13_calibration.png", dpi=200); plt.close()

# 04 importance (permutation, validation) + tree importance
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
p_sorted = imp.sort_values("Importance")
axes[0].barh(p_sorted["Feature"], p_sorted["Importance"], xerr=p_sorted["Std"], color=GREEN)
axes[0].set_title("Permutation importance (drop in PR-AUC, validation)")
t_sorted = tree_imp.sort_values("Importance").tail(10)
axes[1].barh(t_sorted["Feature"], t_sorted["Importance"], color=BLUE)
axes[1].set_title("Gradient Boosting impurity importance")
plt.tight_layout(); plt.savefig(OUT / "04_feature_importance.png", dpi=200); plt.close()

# 14 recall by failure mode
plt.figure(figsize=(7, 4.5))
ax = sns.barplot(data=err_df, x="Failure mode", y="Recall", color=GREEN)
for c, (_, r) in zip(ax.containers[0], err_df.iterrows()):
    ax.text(c.get_x() + c.get_width() / 2, c.get_height() + 0.01,
            f"{int(r['Detected'])}/{int(r['Failures in test set'])}", ha="center")
plt.ylim(0, 1.12); plt.title("Share of Failures Detected, by Failure Mode (test set)")
plt.tight_layout(); plt.savefig(OUT / "14_recall_by_failure_mode.png", dpi=200); plt.close()

# ------------------------------------------------ save model + configuration
joblib.dump(model, MODELS / "gradient_boosting_pipeline.joblib")
config = {
    "model_name": "Gradient Boosting (tuned, engineered features)",
    "features": RAW_FEATURES,
    "engineered_features": ENGINEERED,
    "target": "Machine failure",
    "dataset": "UCI AI4I 2020 Predictive Maintenance Dataset",
    "threshold": thr_f1,
    "thresholds": {"balanced": thr_f1, "high_recall": thr_cost},
    "cost_ratio_fn_to_fp": COST_FN / COST_FP,
    "high_recall_target": RECALL_TARGET,
    "best_params": best_params,
    "train_range": {c: [float(df[c].min()), float(df[c].max())] for c in RAW_NUMERIC},
}
(MODELS / "model_config.json").write_text(json.dumps(config, indent=4), encoding="utf-8")
print("\nSaved model and configuration. Evaluation completed.")
