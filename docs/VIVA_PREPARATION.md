# Viva Preparation Notes

## 30-second pitch
"I built a system that predicts machine failure from sensor readings. The dataset has only 3.4% failures, so I judged it with precision, recall, F1 and PR-AUC instead of accuracy. I added three physics-based features, tuned a Gradient Boosting model with cross-validation, chose the alert threshold on a separate validation set, and tested once on held-out data. F1 went from 0.75 to 0.90. The model sits behind a dashboard with a five-machine simulator."

## Numbers to remember
| Item | Value |
|---|---|
| Dataset | AI4I 2020, 10,000 rows, 339 failures (3.39%) |
| Split | 60% train / 20% validation / 20% test (stratified) |
| Final model | Gradient Boosting, 100 trees, lr 0.05, depth 3 (+3 engineered features) |
| Balanced threshold | 0.32 (max F1 on validation) |
| Test result | Precision 96.6%, Recall 83.8%, F1 89.8%, PR-AUC 0.901, ROC-AUC 0.970 |
| Confusion matrix | TN 1930, FP 2, FN 11, TP 57 |
| Baseline (raw, untuned) | F1 75.2%, PR-AUC 0.782 |
| 95% CI for F1 | 0.835 to 0.947 |
| Detected by mode | PWF 100%, OSF 100%, HDF 97%, TWF 20%, RNF 0% |

## Likely questions and answers
**Why not use accuracy?** 96.6% of machines are healthy, so "always healthy" scores 96.6% accuracy and finds nothing.

**What is target leakage and how did you avoid it?** TWF/HDF/PWF/OSF/RNF are the causes of the failure label; using them would give away the answer. They are excluded, and a unit test checks it.

**Why PR-AUC?** With rare positives, PR-AUC reflects precision and recall on the failure class; ROC-AUC looks good even for weak models because of the many easy negatives.

**What do the engineered features mean?** Temperature gap (heat dissipation), power = torque x angular speed, overstrain = wear x torque. They encode how the failures arise, using only allowed inputs. They raised CV PR-AUC from 0.80 to 0.90.

**Why three splits?** Training for fitting and cross-validation, validation for the threshold, test used once. Otherwise the test score would be optimistically biased.

**Why threshold 0.32 and not 0.5?** F1 on validation peaked there; with rare failures the best cut-off is usually below 0.5. It is a business choice, so the dashboard has a high-recall mode (0.05).

**Why is recall only 84%?** The misses are mostly RNF (random) and TWF (random wear level 200-240). They have no sensor signature, so no model could catch them. Power, overstrain and heat failures are caught 97-100%.

**Is 0.90 F1 reliable?** Only 68 test failures, so the bootstrap interval is 0.84 to 0.95. I report "about 0.90".

**Is the dashboard data real?** No. It is simulated and clamped to the training ranges so the model never extrapolates. The dataset is synthetic too, so real performance is likely lower.

**How would you improve it?** Real telemetry, remaining-useful-life models, SHAP explanations, real maintenance costs for the threshold, drift monitoring.

**What did you correct from your first version?** The first README claimed a 70/15/15 split but the code produced about 55/15/30, and it said the simulation could not trigger alerts when it could (only with unrealistic values). Both were fixed, and the simulator now stays within realistic sensor ranges.

## 20-second live demo
1. `streamlit run app.py` -> Simulation Controls -> select MCH-001.
2. Click **Worn tool under load**, then **Advance 50 steps**.
3. Show the status turn HIGH RISK, the risk trajectory and the "why" warnings.
4. Click **Replace tool** -> risk drops.
5. Show Analytics for the metrics, confusion matrix and confidence intervals.
