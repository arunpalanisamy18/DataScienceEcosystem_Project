import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

df = pd.read_csv("ai4i2020_raw.csv")
out = Path("reports")
out.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# 1. Class imbalance
counts = df["Machine failure"].map(
    {0: "No Failure", 1: "Failure"}
).value_counts()

plt.figure(figsize=(7, 5))
ax = sns.barplot(x=counts.index, y=counts.values)
ax.set_title("Machine Failure Class Distribution")
ax.set_xlabel("Class")
ax.set_ylabel("Number of Records")
for container in ax.containers:
    ax.bar_label(container)
plt.tight_layout()
plt.savefig(out / "01_failure_distribution.png", dpi=200)
plt.close()

# 2. Sensor measurements by outcome
features = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear"
]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, col in enumerate(features):
    sns.boxplot(
        data=df,
        x="Machine failure",
        y=col,
        ax=axes[i]
    )
    axes[i].set_title(col)
    axes[i].set_xlabel("Machine failure (0 = No, 1 = Yes)")

axes[-1].axis("off")
fig.suptitle("Sensor Measurements by Machine Failure Status", y=1.02)
plt.tight_layout()
plt.savefig(out / "02_sensor_comparison.png", dpi=200, bbox_inches="tight")
plt.close()

# 3. Correlation analysis on numeric operating measurements
numeric = features + ["Machine failure"]
corr = df[numeric].corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Between Operating Measurements and Failure")
plt.tight_layout()
plt.savefig(out / "03_correlation_heatmap.png", dpi=200)
plt.close()

# 4. Failure rate by machine type
type_rates = (
    df.groupby("Type")["Machine failure"]
    .agg(["count", "sum", "mean"])
    .rename(columns={
        "count": "Records",
        "sum": "Failures",
        "mean": "Failure rate"
    })
)
type_rates["Failure rate (%)"] = type_rates["Failure rate"] * 100
type_rates.to_csv(out / "failure_rate_by_type.csv")

print("=== EDA SUMMARY ===")
print("\nFailure rate by machine type:")
print(type_rates.round(2).to_string())

print("\nMean sensor values by failure status:")
print(df.groupby("Machine failure")[features].mean().round(2).to_string())

print("\nCorrelation with Machine failure:")
print(corr["Machine failure"].sort_values(ascending=False).round(3).to_string())

print("\nSaved charts and summary under the reports folder:")
for file in sorted(out.iterdir()):
    print(file)

print("\nEDA completed.")


# ---------------------------------------------------------------
# Extra EDA: failure modes and physics-based engineered features
# ---------------------------------------------------------------
from features import add_engineered_features

modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
mode_counts = df[modes].sum().sort_values(ascending=False)
mode_counts.rename("Failures").to_csv(out / "failure_mode_counts.csv")

plt.figure(figsize=(7, 5))
ax = sns.barplot(x=mode_counts.index, y=mode_counts.values)
ax.set_title("Failures by Failure Mode (dataset labels, not model inputs)")
ax.set_xlabel("Failure mode")
ax.set_ylabel("Number of failures")
for container in ax.containers:
    ax.bar_label(container)
plt.tight_layout()
plt.savefig(out / "09_failure_modes.png", dpi=200)
plt.close()

eng = add_engineered_features(df)
eng_cols = ["Temp difference", "Power", "Overstrain"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, col in zip(axes, eng_cols):
    sns.boxplot(data=eng, x="Machine failure", y=col, ax=ax)
    ax.set_title(col)
    ax.set_xlabel("Machine failure (0 = No, 1 = Yes)")
fig.suptitle("Engineered Physics-Based Features by Failure Status", y=1.03)
plt.tight_layout()
plt.savefig(out / "10_engineered_features.png", dpi=200, bbox_inches="tight")
plt.close()

print("\nFailure-mode counts:")
print(mode_counts.to_string())
print("\nMean engineered features by failure status:")
print(eng.groupby("Machine failure")[eng_cols].mean().round(1).to_string())

# Failure rate by machine type (chart)
rates = df.groupby("Type")["Machine failure"].mean().mul(100).reset_index()
rates.columns = ["Type", "Failure rate (%)"]
plt.figure(figsize=(7, 5))
ax = sns.barplot(data=rates, x="Type", y="Failure rate (%)")
ax.set_title("Observed Failure Rate by Machine Type")
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f%%")
plt.tight_layout()
plt.savefig(out / "08_failure_by_type.png", dpi=200)
plt.close()
