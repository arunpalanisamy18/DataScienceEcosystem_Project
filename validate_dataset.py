import pandas as pd
from ucimlrepo import fetch_ucirepo

print("Downloading AI4I 2020 dataset from UCI...")
dataset = fetch_ucirepo(id=601)

X = dataset.data.features
y = dataset.data.targets
df = pd.concat([X, y], axis=1)

print("\n===== DATASET OVERVIEW =====")
print("Dataset:", dataset.metadata.name)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("\nColumn names:")
print(df.columns.tolist())

print("\n===== DATA TYPES =====")
print(df.dtypes.to_string())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum().to_string())
print("Total missing values:", int(df.isnull().sum().sum()))

print("\n===== DUPLICATES =====")
print("Duplicate rows:", int(df.duplicated().sum()))

target = "Machine failure"
if target in df.columns:
    print("\n===== FAILURE DISTRIBUTION =====")
    print(df[target].value_counts(dropna=False).sort_index().to_string())
    print("\nFailure percentages:")
    print((df[target].value_counts(normalize=True) * 100)
          .sort_index().round(2).to_string())
else:
    print("\nWARNING: Target column not found.")

print("\n===== FIRST FIVE RECORDS =====")
print(df.head().to_string(index=False))

print("\n===== NUMERIC SUMMARY =====")
print(df.describe().round(2).to_string())

df.to_csv("ai4i2020_raw.csv", index=False)
print("\nSaved dataset: ai4i2020_raw.csv")
print("Validation completed.")
