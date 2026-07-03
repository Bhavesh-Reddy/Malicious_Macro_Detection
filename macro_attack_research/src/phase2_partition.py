#!/usr/bin/env python3
"""
PHASE 2: CHRONOLOGICAL PARTITIONING & FEATURE DISCOVERY
========================================================
- Drop fingerprinting columns (Flow ID, Source IP, Destination IP, Source Port)
- Sort 2018 data chronologically by Timestamp
- 70/10/20 temporal split (train/val/test)
- Fit baseline Random Forest for feature importances
- Output: Top 10 features by Gini importance
"""

import pandas as pd
import numpy as np
import os
import time
import pickle
import warnings
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

# === PATHS ===
MODEL_DIR = "/home/ryzen/.gemini/antigravity/scratch/nids_infiltration/models"

# ============================================================
# STEP 1: Load cleaned data from Phase 1
# ============================================================
print("=" * 70)
print("PHASE 2: CHRONOLOGICAL PARTITIONING & FEATURE DISCOVERY")
print("=" * 70)

print("\n[1/5] Loading cleaned 2018 data from Phase 1...")
df_2018 = pd.read_parquet(os.path.join(MODEL_DIR, "df_2018_cleaned.parquet"))
print(f"  Shape: {df_2018.shape}")
print(f"  Columns: {list(df_2018.columns)[:10]}... ({df_2018.shape[1]} total)")

# ============================================================
# STEP 2: Drop fingerprinting / high-cardinality metadata columns
# ============================================================
print("\n[2/5] Dropping fingerprinting columns...")

# 2018 doesn't have Flow ID, Source IP, Destination IP, Source Port as columns
# It has: Dst Port, Protocol, Timestamp, and 77 numeric features + Label
# The columns to drop per the plan: Flow ID, Source IP, Destination IP, Source Port
# In 2018 schema, these aren't present (CICFlowMeter CSV only has Dst Port)
# We DO need to drop Timestamp (after sorting) and Protocol (metadata, not behavioral)

drop_cols = []
for col in ['Flow ID', 'Source IP', 'Src IP', 'Destination IP', 'Dst IP', 'Source Port', 'Src Port']:
    if col in df_2018.columns:
        drop_cols.append(col)
        print(f"  Dropping: {col}")

if not drop_cols:
    print("  No fingerprinting columns found (Flow ID, Source/Dst IP, Src Port not in CICFlowMeter CSV)")
    print("  Note: 'Dst Port' is retained as a behavioral feature (destination service indicator)")

# ============================================================
# STEP 3: Sort chronologically and perform temporal split
# ============================================================
print("\n[3/5] Sorting chronologically and performing 70/10/20 temporal split...")

# Sort by Timestamp
df_2018['Timestamp'] = pd.to_datetime(df_2018['Timestamp'])
df_2018 = df_2018.sort_values('Timestamp').reset_index(drop=True)

print(f"  Timestamp range: {df_2018['Timestamp'].min()} → {df_2018['Timestamp'].max()}")

# Verify monotonically non-decreasing
assert df_2018['Timestamp'].is_monotonic_increasing, "Timestamps not sorted!"
print("  ✓ Timestamps verified monotonically non-decreasing")

# Now drop Timestamp (and Protocol) — they are not behavioral features for the model
# Keep Timestamp values for reference before dropping
timestamps = df_2018['Timestamp'].copy()

meta_drop = ['Timestamp']
if 'Protocol' in df_2018.columns:
    meta_drop.append('Protocol')
    print(f"  Dropping 'Protocol' column (metadata, not a behavioral feature)")

df_2018 = df_2018.drop(columns=meta_drop + drop_cols)
print(f"  Shape after dropping metadata: {df_2018.shape}")

# 70/10/20 temporal split
n = len(df_2018)
n_train = int(n * 0.70)
n_val = int(n * 0.10)
n_test = n - n_train - n_val

df_train = df_2018.iloc[:n_train].copy()
df_val   = df_2018.iloc[n_train:n_train + n_val].copy()
df_test  = df_2018.iloc[n_train + n_val:].copy()

print(f"\n  Split sizes:")
print(f"    Training (70%):   {len(df_train):,} rows  [{timestamps.iloc[0]} → {timestamps.iloc[n_train-1]}]")
print(f"    Validation (10%): {len(df_val):,} rows  [{timestamps.iloc[n_train]} → {timestamps.iloc[n_train+n_val-1]}]")
print(f"    Testing (20%):    {len(df_test):,} rows  [{timestamps.iloc[n_train+n_val]} → {timestamps.iloc[-1]}]")

# Class distribution per split
for name, split in [("Training", df_train), ("Validation", df_val), ("Testing", df_test)]:
    benign = (split['Label'] == 'Benign').sum()
    infil = (split['Label'] == 'Infiltration').sum()
    print(f"    {name}: Benign={benign:,}, Infiltration={infil:,} ({infil/len(split)*100:.2f}%)")

# ============================================================
# STEP 4: Prepare features and leakage-free imputation
# ============================================================
print("\n[4/5] Preparing features with leakage-free processing...")

# Separate features and labels
X_train = df_train.drop(columns=['Label'])
y_train = (df_train['Label'] == 'Infiltration').astype(int)

X_val = df_val.drop(columns=['Label'])
y_val = (df_val['Label'] == 'Infiltration').astype(int)

X_test = df_test.drop(columns=['Label'])
y_test = (df_test['Label'] == 'Infiltration').astype(int)

# Force all feature columns to numeric (some may be object due to repeated CSV headers)
print("  Converting all features to numeric dtype...")
obj_cols_found = []
for X_name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
    for col in X.columns:
        if X[col].dtype == object:
            if col not in obj_cols_found:
                obj_cols_found.append(col)
            X[col] = pd.to_numeric(X[col], errors='coerce')
if obj_cols_found:
    print(f"  Converted {len(obj_cols_found)} object columns to numeric: {obj_cols_found[:5]}...")
else:
    print("  All columns already numeric")

# Re-impute using ONLY training medians (leakage-free)
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
train_medians = X_train.median()

for X in [X_train, X_val, X_test]:
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(train_medians, inplace=True)

# Verify no remaining issues
assert X_train.isnull().sum().sum() == 0, "Training still has NaN!"
assert X_val.isnull().sum().sum() == 0, "Validation still has NaN!"
assert X_test.isnull().sum().sum() == 0, "Testing still has NaN!"
print("  ✓ Leakage-free imputation complete (medians from training set only)")
print(f"  Feature count: {X_train.shape[1]}")
print(f"  All dtypes numeric: {all(X_train.dtypes.apply(lambda x: np.issubdtype(x, np.number)))}")

# ============================================================
# STEP 5: Baseline Random Forest for feature importance
# ============================================================
print("\n[5/5] Fitting baseline Random Forest for Gini feature importance...")
print("  (This may take 1-2 minutes on ~660K training samples)")

t0 = time.time()
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
elapsed = time.time() - t0
print(f"  ✓ Random Forest trained in {elapsed:.1f}s")

# Feature importances
importances = pd.Series(rf.feature_importances_, index=X_train.columns)
importances = importances.sort_values(ascending=False)

# ============================================================
# OUTPUT: Top 10 features
# ============================================================
print("\n" + "=" * 70)
print("PHASE 2 OUTPUT: Top 10 Features by Gini Importance")
print("=" * 70)

print()
for i, (feat, score) in enumerate(importances.head(10).items(), 1):
    print(f"  {i:2d}. {feat:<30s}  {score:.6f}")

print(f"\n  Total features: {len(importances)}")
print(f"  Top-10 cumulative importance: {importances.head(10).sum():.4f}")
print(f"  Bottom 50% features contribute: {importances.tail(len(importances)//2).sum():.4f}")

# ============================================================
# SAVE artifacts for Phase 3
# ============================================================
print("\nSaving Phase 2 artifacts...")

# Save splits
X_train.to_parquet(os.path.join(MODEL_DIR, "X_train.parquet"), index=False)
X_val.to_parquet(os.path.join(MODEL_DIR, "X_val.parquet"), index=False)
X_test.to_parquet(os.path.join(MODEL_DIR, "X_test.parquet"), index=False)
y_train.to_frame('label').to_parquet(os.path.join(MODEL_DIR, "y_train.parquet"), index=False)
y_val.to_frame('label').to_parquet(os.path.join(MODEL_DIR, "y_val.parquet"), index=False)
y_test.to_frame('label').to_parquet(os.path.join(MODEL_DIR, "y_test.parquet"), index=False)

# Save training medians for Phase 4 (OOD imputation)
pd.to_pickle(train_medians, os.path.join(MODEL_DIR, "train_medians.pkl"))

# Save feature list for Phase 4 alignment
pd.to_pickle(list(X_train.columns), os.path.join(MODEL_DIR, "feature_columns.pkl"))

# Save Random Forest baseline model
pd.to_pickle(rf, os.path.join(MODEL_DIR, "model_rf_baseline.pkl"))

# Save feature importances
importances.to_csv(os.path.join(MODEL_DIR, "feature_importances_rf.csv"))

print(f"  ✓ Saved to {MODEL_DIR}")
print("\n[STATE: PAUSE — Awaiting user confirmation of Phase 2 metrics before proceeding]")
