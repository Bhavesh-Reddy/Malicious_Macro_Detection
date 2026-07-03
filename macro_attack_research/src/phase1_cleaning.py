#!/usr/bin/env python3
"""
PHASE 1: DATA CLEANING, SANITIZATION & STRUCTURAL AUDIT
========================================================
- Strip column header whitespace
- Scan for NaN/Inf anomalies
- Normalize Timestamp to ISO 8601
- Binarize labels (Benign vs Infiltration)
- Build column mapping between 2018 and 2017 schemas
- Output: Markdown metrics table
"""

import pandas as pd
import numpy as np
import os
import warnings
import pickle

warnings.filterwarnings('ignore')

# === PATHS ===
BASE = "/home/ryzen/Desktop/Bhavesh_Intern/macro_attack_research/data"
FILE_2018_WED = os.path.join(BASE, "dataset_a_2018", "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv")
FILE_2018_THU = os.path.join(BASE, "dataset_a_2018", "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv")
FILE_2017     = os.path.join(BASE, "dataset_b_2017", "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")

OUT_DIR = "/home/ryzen/.gemini/antigravity/scratch/nids_infiltration/models"

# ============================================================
# STEP 1: Load and strip column headers
# ============================================================
print("=" * 70)
print("PHASE 1: DATA CLEANING, SANITIZATION & STRUCTURAL AUDIT")
print("=" * 70)

print("\n[1/6] Loading datasets...")
df_2018_wed = pd.read_csv(FILE_2018_WED, low_memory=False)
print(f"  ✓ Wed 28-02-2018: {df_2018_wed.shape}")
df_2018_thu = pd.read_csv(FILE_2018_THU, low_memory=False)
print(f"  ✓ Thu 01-03-2018: {df_2018_thu.shape}")
df_2017     = pd.read_csv(FILE_2017, low_memory=False, encoding='utf-8')
print(f"  ✓ Thu 2017 Infiltration: {df_2017.shape}")

print("\n[2/6] Stripping whitespace from column headers...")
for name, df in [("2018_Wed", df_2018_wed), ("2018_Thu", df_2018_thu), ("2017", df_2017)]:
    before = list(df.columns)
    df.columns = df.columns.str.strip()
    changed = sum(1 for a, b in zip(before, df.columns) if a != b)
    print(f"  {name}: {changed} columns had whitespace stripped")

# ============================================================
# STEP 2: Column mapping — 2017 uses verbose names, 2018 uses abbreviated
# ============================================================
print("\n[3/6] Building column mapping between 2017 → 2018 schema...")

# Canonical mapping: 2017 verbose name → 2018 abbreviated name
COL_MAP_2017_TO_2018 = {
    'Destination Port': 'Dst Port',
    'Flow Duration': 'Flow Duration',
    'Total Fwd Packets': 'Tot Fwd Pkts',
    'Total Backward Packets': 'Tot Bwd Pkts',
    'Total Length of Fwd Packets': 'TotLen Fwd Pkts',
    'Total Length of Bwd Packets': 'TotLen Bwd Pkts',
    'Fwd Packet Length Max': 'Fwd Pkt Len Max',
    'Fwd Packet Length Min': 'Fwd Pkt Len Min',
    'Fwd Packet Length Mean': 'Fwd Pkt Len Mean',
    'Fwd Packet Length Std': 'Fwd Pkt Len Std',
    'Bwd Packet Length Max': 'Bwd Pkt Len Max',
    'Bwd Packet Length Min': 'Bwd Pkt Len Min',
    'Bwd Packet Length Mean': 'Bwd Pkt Len Mean',
    'Bwd Packet Length Std': 'Bwd Pkt Len Std',
    'Flow Bytes/s': 'Flow Byts/s',
    'Flow Packets/s': 'Flow Pkts/s',
    'Flow IAT Mean': 'Flow IAT Mean',
    'Flow IAT Std': 'Flow IAT Std',
    'Flow IAT Max': 'Flow IAT Max',
    'Flow IAT Min': 'Flow IAT Min',
    'Fwd IAT Total': 'Fwd IAT Tot',
    'Fwd IAT Mean': 'Fwd IAT Mean',
    'Fwd IAT Std': 'Fwd IAT Std',
    'Fwd IAT Max': 'Fwd IAT Max',
    'Fwd IAT Min': 'Fwd IAT Min',
    'Bwd IAT Total': 'Bwd IAT Tot',
    'Bwd IAT Mean': 'Bwd IAT Mean',
    'Bwd IAT Std': 'Bwd IAT Std',
    'Bwd IAT Max': 'Bwd IAT Max',
    'Bwd IAT Min': 'Bwd IAT Min',
    'Fwd PSH Flags': 'Fwd PSH Flags',
    'Bwd PSH Flags': 'Bwd PSH Flags',
    'Fwd URG Flags': 'Fwd URG Flags',
    'Bwd URG Flags': 'Bwd URG Flags',
    'Fwd Header Length': 'Fwd Header Len',  # Note: 2017 has this twice
    'Bwd Header Length': 'Bwd Header Len',
    'Fwd Packets/s': 'Fwd Pkts/s',
    'Bwd Packets/s': 'Bwd Pkts/s',
    'Min Packet Length': 'Pkt Len Min',
    'Max Packet Length': 'Pkt Len Max',
    'Packet Length Mean': 'Pkt Len Mean',
    'Packet Length Std': 'Pkt Len Std',
    'Packet Length Variance': 'Pkt Len Var',
    'FIN Flag Count': 'FIN Flag Cnt',
    'SYN Flag Count': 'SYN Flag Cnt',
    'RST Flag Count': 'RST Flag Cnt',
    'PSH Flag Count': 'PSH Flag Cnt',
    'ACK Flag Count': 'ACK Flag Cnt',
    'URG Flag Count': 'URG Flag Cnt',
    'CWE Flag Count': 'CWE Flag Count',  # same in both
    'ECE Flag Count': 'ECE Flag Cnt',
    'Down/Up Ratio': 'Down/Up Ratio',
    'Average Packet Size': 'Pkt Size Avg',
    'Avg Fwd Segment Size': 'Fwd Seg Size Avg',
    'Avg Bwd Segment Size': 'Bwd Seg Size Avg',
    'Fwd Avg Bytes/Bulk': 'Fwd Byts/b Avg',
    'Fwd Avg Packets/Bulk': 'Fwd Pkts/b Avg',
    'Fwd Avg Bulk Rate': 'Fwd Blk Rate Avg',
    'Bwd Avg Bytes/Bulk': 'Bwd Byts/b Avg',
    'Bwd Avg Packets/Bulk': 'Bwd Pkts/b Avg',
    'Bwd Avg Bulk Rate': 'Bwd Blk Rate Avg',
    'Subflow Fwd Packets': 'Subflow Fwd Pkts',
    'Subflow Fwd Bytes': 'Subflow Fwd Byts',
    'Subflow Bwd Packets': 'Subflow Bwd Pkts',
    'Subflow Bwd Bytes': 'Subflow Bwd Byts',
    'Init_Win_bytes_forward': 'Init Fwd Win Byts',
    'Init_Win_bytes_backward': 'Init Bwd Win Byts',
    'act_data_pkt_fwd': 'Fwd Act Data Pkts',
    'min_seg_size_forward': 'Fwd Seg Size Min',
    'Active Mean': 'Active Mean',
    'Active Std': 'Active Std',
    'Active Max': 'Active Max',
    'Active Min': 'Active Min',
    'Idle Mean': 'Idle Mean',
    'Idle Std': 'Idle Std',
    'Idle Max': 'Idle Max',
    'Idle Min': 'Idle Min',
    'Label': 'Label',
}

# Handle duplicate "Fwd Header Length" in 2017 — the second occurrence is 
# actually "Fwd Header Length.1" after pandas dedup
cols_2017_before = list(df_2017.columns)
if cols_2017_before.count('Fwd Header Length') == 2:
    # pandas auto-handles duplicate columns by appending .1
    # Let's check what pandas actually did
    pass

# Rename 2017 columns to match 2018 schema
df_2017.rename(columns=COL_MAP_2017_TO_2018, inplace=True)

# Handle any remaining duplicates from the Fwd Header Length issue
if 'Fwd Header Length.1' in df_2017.columns:
    df_2017.rename(columns={'Fwd Header Length.1': 'Fwd Header Len'}, inplace=True)

# Deduplicate columns — keep the first occurrence of any duplicate column name
df_2017 = df_2017.loc[:, ~df_2017.columns.duplicated(keep='first')]
print(f"  2017 columns after dedup: {df_2017.shape[1]}")

# Check alignment
cols_2018 = set(df_2018_wed.columns)
cols_2017 = set(df_2017.columns)

common = cols_2018 & cols_2017
only_2018 = cols_2018 - cols_2017
only_2017 = cols_2017 - cols_2018

print(f"  Common columns: {len(common)}")
print(f"  Only in 2018: {only_2018}")
print(f"  Only in 2017: {only_2017}")

# ============================================================
# STEP 3: Label binarization
# ============================================================
print("\n[4/6] Binarizing labels...")

def binarize_labels(df, name):
    """Map labels to Benign/Infiltration. Drop other attack types."""
    label_col = 'Label'
    df[label_col] = df[label_col].str.strip()
    
    unique = df[label_col].unique()
    print(f"  {name} unique labels: {unique}")
    
    # Normalize: BENIGN/Benign → 0, Infiltration/Infilteration → 1
    label_map = {}
    for lbl in unique:
        lbl_lower = lbl.lower()
        if lbl_lower == 'benign':
            label_map[lbl] = 'Benign'
        elif 'infil' in lbl_lower:
            label_map[lbl] = 'Infiltration'
        else:
            label_map[lbl] = '__DROP__'
    
    df[label_col] = df[label_col].map(label_map)
    n_dropped = (df[label_col] == '__DROP__').sum()
    if n_dropped > 0:
        print(f"  ⚠ {name}: Dropping {n_dropped} rows with non-Infiltration attack labels")
        df = df[df[label_col] != '__DROP__'].copy()
    
    return df

df_2018_wed = binarize_labels(df_2018_wed, "2018_Wed")
df_2018_thu = binarize_labels(df_2018_thu, "2018_Thu")
df_2017     = binarize_labels(df_2017, "2017")

# ============================================================
# STEP 4: Combine 2018 files and normalize Timestamp
# ============================================================
print("\n[5/6] Normalizing timestamps to ISO 8601...")

# 2018 has Timestamp column in format DD/MM/YYYY HH:MM:SS
df_2018 = pd.concat([df_2018_wed, df_2018_thu], ignore_index=True)
print(f"  Combined 2018 shape: {df_2018.shape}")

# Parse 2018 Timestamp
df_2018['Timestamp'] = pd.to_datetime(df_2018['Timestamp'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
ts_null = df_2018['Timestamp'].isnull().sum()
print(f"  2018 Timestamp parse failures: {ts_null}")

# Format to ISO 8601 string
df_2018['Timestamp'] = df_2018['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

print(f"  2018 Timestamp range: {df_2018['Timestamp'].min()} → {df_2018['Timestamp'].max()}")

# 2017 has NO Timestamp column — note this for Phase 4
print(f"  2017 has Timestamp column: {'Timestamp' in df_2017.columns}")

# ============================================================
# STEP 5: Scan for NaN/Inf anomalies
# ============================================================
print("\n[6/6] Scanning for NaN/Inf/-Inf anomalies...")

def scan_anomalies(df, name):
    """Count NaN, Inf, -Inf across all numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    nan_count = 0
    inf_count = 0
    
    for col in numeric_cols:
        series = df[col]
        # Guard against duplicate column names returning a DataFrame
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        nan_count += int(series.isnull().sum())
        inf_count += int(np.isinf(series.values.astype(float)).sum())
    
    # Also check for string 'Infinity', 'NaN' etc in object columns
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        if col in ('Label', 'Timestamp'):
            continue
        series = df[col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        numeric_converted = pd.to_numeric(series, errors='coerce')
        extra_nan = int(numeric_converted.isnull().sum()) - int(series.isnull().sum())
        if extra_nan > 0:
            nan_count += extra_nan
    
    total = nan_count + inf_count
    print(f"  {name}: NaN={nan_count}, Inf={inf_count}, Total={total}")
    return total

# For 2018, some columns might have 'Infinity' as string — convert numeric columns
print("  Converting 2018 Flow Byts/s and Flow Pkts/s to numeric...")
for col in ['Flow Byts/s', 'Flow Pkts/s']:
    if col in df_2018.columns:
        df_2018[col] = pd.to_numeric(df_2018[col], errors='coerce')

# Same for 2017
for col in ['Flow Byts/s', 'Flow Pkts/s']:
    if col in df_2017.columns:
        df_2017[col] = pd.to_numeric(df_2017[col], errors='coerce')

anomaly_2018 = scan_anomalies(df_2018, "2018_Combined")
anomaly_2017 = scan_anomalies(df_2017, "2017")

# Replace Inf with NaN, then impute with column medians (from 2018 training data only — leakage-free)
print("\n  Replacing Inf/-Inf with NaN...")
numeric_cols_2018 = df_2018.select_dtypes(include=[np.number]).columns
numeric_cols_2017 = df_2017.select_dtypes(include=[np.number]).columns

df_2018[numeric_cols_2018] = df_2018[numeric_cols_2018].replace([np.inf, -np.inf], np.nan)
df_2017[numeric_cols_2017] = df_2017[numeric_cols_2017].replace([np.inf, -np.inf], np.nan)

# Compute medians from 2018 ONLY (these will be used for imputation)
# NOTE: In Phase 2 we'll recompute from only the TRAINING split — for now we impute 
# to get clean data, then re-impute properly during the split phase
medians_2018 = df_2018[numeric_cols_2018].median()
nan_before_2018 = df_2018[numeric_cols_2018].isnull().sum().sum()
nan_before_2017 = df_2017[numeric_cols_2017].isnull().sum().sum()

df_2018[numeric_cols_2018] = df_2018[numeric_cols_2018].fillna(medians_2018)
# For 2017, impute with 2018 medians (to prevent leakage from 2017 into model)
common_numeric = [c for c in numeric_cols_2017 if c in medians_2018.index]
for col in common_numeric:
    df_2017[col] = df_2017[col].fillna(medians_2018[col])
# Any remaining NaN in 2017-only columns, fill with their own median
remaining_nan_cols = df_2017.columns[df_2017.isnull().any()].tolist()
if remaining_nan_cols:
    df_2017[remaining_nan_cols] = df_2017[remaining_nan_cols].fillna(df_2017[remaining_nan_cols].median())

print(f"  2018: Imputed {nan_before_2018} NaN values with column medians")
print(f"  2017: Imputed {nan_before_2017} NaN values")

# Verify clean
assert df_2018[numeric_cols_2018].isnull().sum().sum() == 0, "2018 still has NaN!"
assert df_2017[numeric_cols_2017].isnull().sum().sum() == 0, "2017 still has NaN!"
print("  ✓ All anomalies resolved — no remaining NaN/Inf")

# ============================================================
# OUTPUT: Phase 1 Metrics Table
# ============================================================
print("\n" + "=" * 70)
print("PHASE 1 OUTPUT: Structural Audit Summary")
print("=" * 70)

# Count labels
def count_labels(df):
    benign = (df['Label'] == 'Benign').sum()
    infil = (df['Label'] == 'Infiltration').sum()
    return benign, infil

b_wed, i_wed = count_labels(df_2018_wed)
b_thu, i_thu = count_labels(df_2018_thu)
b_2018, i_2018 = count_labels(df_2018)
b_2017, i_2017 = count_labels(df_2017)

print(f"""
| File Name | Total Rows | Total Columns | Benign Count | Infiltration Count | Total Anomaly/NaN Count |
|-----------|-----------|---------------|-------------|-------------------|------------------------|
| Wed-28-02-2018 | {len(df_2018_wed):,} | {df_2018_wed.shape[1]} | {b_wed:,} | {i_wed:,} | (included in combined) |
| Thu-01-03-2018 | {len(df_2018_thu):,} | {df_2018_thu.shape[1]} | {b_thu:,} | {i_thu:,} | (included in combined) |
| **2018 Combined** | **{len(df_2018):,}** | **{df_2018.shape[1]}** | **{b_2018:,}** | **{i_2018:,}** | **{anomaly_2018:,}** |
| **2017 Infiltration** | **{len(df_2017):,}** | **{df_2017.shape[1]}** | **{b_2017:,}** | **{i_2017:,}** | **{anomaly_2017:,}** |
""")

# Class imbalance analysis
ratio_2018 = b_2018 / max(i_2018, 1)
ratio_2017 = b_2017 / max(i_2017, 1)
print(f"Class Imbalance Ratio (Benign:Infiltration):")
print(f"  2018: {ratio_2018:.1f}:1")
print(f"  2017: {ratio_2017:.1f}:1")
print(f"  Infiltration prevalence 2018: {i_2018/len(df_2018)*100:.4f}%")
print(f"  Infiltration prevalence 2017: {i_2017/len(df_2017)*100:.4f}%")

# ============================================================
# SAVE cleaned dataframes for Phase 2
# ============================================================
print("\nSaving cleaned dataframes...")
df_2018.to_parquet(os.path.join(OUT_DIR, "df_2018_cleaned.parquet"), index=False)
df_2017.to_parquet(os.path.join(OUT_DIR, "df_2017_cleaned.parquet"), index=False)

# Save medians for leakage-free imputation later
pd.to_pickle(medians_2018, os.path.join(OUT_DIR, "medians_2018.pkl"))

# Save column mapping for Phase 4
pd.to_pickle(COL_MAP_2017_TO_2018, os.path.join(OUT_DIR, "col_map_2017_to_2018.pkl"))

print(f"  ✓ Saved to {OUT_DIR}")
print("\n[STATE: PAUSE — Awaiting user confirmation of Phase 1 metrics before proceeding]")
