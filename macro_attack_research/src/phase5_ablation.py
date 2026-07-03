#!/usr/bin/env python3
"""
PHASE 5: ABLATION STUDY
=============================================================
Systematically remove or modify components and observe the impact on performance.
Base Model: LightGBM (fastest training time, strong internal F1)

Experiments:
  A0: Baseline (Temporal Split, All Features, Balanced)
  A1: Remove Top-3 Features (Dst Port, Flow IAT Max, Idle Max)
  A2: Remove Class Balancing (scale_pos_weight=1)
  A3: Use only Top-10 Features
  A4: Random Split (Leakage simulation) instead of Temporal Split
  A5: Scramble Top Feature (Dst Port)
"""

import pandas as pd
import numpy as np
import os
import time
import pickle
import warnings
from sklearn.metrics import classification_report, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
import lightgbm as lgb

warnings.filterwarnings('ignore')

# === PATHS ===
MODEL_DIR = "/home/ryzen/.gemini/antigravity/scratch/nids_infiltration/models"
print("=" * 70)
print("PHASE 5: ABLATION STUDY")
print("=" * 70)

# ============================================================
# STEP 1: Load Base Data and Artifacts
# ============================================================
print("\n[1/3] Loading data and artifacts...")
X_train = pd.read_parquet(os.path.join(MODEL_DIR, "X_train.parquet"))
X_val   = pd.read_parquet(os.path.join(MODEL_DIR, "X_val.parquet"))
X_test  = pd.read_parquet(os.path.join(MODEL_DIR, "X_test.parquet"))
y_train = pd.read_parquet(os.path.join(MODEL_DIR, "y_train.parquet"))['label']
y_val   = pd.read_parquet(os.path.join(MODEL_DIR, "y_val.parquet"))['label']
y_test  = pd.read_parquet(os.path.join(MODEL_DIR, "y_test.parquet"))['label']

df_2018 = pd.read_parquet(os.path.join(MODEL_DIR, "df_2018_cleaned.parquet"))

rf_importances = pd.read_csv(os.path.join(MODEL_DIR, "feature_importances_rf.csv"), index_col=0)
top_features = list(rf_importances.index)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

def train_eval_lgbm(X_tr, y_tr, X_v, y_v, X_te, y_te, sp_weight, name):
    print(f"\n  Running: {name}...")
    t0 = time.time()
    model = lgb.LGBMClassifier(
        n_estimators=100,  # Reduced for speed in ablation
        num_leaves=31,
        learning_rate=0.1,
        scale_pos_weight=sp_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_v, y_v)],
        callbacks=[lgb.early_stopping(15, verbose=False), lgb.log_evaluation(0)]
    )
    y_proba = model.predict_proba(X_te)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    
    precision_arr, recall_arr, _ = precision_recall_curve(y_te, y_proba)
    pr_auc = auc(recall_arr, precision_arr)
    
    # Calculate F1 manually for Infiltration class
    tp = ((y_pred == 1) & (y_te == 1)).sum()
    fp = ((y_pred == 1) & (y_te == 0)).sum()
    fn = ((y_pred == 0) & (y_te == 1)).sum()
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    
    elapsed = time.time() - t0
    print(f"    ✓ Done in {elapsed:.1f}s | F1: {f1:.4f} | PR-AUC: {pr_auc:.4f}")
    
    return {'Experiment': name, 'F1-Score': f1, 'PR-AUC': pr_auc}

results = []

# ============================================================
# A0: Baseline (Temporal)
# ============================================================
res = train_eval_lgbm(X_train, y_train, X_val, y_val, X_test, y_test, scale_pos_weight, "A0: Baseline (Temporal, All Features)")
results.append(res)

# ============================================================
# A1: Remove Top-3 Features
# ============================================================
drop_cols = top_features[:3]
print(f"    (Dropping: {drop_cols})")
res = train_eval_lgbm(
    X_train.drop(columns=drop_cols), y_train,
    X_val.drop(columns=drop_cols), y_val,
    X_test.drop(columns=drop_cols), y_test,
    scale_pos_weight, "A1: Remove Top-3 Features"
)
results.append(res)

# ============================================================
# A2: Remove Class Balancing
# ============================================================
res = train_eval_lgbm(X_train, y_train, X_val, y_val, X_test, y_test, 1.0, "A2: Remove Class Balancing")
results.append(res)

# ============================================================
# A3: Top-10 Features Only
# ============================================================
keep_cols = top_features[:10]
res = train_eval_lgbm(
    X_train[keep_cols], y_train,
    X_val[keep_cols], y_val,
    X_test[keep_cols], y_test,
    scale_pos_weight, "A3: Top-10 Features Only"
)
results.append(res)

# ============================================================
# A4: Random Split (Leakage Simulation)
# ============================================================
print("\n[2/3] Simulating Leakage with Random Split...")
X_all = df_2018.drop(columns=['Label', 'Timestamp', 'Protocol'], errors='ignore')

# Handle object columns and NaNs just like Phase 2
for col in X_all.columns:
    if X_all[col].dtype == object:
        X_all[col] = pd.to_numeric(X_all[col], errors='coerce')
X_all.replace([np.inf, -np.inf], np.nan, inplace=True)
X_all.fillna(X_all.median(), inplace=True)

y_all = (df_2018['Label'] == 'Infiltration').astype(int)

# 70/10/20 random split
X_tr_r, X_tmp, y_tr_r, y_tmp = train_test_split(X_all, y_all, test_size=0.3, random_state=42)
X_v_r, X_te_r, y_v_r, y_te_r = train_test_split(X_tmp, y_tmp, test_size=0.6666, random_state=42) # 10% / 20%

res = train_eval_lgbm(X_tr_r, y_tr_r, X_v_r, y_v_r, X_te_r, y_te_r, scale_pos_weight, "A4: Random Split (Leakage)")
results.append(res)

# ============================================================
# A5: Scramble Top Feature (Dst Port)
# ============================================================
X_test_scrambled = X_test.copy()
# Permute the values of Dst Port randomly to destroy its predictive power
np.random.seed(42)
X_test_scrambled['Dst Port'] = np.random.permutation(X_test_scrambled['Dst Port'].values)

res = train_eval_lgbm(X_train, y_train, X_val, y_val, X_test_scrambled, y_test, scale_pos_weight, "A5: Scramble Dst Port at Inference")
results.append(res)


# ============================================================
# SUMMARY
# ============================================================
print("\n[3/3] Ablation Results Summary")
print("=" * 70)
df_res = pd.DataFrame(results)

print(f"""
| Experiment | F1-Score | PR-AUC |
|------------|----------|--------|
| {df_res.iloc[0]['Experiment']} | {df_res.iloc[0]['F1-Score']:.4f} | {df_res.iloc[0]['PR-AUC']:.4f} |
| {df_res.iloc[1]['Experiment']} | {df_res.iloc[1]['F1-Score']:.4f} | {df_res.iloc[1]['PR-AUC']:.4f} |
| {df_res.iloc[2]['Experiment']} | {df_res.iloc[2]['F1-Score']:.4f} | {df_res.iloc[2]['PR-AUC']:.4f} |
| {df_res.iloc[3]['Experiment']} | {df_res.iloc[3]['F1-Score']:.4f} | {df_res.iloc[3]['PR-AUC']:.4f} |
| {df_res.iloc[4]['Experiment']} | {df_res.iloc[4]['F1-Score']:.4f} | {df_res.iloc[4]['PR-AUC']:.4f} |
| {df_res.iloc[5]['Experiment']} | {df_res.iloc[5]['F1-Score']:.4f} | {df_res.iloc[5]['PR-AUC']:.4f} |
""")

print("\nAblation Discussion Points (for final report):")
print("- Random Split (A4) typically shows inflated, unrealistic performance because temporal relationships (the natural progression of an attack) are destroyed, causing train/test leakage.")
print("- Removing class balancing (A2) usually drops Recall dramatically, lowering the overall F1-score on imbalanced datasets.")
print("- Scrambling the top feature (A5) demonstrates model dependence on port numbers, highlighting potential brittleness if attackers change ports.")

pd.DataFrame(results).to_csv(os.path.join(MODEL_DIR, "phase5_ablation.csv"), index=False)
print("\n[STATE: COMPLETED Phase 5 — Ablation Study Finished]")
