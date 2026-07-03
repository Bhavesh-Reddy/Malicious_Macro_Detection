#!/usr/bin/env python3
"""
PHASE 3: MODEL OPTIMIZATION & INTERNAL PERFORMANCE EVALUATION
==============================================================
Train 3 models on the temporal training split, tune on validation, evaluate on test:
  Model 1: Random Forest (baseline)
  Model 2: LightGBM (gradient boosting)
  Model 3: XGBoost (regularized gradient boosting)

Metrics per model:
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1)
  - PR-AUC, ROC-AUC
  - FPR, FNR
  - Training Time, Inference Time
"""

import pandas as pd
import numpy as np
import os
import time
import pickle
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_curve, auc, roc_auc_score,
    average_precision_score
)
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

# === PATHS ===
MODEL_DIR = "/home/ryzen/.gemini/antigravity/scratch/nids_infiltration/models"

# ============================================================
# STEP 1: Load Phase 2 artifacts
# ============================================================
print("=" * 70)
print("PHASE 3: MODEL OPTIMIZATION & INTERNAL PERFORMANCE EVALUATION")
print("=" * 70)

print("\n[1/4] Loading Phase 2 splits...")
X_train = pd.read_parquet(os.path.join(MODEL_DIR, "X_train.parquet"))
X_val   = pd.read_parquet(os.path.join(MODEL_DIR, "X_val.parquet"))
X_test  = pd.read_parquet(os.path.join(MODEL_DIR, "X_test.parquet"))
y_train = pd.read_parquet(os.path.join(MODEL_DIR, "y_train.parquet"))['label']
y_val   = pd.read_parquet(os.path.join(MODEL_DIR, "y_val.parquet"))['label']
y_test  = pd.read_parquet(os.path.join(MODEL_DIR, "y_test.parquet"))['label']

print(f"  Train: {X_train.shape}, pos={y_train.sum():,} ({y_train.mean()*100:.2f}%)")
print(f"  Val:   {X_val.shape}, pos={y_val.sum():,} ({y_val.mean()*100:.2f}%)")
print(f"  Test:  {X_test.shape}, pos={y_test.sum():,} ({y_test.mean()*100:.2f}%)")

# Class imbalance ratio for scale_pos_weight
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count
print(f"  scale_pos_weight = {scale_pos_weight:.2f}")


# ============================================================
# HELPER: Full evaluation function
# ============================================================
def evaluate_model(model, X_test, y_test, model_name, train_time):
    """Compute all required metrics for a model."""
    print(f"\n{'─' * 60}")
    print(f"  {model_name} — Evaluation on Temporal Test Set")
    print(f"{'─' * 60}")
    
    # Inference time
    t0 = time.time()
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.predict(X_test)
    y_pred = (y_proba >= 0.5).astype(int)
    inference_time = time.time() - t0
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted Benign  Predicted Infiltration")
    print(f"    Actual Benign       {tn:>10,}          {fp:>10,}")
    print(f"    Actual Infiltr.     {fn:>10,}          {tp:>10,}")
    
    # Classification Report
    print(f"\n  Classification Report:")
    report = classification_report(y_test, y_pred, 
                                   target_names=['Benign', 'Infiltration'],
                                   digits=4)
    for line in report.split('\n'):
        print(f"    {line}")
    
    # PR-AUC
    precision_arr, recall_arr, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_arr, precision_arr)
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # FPR, FNR
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    print(f"\n  PR-AUC:          {pr_auc:.6f}")
    print(f"  ROC-AUC:         {roc_auc:.6f}")
    print(f"  FPR:             {fpr:.6f}")
    print(f"  FNR:             {fnr:.6f}")
    print(f"  Training Time:   {train_time:.2f}s")
    print(f"  Inference Time:  {inference_time:.4f}s")
    
    return {
        'model_name': model_name,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'fnr': fnr,
        'train_time': train_time,
        'inference_time': inference_time,
        'precision_benign': cm[0,0] / (cm[0,0] + cm[1,0]) if (cm[0,0] + cm[1,0]) > 0 else 0,
        'recall_benign': cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0,
        'precision_infil': cm[1,1] / (cm[0,1] + cm[1,1]) if (cm[0,1] + cm[1,1]) > 0 else 0,
        'recall_infil': cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0,
        'f1_infil': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
    }


# ============================================================
# MODEL 1: Random Forest (Baseline)
# ============================================================
print("\n[2/4] Training Model 1: Random Forest (Baseline)...")
print("  Justification: Non-parametric ensemble, robust to feature scaling,")
print("  serves as interpretable baseline with class_weight='balanced'")

t0 = time.time()
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_train_time = time.time() - t0
print(f"  ✓ Random Forest trained in {rf_train_time:.1f}s")

metrics_rf = evaluate_model(rf, X_test, y_test, "Random Forest (Baseline)", rf_train_time)

# Save model
pd.to_pickle(rf, os.path.join(MODEL_DIR, "model_rf.pkl"))


# ============================================================
# MODEL 2: LightGBM (Advanced)
# ============================================================
print("\n\n[3/4] Training Model 2: LightGBM (Advanced)...")
print("  Justification: Histogram-based gradient boosting, efficient on large datasets,")
print("  native is_unbalance support, leaf-wise tree growth for deeper patterns")

t0 = time.time()
lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    num_leaves=63,
    learning_rate=0.05,
    min_child_samples=50,
    scale_pos_weight=scale_pos_weight,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# Use validation set for early stopping
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
)
lgb_train_time = time.time() - t0
print(f"  ✓ LightGBM trained in {lgb_train_time:.1f}s (best_iteration={lgb_model.best_iteration_})")

metrics_lgb = evaluate_model(lgb_model, X_test, y_test, "LightGBM (Advanced)", lgb_train_time)

# Save model
pd.to_pickle(lgb_model, os.path.join(MODEL_DIR, "model_lgb.pkl"))


# ============================================================
# MODEL 3: XGBoost (Advanced)
# ============================================================
print("\n\n[4/4] Training Model 3: XGBoost (Advanced)...")
print("  Justification: Regularized gradient boosting (L1/L2), strong generalization,")
print("  depth-wise growth complements LightGBM's leaf-wise approach")

t0 = time.time()
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=2.0,
    min_child_weight=10,
    gamma=0.1,
    random_state=42,
    n_jobs=-1,
    eval_metric='aucpr',
    early_stopping_rounds=50,
    verbosity=0
)

# Use validation set for early stopping
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
xgb_train_time = time.time() - t0
print(f"  ✓ XGBoost trained in {xgb_train_time:.1f}s (best_iteration={xgb_model.best_iteration})")

metrics_xgb = evaluate_model(xgb_model, X_test, y_test, "XGBoost (Advanced)", xgb_train_time)

# Save model
pd.to_pickle(xgb_model, os.path.join(MODEL_DIR, "model_xgb.pkl"))


# ============================================================
# COMPARATIVE SUMMARY TABLE
# ============================================================
print("\n\n" + "=" * 70)
print("PHASE 3 OUTPUT: Comparative Model Evaluation Summary")
print("=" * 70)

all_metrics = [metrics_rf, metrics_lgb, metrics_xgb]

print(f"""
| Metric                | Random Forest  | LightGBM       | XGBoost        |
|-----------------------|----------------|----------------|----------------|
| PR-AUC                | {metrics_rf['pr_auc']:.6f}       | {metrics_lgb['pr_auc']:.6f}       | {metrics_xgb['pr_auc']:.6f}       |
| ROC-AUC               | {metrics_rf['roc_auc']:.6f}       | {metrics_lgb['roc_auc']:.6f}       | {metrics_xgb['roc_auc']:.6f}       |
| Infiltration Precision| {metrics_rf['precision_infil']:.6f}       | {metrics_lgb['precision_infil']:.6f}       | {metrics_xgb['precision_infil']:.6f}       |
| Infiltration Recall   | {metrics_rf['recall_infil']:.6f}       | {metrics_lgb['recall_infil']:.6f}       | {metrics_xgb['recall_infil']:.6f}       |
| Infiltration F1       | {metrics_rf['f1_infil']:.6f}       | {metrics_lgb['f1_infil']:.6f}       | {metrics_xgb['f1_infil']:.6f}       |
| FPR                   | {metrics_rf['fpr']:.6f}       | {metrics_lgb['fpr']:.6f}       | {metrics_xgb['fpr']:.6f}       |
| FNR                   | {metrics_rf['fnr']:.6f}       | {metrics_lgb['fnr']:.6f}       | {metrics_xgb['fnr']:.6f}       |
| Training Time (s)     | {metrics_rf['train_time']:.2f}           | {metrics_lgb['train_time']:.2f}           | {metrics_xgb['train_time']:.2f}           |
| Inference Time (s)    | {metrics_rf['inference_time']:.4f}         | {metrics_lgb['inference_time']:.4f}         | {metrics_xgb['inference_time']:.4f}         |
""")

# Identify best model
best = max(all_metrics, key=lambda m: m['pr_auc'])
print(f"  ★ Best model by PR-AUC: {best['model_name']} ({best['pr_auc']:.6f})")

# Save comparative results
pd.DataFrame(all_metrics).to_csv(os.path.join(MODEL_DIR, "phase3_comparison.csv"), index=False)

print(f"\n  All models saved to {MODEL_DIR}")
print("\n[STATE: PAUSE — Awaiting user confirmation of Phase 3 metrics before proceeding]")
