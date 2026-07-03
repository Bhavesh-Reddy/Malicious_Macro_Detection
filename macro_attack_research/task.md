# NIDS Infiltration Detection — Task Tracker

## Pre-requisites
- [x] Create project directory structure
- [x] Install LightGBM & XGBoost
- [x] Download 2018 infiltration CSVs from AWS S3
- [x] Download 2017 infiltration CSV

## Phase 1: Data Cleaning, Sanitization & Structural Audit
- [x] Strip column headers
- [x] Scan for NaN/Inf anomalies
- [x] Normalize timestamps to ISO 8601
- [x] Binarize labels
- [x] Print Phase 1 metrics table

## Phase 2: Chronological Partitioning & Feature Discovery
- [x] Drop fingerprinting columns
- [x] Chronological sort & 70/10/20 split
- [x] Fit baseline Random Forest for feature importance
- [x] Print Top 10 features

## Phase 3: Model Optimization & Internal Performance
- [x] Train 3 models (RF baseline + LightGBM + XGBoost)
- [x] Evaluate on temporal test set
- [x] Print confusion matrix, classification report, PR-AUC, ROC-AUC per model

## Phase 4: OOD Zero-Trust Generalization
- [x] Align 2017 features to 2018 schema
- [x] Predict on 2017 with frozen model
- [x] Print cross-dataset metrics & analysis

## Phase 5: Ablation Study
- [x] Experiment A1: Remove top-3 features
- [x] Experiment A2: Remove class balancing
- [x] Experiment A3: Feature subset (Top-10 vs All)
- [x] Experiment A4: Random vs Temporal split
- [x] Experiment A5: Scramble top feature (Dst Port)
- [x] Print comparative ablation metrics

## Phase 6: Real-Time Inference Adapter
- [x] Design real-time streaming class
- [x] Implement preprocessing pipeline matching Phase 1 & 2
- [x] Simulate real-time stream and calculate throughput

## Phase 7: Continuous Online Retraining
- [x] Develop incremental learning loop on 2017 OOD dataset
- [x] Evaluate performance drift chunk-by-chunk
- [x] Print metrics showing adaptation over time
