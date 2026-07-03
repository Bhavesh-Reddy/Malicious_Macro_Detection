#!/usr/bin/env python3
"""
PHASE 6: REAL-TIME INFERENCE ADAPTER
=============================================================
Simulate a production deployment environment where network traffic 
flows arrive in real-time (simulated via yielding rows from test set).
Adapter applies the exact preprocessing pipeline and generates inference.

Output:
- Throughput (events per second)
- End-to-end latency per event
- Simulated real-time detection logging
"""

import pandas as pd
import numpy as np
import os
import time
import pickle
import warnings

warnings.filterwarnings('ignore')

# === PATHS ===
MODEL_DIR = "/home/ryzen/.gemini/antigravity/scratch/nids_infiltration/models"
print("=" * 70)
print("PHASE 6: REAL-TIME INFERENCE ADAPTER")
print("=" * 70)

# ============================================================
# ADAPTER CLASS DEFINITION
# ============================================================
class NIDSRealTimeAdapter:
    def __init__(self, model_path, feature_cols_path, medians_path):
        """Initialize the adapter with frozen artifacts from Phase 1-3."""
        print("  [Adapter] Initializing engine and loading artifacts...")
        
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
            
        with open(feature_cols_path, "rb") as f:
            self.feature_columns = pickle.load(f)
            
        with open(medians_path, "rb") as f:
            self.train_medians = pickle.load(f)
            
        print(f"  [Adapter] Ready. Expected feature dimensions: {len(self.feature_columns)}")
        
    def preprocess_event(self, event_dict):
        """Apply Phase 1 & 2 cleaning dynamically to a single event."""
        # 1. Convert to DataFrame (single row)
        df_event = pd.DataFrame([event_dict])
        
        # 2. Extract only expected features (aligning schema)
        X_event = pd.DataFrame(index=[0])
        for col in self.feature_columns:
            if col in df_event.columns:
                val = df_event.at[0, col]
                # Coerce to numeric
                try:
                    X_event.at[0, col] = float(val)
                except (ValueError, TypeError):
                    X_event.at[0, col] = np.nan
            else:
                X_event.at[0, col] = np.nan
                
        # 3. Impute NaNs and Infs with training medians (leakage-free)
        X_event.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Using a dictionary for fillna is faster for single rows
        X_event.fillna(self.train_medians, inplace=True)
        
        return X_event
        
    def predict(self, event_dict):
        """End-to-end prediction for a single event."""
        t0 = time.time()
        
        # Preprocess
        X_clean = self.preprocess_event(event_dict)
        
        # Inference
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X_clean)[0, 1]
        else:
            proba = self.model.predict(X_clean)[0]
            
        is_attack = int(proba >= 0.5)
        latency = time.time() - t0
        
        return is_attack, proba, latency

# ============================================================
# SIMULATION ENGINE
# ============================================================
def simulate_stream(n_events=1000):
    """Simulate a network stream using the Phase 2 test set."""
    # We use the best model: Random Forest based on PR-AUC
    model_path = os.path.join(MODEL_DIR, "model_rf.pkl")
    features_path = os.path.join(MODEL_DIR, "feature_columns.pkl")
    medians_path = os.path.join(MODEL_DIR, "train_medians.pkl")
    
    adapter = NIDSRealTimeAdapter(model_path, features_path, medians_path)
    
    print("\n  [Stream] Loading test data to simulate live traffic...")
    # Load test set (as dictionary records)
    X_test = pd.read_parquet(os.path.join(MODEL_DIR, "X_test.parquet"))
    y_test = pd.read_parquet(os.path.join(MODEL_DIR, "y_test.parquet"))['label']
    
    # We will just take the first `n_events`
    X_sample = X_test.iloc[:n_events]
    y_sample = y_test.iloc[:n_events]
    
    events = X_sample.to_dict('records')
    actuals = y_sample.values
    
    print(f"  [Stream] Commencing real-time simulation for {n_events} events...\n")
    
    total_latency = 0
    detections = 0
    false_positives = 0
    
    t_start = time.time()
    
    # Process events one by one to simulate true streaming
    for i, event in enumerate(events):
        actual_label = actuals[i]
        
        is_attack, proba, latency = adapter.predict(event)
        total_latency += latency
        
        # Print alerts for detected attacks
        if is_attack:
            detections += 1
            if actual_label == 0:
                false_positives += 1
                status = "FALSE ALARM"
            else:
                status = "TRUE POSITIVE"
            
            # Print only first 5 alerts to avoid console spam
            if detections <= 5:
                print(f"    [ALERT] Infiltration detected! (Confidence: {proba*100:.1f}%) | Latency: {latency*1000:.2f}ms | Status: {status}")
                
    total_time = time.time() - t_start
    
    print("\n" + "=" * 70)
    print("PHASE 6 OUTPUT: Real-Time Adapter Performance metrics")
    print("=" * 70)
    
    throughput = n_events / total_time
    avg_latency = (total_latency / n_events) * 1000 # ms
    
    print(f"  Total Events Processed: {n_events:,}")
    print(f"  Total Stream Time:      {total_time:.4f} seconds")
    print(f"  System Throughput:      {throughput:,.1f} events/sec")
    print(f"  Average E2E Latency:    {avg_latency:.4f} ms/event")
    print(f"  Alerts Generated:       {detections}")
    if detections > 0:
        print(f"  False Positive Rate:    {false_positives/detections*100:.1f}% of alerts")
        
    print("\n[STATE: COMPLETED Phase 6 — Pipeline entirely finished]")

if __name__ == "__main__":
    # Simulate with 5,000 events to get stable throughput metrics
    simulate_stream(n_events=5000)
