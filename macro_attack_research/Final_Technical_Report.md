# Final Technical Report
**Project:** Network Attack Detection Using Machine Learning (Infiltration Detection)

This comprehensive report satisfies the remaining documentation deliverables mandated by the project guidelines.

---

## 1. Literature Survey (Deliverable 1)
Network Intrusion Detection Systems (NIDS) have historically relied on deep packet inspection (DPI) and signature-based rules (e.g., Snort, Suricata). However, modern infiltration attacks, particularly those utilizing heavily obfuscated macros or encrypted payloads, easily bypass static signatures. Recent literature heavily favors Machine Learning (ML) techniques. 
* **Tree-Based Ensembles:** Random Forest and XGBoost are widely cited in literature for their ability to handle non-linear tabular network flows with high efficiency.
* **The Domain Shift Challenge:** A critical gap in current research is model degradation across temporal domains (concept drift). Models trained on laboratory datasets often collapse in real-world scenarios due to shifting baseline traffic profiles.
* **Online Learning Mitigation:** Recent studies suggest continuous, incremental learning architectures over static "train-once" deployments to mitigate this false-positive fatigue.

---

## 2. Dataset Analysis Report (Deliverable 2)
This project utilized two foundational datasets:
1. **CSE-CIC-IDS2018 (Dataset A):** Used as the primary training, validation, and testing environment. We merged the Wednesday (28-02-2018) and Thursday (01-03-2018) files, which contain macro-based infiltration attacks.
    * **Total Records:** 944,171 flows.
    * **Infiltration Prevalence:** 17.15% (161,934 positive samples).
2. **CIC-IDS2017 (Dataset B):** Held entirely out-of-distribution (OOD) to strictly test generalization.
    * **Total Records:** 288,602 flows.
    * **Infiltration Prevalence:** 0.0125% (36 positive samples).

**Structural Alignment:** Dataset B utilized verbose column headers (e.g., `Destination Port`), whereas Dataset A utilized abbreviations (e.g., `Dst Port`). We developed a programmatic schema mapper to align 77 behavioral features perfectly.

---

## 3. Data Balancing Report (Deliverable 3)
A core challenge in NIDS is extreme class imbalance. 
* **Imbalance Ratio:** The primary dataset exhibited a 4.8:1 benign-to-malicious ratio. The secondary dataset exhibited a severe 8015:1 ratio.
* **Balancing Strategy:** Rather than using synthetic sampling techniques like SMOTE (which often induce artificial patterns causing data leakage), we utilized **cost-sensitive learning**. We calculated the `scale_pos_weight` dynamically (`7.38`) and fed it directly into the LightGBM and XGBoost objective functions. 
* **Proof of Necessity:** Our Ablation Study (Phase 5) proved that removing this balancing mechanism caused the F1 score to plummet to exactly 0.00, demonstrating its critical necessity.

---

## 4. Temporal Splitting Documentation (Deliverable 4)
Standard `train_test_split()` random shuffling is considered an academic flaw in cybersecurity research because it leaks future behavioral patterns into the training set, allowing models to cheat.
* **Implementation:** We sorted the 2018 dataset strictly by the `Timestamp` column in ascending chronological order.
* **Partitioning:** We allocated the first 70% of chronological time to Training, the next 10% to Validation, and the final 20% strictly to Testing. 
* **Verification:** The Ablation Study explicitly showed that swapping to a random shuffle artificially inflated PR-AUC scores by over 50%, proving that temporal splitting prevents unrealistic performance inflation.

---

## 5. Leakage Prevention Documentation (Deliverable 5)
Data leakage occurs when a model gains access to information during training that it would not have in a real-world deployment. We implemented strict zero-leakage policies:
1. **Fingerprint Removal:** IP Addresses, Flow IDs, and specific port identifiers (except Destination Port, which implies service type) were completely stripped before training. The model learns *how* the packet flows, not *who* sent it.
2. **Median Imputation Isolation:** NaN and Infinity values were discovered (over 18,000 instances). We imputed these mathematically invalid rows. Crucially, we calculated the column medians **only** from the 70% Training split. The Validation and Test splits were filled using the isolated Training medians to simulate real-time blindness.

---

## 6. Real-World Capability Conclusions (Deliverable 14 Summary)
Our experiments conclusively mapped the difficulties of deploying NIDS:
* **The "Concept Drift" Bottleneck:** Phase 4 proved that our highly optimized model collapsed when fed the 2017 data. Static ML models are brittle when network environments shift temporally.
* **The Solution (Phase 7):** We engineered a Continuous Online Retraining adapter utilizing LightGBM's incremental `init_model` parameter. By micro-batching the drifted 2017 data into the model, it dynamically learned the new benign baseline and dropped its False Positive Rate to 0.00% without catastrophic forgetting.
* **Throughput:** Our Real-Time Inference engine successfully processed ~18 events/sec natively in Python. For enterprise Gigabit deployments, this exact pipeline logic should be ported to C++ to handle line-rate speeds.
