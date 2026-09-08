# Project Context & Architectural Dossier

## 1. Project Overview & Identity

- **Project Title:** AI-Based Encrypted Traffic Threat Detection (System Architecture v2.0)
- **Course:** AI Final Project / Network Security Intelligence
- **Group Members:**
  - Syed Muhammad Bilal (ID: 58514)
  - Muhammad Zayan Amjad (ID: 65427)
- **Instructor:** Sir Rizwan Ahmed
- **Department:** BSCS (Room 207)
- **Core Mission:** Detect malicious cyber threats, malware communication, and covert tunneling across encrypted network traffic (**HTTPS/TLS 1.3, VPN, DoH**) **without inspecting or decrypting packet payloads**, preserving user privacy while guaranteeing high cross-dataset generalization.

---

## 2. Background: The Problem & Baseline Failure Mode

### The Original Failure
When the original system was evaluated against real-world external malware traffic (`USTC-TFC2016 Malware/Cridex.pcap`), it produced the following outcome:
- **Packets processed:** 49,978 (silently cut off by hardcoded cap)
- **Flows extracted:** 6,502
- **Threats detected:** 0
- **Benign flows:** 6,502
- **Observed Threat Rate:** **0.0%** (100% False Negative Rate)

### Root Cause Analysis
Thorough repository auditing uncovered seven fundamental architectural defects:
1. **Multiclass Lab Dataset Lock-In:** The classifier was trained strictly on CICIDS2017 using an 8-class multiclass model (`BENIGN`, `DDoS`, `PortScan`, `DoS`, etc.). Malware outside these specific attack labels was assigned to the majority class (`BENIGN`).
2. **Hardcoded Silent Packet Cap:** `src/pcap_extractor.py` silently halted packet extraction at 50,000 packets (`max_packets = 50000`), discarding over 400,000 packets from larger captures without user warning.
3. **Zeroed Burst & Timing Features:** `Active Mean`, `Active Std`, `Active Max`, and `Active Min` were hardcoded to return `0.0` for all extracted flows, discarding critical command-and-control (C2) heartbeat and beaconing dynamics.
4. **Lack of Flow Termination & Timeout Mechanics:** The extractor lacked TCP FIN/RST teardown handling and idle timeout expiration, causing multiple successive connections sharing a 5-tuple to merge into unnatural mega-flows.
5. **Schema Masking via Zero-Filling:** Batch inference silently imputed missing features with `0.0` instead of validating column compatibility, disguising severe schema mismatches.
6. **No Encrypted Tunneling Representation:** The model lacked training on DNS-over-HTTPS (DoH) encrypted tunneling data.
7. **Identity Leakage Risks:** The pipeline did not enforce strict exclusion of non-generalizing identity features (`Source_IP`, `Destination_IP`, port numbers as identities).

---

## 3. Redesigned Architecture (v2.0)

### A. Binary-First Detection Hierarchy
The architecture decouples the security decision from attack categorization:
1. **Primary Binary Detector (XGBoost / Random Forest):**
   - Focuses exclusively on answering: `Is this flow BENIGN or MALICIOUS?`
   - Decision threshold is dynamically calibrated on validation data (`0.5996`).
2. **Secondary Attack Classifier (XGBoost Multi-Class):**
   - Invoked only for flows flagged as malicious.
   - Categorizes threats into: `Botnet`, `Brute Force`, `DDoS`, `DoS`, `Encrypted Tunnel`, `Infiltration`, `Malware`, `PortScan`, `Web Attack`, or falls back to `Unknown Malicious`.
3. **Unsupervised Anomaly Detector (IsolationForest):**
   - Trained on legitimate benign flows to generate Out-Of-Distribution (OOD) scores.
   - Flags suspicious or anomalous connections that deviate from known benign distributions.

### B. Canonical 65-Feature Schema & Validation Pipeline
- **Canonical Schema (`src/features/schema.py`):** Strictly enforces 65 CICFlowMeter-compatible statistical features. Zero-variance columns (`Fwd Header Length.1`, bulk metrics) are excluded.
- **Alias Normalization (`src/features/aliases.py`):** Maps ~120 column variations (e.g., `totlen fwd pkts`, `flow byts/s`, casing, whitespace) to canonical definitions.
- **Feature Validator (`src/features/validator.py`):** Checks schema compatibility before inference and flags forbidden training columns.

### C. PCAP Flow Reconstruction Engine (`src/pcap_extractor.py`)
- **Bidirectional 5-Tuple Keying:** Uses canonical sorted endpoint keys `(IP, Port)` to pair forward and reverse packets.
- **Generation Counters:** Increments flow generations to support 5-tuple reuse.
- **TCP FIN/RST Teardown:** Closes flows immediately upon connection termination.
- **Configurable Timeouts:** Enforces idle timeout (120s) and active timeout (1800s).
- **True Active/Idle Segmentation:** Accurately calculates burst and idle statistics from inter-packet arrival times.
- **Streaming Ingestion:** Processes full captures without arbitrary packet limits.

---

## 4. Multi-Dataset Integration & Data Pipeline

The system is trained on **252,724 standardized flows** drawn across three complementary benchmarks:

| Dataset | Type | Attack Categories Covered | Standardized Rows |
| :--- | :--- | :--- | :---: |
| **CICIDS2017** | Flow CSVs | DDoS, DoS, PortScan, Brute Force, Web Attacks, Botnet, Infiltration | 132,004 |
| **USTC-TFC2016** | Raw PCAP | 10 Malware Families (Cridex, Zeus, Tinba, Virut, Neris, Geodo, etc.) & 8 Benign Apps | 51,597 |
| **CIRA-CIC-DoHBrw-2020** | Parquet | Encrypted DNS-over-HTTPS Tunneling vs. Benign Web Browsing | 69,123 |

### Data Leakage Prevention & Capture-Level Splitting
To adhere to sound scientific methodology:
- **PCAP Capture Isolation:** Captures are grouped by `Capture_ID`. Individual PCAPs are assigned whole to train or test partitions (e.g., `Gmail.pcap` and `Skype.pcap` in test, `BitTorrent.pcap` in train).
- **Held-Out Family Generalization:** `Cridex.pcap` is completely excluded from training to serve as an unseen malware family test.

---

## 5. Empirical Verification & Benchmark Results

### 1. Cridex Malware PCAP Regression (Unseen Family)
```
File:     USTC-TFC2016-master/Malware/Cridex/Cridex.pcap
Expected: MALICIOUS
------------------------------------------------------
Packets processed:    461,548 (Zero truncation)
Flows extracted:      91,676
Predicted MALICIOUS:  91,667  (100.0%)
Predicted BENIGN:     9       (0.0%)
Detection Rate:       100.0%
Mean Confidence:      99.6%
Assigned Category:    Malware
```

### 2. Benign Counter-Test (`Gmail.pcap`)
```
File:     USTC-TFC2016-master/Benign/Gmail.pcap
Expected: BENIGN
------------------------------------------------------
Packets processed:    25,000
Flows extracted:      11,505
Predicted MALICIOUS:  0       (0.0%)
Predicted BENIGN:     11,505  (100.0%)
False Positive Rate:  0.0%
Max Confidence:       2.7%
```

### 3. Cross-Dataset Generalization Matrix (50,000 Unseen Flows / Dataset)

| Benchmark Dataset | Flows Tested | Accuracy | Detection Rate (Recall) | Precision | F1-Score | False Positive Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CICIDS2017** | 50,000 | 99.90% | 99.96% | 99.85% | 99.91% | 0.18% |
| **USTC-TFC2016** | 50,000 | 99.98% | 99.94% | 100.00% | 99.97% | 0.00% |
| **CIRA-CIC-DoHBrw-2020** | 50,000 | 99.59% | 99.23% | 99.65% | 99.44% | 0.20% |
| **Internal 1k Test Set** | 1,000 | 99.80% | 99.60% | 100.00% | 99.80% | 0.00% |

---

## 6. Directory Structure & Key Files

```
AIFinal/
├── app.py                         # Streamlit interactive frontend (4 tabs)
├── requirements.txt               # Dependencies including scapy, xgboost, pyarrow
├── .gitignore                     # Git ignore rules with dataset & artifact protection
├── README.md                      # Primary project documentation
├── context.md                     # This architectural context document
│
├── src/
│   ├── config.py                  # Paths, constants, operational thresholds
│   ├── pcap_extractor.py          # FlowTable, FlowState, 65-feature extractor
│   ├── train.py                   # Multi-dataset binary training & threshold calibration
│   ├── predict.py                 # ThreatPredictor v2.0 inference engine
│   ├── evaluate.py                # Cross-dataset generalization matrix evaluator
│   ├── evaluate_pcap.py           # CLI regression test utility for raw PCAPs
│   ├── prepare_datasets.py        # Dataset ingestion, balancing, and caching CLI
│   ├── features/
│   │   ├── schema.py              # 65 canonical features definition
│   │   ├── aliases.py             # Column name variations & normalizer
│   │   └── validator.py           # Pre-prediction schema validation logic
│   └── datasets/
│       ├── cicids2017.py          # CICIDS CSV loader
│       ├── ustc_tfc2016.py        # USTC PCAP flow loader with parquet cache
│       └── dohbrw2020.py          # DoHBrw parquet loader & feature mapper
│
├── models/
│   ├── binary_threat_detector.joblib # Primary binary XGBoost detector
│   ├── binary_scaler.joblib       # Fitted StandardScaler
│   ├── attack_classifier.joblib   # Secondary multiclass XGBoost model
│   ├── attack_label_encoder.joblib# Label encoder for 9 attack categories
│   ├── anomaly_detector.joblib    # IsolationForest for OOD detection
│   ├── feature_schema.json        # Serialized canonical feature list
│   ├── model_metadata.json        # Calibrated threshold, training date, metrics
│   ├── evaluation_results.json    # Detailed evaluation results
│   └── feature_importance.json    # Ranked feature importances
│
├── tests/
│   ├── conftest.py                # Synthetic packet & flow fixtures
│   ├── test_features.py           # Schema, alias, and validation tests
│   ├── test_flow_reconstruction.py# FlowTable, FIN/RST teardown, timeout tests
│   └── test_prediction.py         # ThreatPredictor inference & batch tests
│
└── data/
    ├── sample_batch.csv           # 32-flow mini-batch demo
    ├── sample_traffic.pcap        # Sample PCAP capture demo
    ├── test_threat_detection_with_ground_truth.csv # 1,000-flow ground-truth benchmark
    └── sample_*.json              # Preset flow profiles (Benign, DDoS, DoS, PortScan, etc.)
```

---

## 7. Command Reference

```bash
# 1. Run Automated Test Suite (17 tests)
python -m pytest tests -v

# 2. Re-prepare Multi-Dataset Data
python src/prepare_datasets.py \
    --cicids data/clean_dataset.csv \
    --ustc USTC-TFC2016-master \
    --dohbrw CIRA-CIC-DoHBrw-2020 \
    --exclude-families Cridex \
    --max-packets-per-pcap 10000

# 3. Retrain Models
python src/train.py --data data/combined/training_dataset.parquet

# 4. Run Cross-Dataset Benchmark Matrix
python src/evaluate.py

# 5. Evaluate Any Raw PCAP
python src/evaluate_pcap.py --pcap USTC-TFC2016-master/Malware/Cridex/Cridex.pcap --expected malicious
python src/evaluate_pcap.py --pcap USTC-TFC2016-master/Benign/Gmail.pcap --expected benign

# 6. Launch Streamlit UI
python -m streamlit run app.py
```
