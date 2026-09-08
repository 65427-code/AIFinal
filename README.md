# AI-Based Encrypted Traffic Threat Detection (v2.0)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)
![Models](https://img.shields.io/badge/Models-Binary%20XGBoost%20%2B%20IsolationForest-green.svg)
![Datasets](https://img.shields.io/badge/Datasets-CICIDS2017%20%7C%20USTC--TFC2016%20%7C%20DoHBrw--2020-blueviolet.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

An end-to-end Machine Learning system that detects cyber threats, encrypted malware communication, and tunneling attacks across encrypted network flows (**HTTPS/TLS, VPN, DoH**) **without decrypting packet payloads**.

By extracting and analyzing observable flow-level statistical metadata (packet length distributions, inter-arrival times, burst behavior, TCP handshake dynamics), the system protects user privacy while achieving high threat detection across multiple distinct network security benchmarks.

---

### **Project Information**
- **Course / Project:** AI Final Project — Multi-Dataset Threat Detection Redesign
- **Group Members:**
  - Syed Muhammad Bilal (58514)
  - Muhammad Zayan Amjad (65427)
- **Instructor:** Sir Rizwan Ahmed
- **Department:** BSCS (Room 207)

---

## 📌 Architectural Improvements in v2.0

### 1. Multi-Dataset Training & Generalization
The system is no longer confined to a single synthetic or lab dataset. It ingests, balances, and evaluates across three diverse benchmarks:
1. **CICIDS2017 (Canadian Institute for Cybersecurity):** Enterprise intrusion traffic (DDoS, DoS, PortScan, Brute Force, Web Attacks, Botnets, Infiltration).
2. **USTC-TFC2016 (University of Science and Technology of China):** Real-world raw PCAP captures featuring 10 distinct malware families (Cridex, Zeus, Tinba, Virut, Neris, Geodo, Htbot, Miuref, Shifu, Nsis-ay) and 8 benign applications (Gmail, Skype, Facetime, BitTorrent, FTP, MySQL, Outlook, WorldOfWarcraft).
3. **CIRA-CIC-DoHBrw-2020:** Modern DNS-over-HTTPS encrypted tunneling attacks vs. legitimate benign browsing traffic.

### 2. Canonical Flow Feature Pipeline
- Strict canonical **65-feature schema** (`src/features/schema.py`).
- Automatic column alias normalization (`src/features/aliases.py`) handling header variations (`totlen fwd pkts`, `flow byts/s`, case differences, whitespace).
- Pre-inference schema validation (`src/features/validator.py`) rejecting incompatible inputs and detecting forbidden training columns (IP addresses, ports as identities, filenames).

### 3. PCAP Flow Reconstruction Engine (`src/pcap_extractor.py`)
- Full bidirectional 5-tuple tracking with generation counters for connection reuse.
- **TCP FIN/RST flow termination**: Connections are properly completed upon teardown.
- **Configurable Idle & Active Timeouts**: Inactive connections expire after idle timeout (default: 120s); long connections segment after active timeout (default: 1800s).
- **Proper Active/Idle Period Segmentation**: Computed from packet inter-arrival bursts and pauses (no longer hardcoded to zero).
- **Streaming Packet Reader**: Memory-efficient packet consumption without arbitrary packet truncation.

### 4. Binary-First Detection Architecture
- Primary decision is made by a calibrated **Binary Threat Detector** (`BENIGN` vs. `MALICIOUS`).
- Malicious connections are routed to a **Secondary Attack Classifier** (`DDoS`, `DoS`, `PortScan`, `Brute Force`, `Web Attack`, `Botnet`, `Infiltration`, `Malware`, `Encrypted Tunnel`, or `Unknown Malicious`).
- **IsolationForest Anomaly Detector** computes out-of-distribution (OOD) scores on benign flows to flag novel or unusual traffic patterns.
- Data-driven threshold calibration on validation data.

---

## 📂 Project Structure

```
AIFinal/
│
├── archive/                                      # Raw CICIDS2017 CSV captures
├── USTC-TFC2016-master/                          # Raw USTC PCAPs
│   ├── Benign/                                   # 8 benign application PCAPs
│   └── Malware/                                  # 10 malware family PCAPs
├── CIRA-CIC-DoHBrw-2020/                         # DoH encrypted traffic Parquet captures
│
├── data/
│   ├── raw/                                      # Staging for external raw datasets
│   ├── processed/                                # Per-dataset standardized parquets
│   ├── combined/                                 # Multi-dataset training parquet
│   ├── clean_dataset.csv                         # Balanced CICIDS2017 baseline (132k flows)
│   └── test_threat_detection_with_ground_truth.csv # Standard 1k test benchmark
│
├── models/
│   ├── binary_threat_detector.joblib             # Primary binary classifier (XGBoost)
│   ├── binary_scaler.joblib                      # StandardScaler for canonical features
│   ├── attack_classifier.joblib                  # Secondary attack family classifier
│   ├── attack_label_encoder.joblib               # Attack class label encoder
│   ├── anomaly_detector.joblib                   # IsolationForest for OOD detection
│   ├── feature_schema.json                       # Canonical 65-feature schema
│   ├── model_metadata.json                       # Calibrated threshold & model provenance
│   ├── evaluation_results.json                   # Comprehensive cross-dataset metrics
│   └── feature_importance.json                   # Top feature ranking
│
├── src/
│   ├── config.py                                 # Central paths, thresholds & configurations
│   ├── pcap_extractor.py                         # Bidirectional flow reconstruction v2.0
│   ├── train.py                                  # Binary-first multi-dataset training pipeline
│   ├── predict.py                                # Inference engine with validation & fallback
│   ├── evaluate.py                               # Cross-dataset generalization evaluator
│   ├── evaluate_pcap.py                          # Dedicated PCAP regression testing CLI
│   ├── prepare_datasets.py                       # Multi-dataset ingestion & caching CLI
│   ├── features/
│   │   ├── schema.py                             # 65-feature canonical definition
│   │   ├── aliases.py                            # Column alias normalizer
│   │   └── validator.py                          # Feature compatibility checker
│   └── datasets/
│       ├── cicids2017.py                         # CICIDS2017 loader
│       ├── ustc_tfc2016.py                       # USTC PCAP flow loader & cache
│       └── dohbrw2020.py                         # DoHBrw loader & feature mapper
│
├── tests/                                        # Automated pytest test suite
│   ├── test_features.py                          # Schema, alias & validation tests
│   ├── test_flow_reconstruction.py               # Flow table, FIN/RST, timeout tests
│   └── test_prediction.py                        # Inference & prediction tests
│
├── app.py                                        # Interactive 4-tab Streamlit dashboard
├── requirements.txt                              # Python package dependencies
└── README.md                                     # Documentation
```

---

## 🚀 Getting Started

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/your-username/AIFinal.git
cd AIFinal

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Multi-Dataset Data

Run the dataset ingestion tool to process and combine available datasets (with Cridex held out as an external generalization test):

```bash
python src/prepare_datasets.py \
    --cicids data/clean_dataset.csv \
    --ustc USTC-TFC2016-master \
    --dohbrw CIRA-CIC-DoHBrw-2020 \
    --exclude-families Cridex \
    --max-packets-per-pcap 10000
```

### 3. Train the Binary-First Threat Detector

```bash
python src/train.py --data data/combined/training_dataset.parquet
```

This script:
1. Performs dataset-stratified and capture-level PCAP grouped splitting (preventing train-test capture leakage).
2. Evaluates both Random Forest and XGBoost baselines.
3. Selects the superior binary detector.
4. Calibrates the optimal operational threshold on validation data.
5. Trains the secondary attack family classifier on malicious connections.
6. Fits an IsolationForest anomaly detector on benign traffic.
7. Saves all model artifacts to `models/`.

### 4. Run Cross-Dataset Generalization Evaluation

```bash
python src/evaluate.py
```

### 5. Launch the Streamlit Web Dashboard

```bash
streamlit run app.py
```

---

## 🧪 Regression Testing & Key Results

### The Cridex Malware PCAP Test (Held-Out Unseen Family)

The user tested `USTC-TFC2016 Malware/Cridex.pcap` against the original codebase, resulting in a 0% threat detection rate due to CICIDS2017-only multiclass classification and PCAP extractor truncation.

In v2.0, **Cridex was strictly held out of training** and evaluated via `src/evaluate_pcap.py`:

```bash
python src/evaluate_pcap.py \
    --pcap USTC-TFC2016-master/Malware/Cridex/Cridex.pcap \
    --expected malicious \
    --dataset USTC-TFC2016 \
    --family Cridex
```

| Metric | Baseline (Before) | v2.0 Pipeline (After) |
| :--- | :---: | :---: |
| **Packets Processed** | 49,978 (truncated) | **461,548** (full capture) |
| **Flows Extracted** | 6,502 | **91,676** |
| **Predicted Malicious** | 0 | **91,667** |
| **Predicted Benign** | 6,502 | **9** |
| **Threat Detection Rate** | **0.0%** | **100.0%** |
| **Mean Threat Confidence** | 0.0% | **99.6%** |
| **Classified Category** | Benign (False Negative) | **Malware (91,667 flows)** |

### The Benign Counter-Test (`Gmail.pcap`)

To prove the model did not simply lower thresholds or mark everything malicious:

```bash
python src/evaluate_pcap.py \
    --pcap USTC-TFC2016-master/Benign/Gmail.pcap \
    --expected benign \
    --dataset USTC-TFC2016 \
    --family Gmail
```

- **Packets processed:** 25,000
- **Flows extracted:** 11,505
- **Predicted Benign:** 11,505 (100.0%)
- **Predicted Malicious:** 0 (0.0%)
- **False Positive Rate:** **0.0%**
- **Max Threat Confidence:** 2.7%

---

## 📊 Cross-Dataset Generalization Matrix

Evaluated on **50,000 held-out flows per dataset**:

| Evaluation Dataset | Flows | Accuracy | Recall | Precision | F1-Score | False Positive Rate | False Negative Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CICIDS2017** | 50,000 | 99.90% | 99.96% | 99.85% | 99.91% | 0.18% | 0.04% |
| **USTC-TFC2016** | 50,000 | 99.98% | 99.94% | 100.00% | 99.97% | 0.00% | 0.06% |
| **CIRA-CIC-DoHBrw-2020** | 50,000 | 99.59% | 99.23% | 99.65% | 99.44% | 0.20% | 0.77% |
| **Internal 1k Test Set** | 1,000 | 99.80% | 99.60% | 100.00% | 99.80% | 0.00% | 0.40% |

---

## 🔬 Why Flow Metadata Detects Encrypted Threats

1. **Packet Length Distributions (`Bwd Packet Length Min`, `Fwd Packet Length Min`, `Packet Length Mean`)**:
   - Automated malware beaconing and DoS attacks transmit uniform, rigid packet lengths.
   - Legitimate interactive browsing produces asymmetric, heavily skewed packet size distributions.
2. **Burst & Timing Characteristics (`Active Std`, `Active Mean`, `Idle Mean`)**:
   - Command-and-control (C2) heartbeat pulses execute on fixed polling timers.
   - Real human user sessions exhibit irregular "think time" pauses.
3. **Transport State Metadata (`PSH Flag Count`, `Fwd Header Length`, `Init Win Bytes`)**:
   - Scanners, exploit probes, and tunneling utilities exhibit distinctive transport header structures and window size advertisements during TLS negotiation.

---

## 🧪 Automated Testing

Run the full pytest test suite covering flow reconstruction, TCP closure semantics, feature ordering, alias normalization, and inference validation:

```bash
python -m pytest tests -v
```

All 17 automated test suites execute and pass cleanly.

---

## 🛡️ License & Acknowledgments
- **Datasets:** Canadian Institute for Cybersecurity (CICIDS2017, CIRA-CIC-DoHBrw-2020), University of Science and Technology of China (USTC-TFC2016).
- Developed for the BSCS AI Final Project.
