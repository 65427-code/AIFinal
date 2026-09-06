# AI-Based Encrypted Traffic Threat Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)
![Models](https://img.shields.io/badge/Models-Random%20Forest%20%7C%20XGBoost-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

An end-to-end Machine Learning pipeline that detects cyber threats and malware communication inside encrypted network flows (**HTTPS/TLS, VPN**) **without decrypting packet payloads**. By extracting and analyzing flow-level statistical patterns and metadata, the system protects user privacy while achieving over **99.9% threat detection accuracy**.

---

### **Project Information**
- **Course / Project:** AI Final Project
- **Group Members:**
  - Syed Muhammad Bilal (58514)
  - Muhammad Zayan Amjad (65427)
- **Instructor:** Sir Rizwan Ahmed
- **Department:** BSCS (Room 207)

---

## 📌 Key Features

- **Privacy-Preserving Threat Detection:** Inspects only flow metadata (packet lengths, inter-arrival times, flow duration, TCP flags) that remain visible under TLS/HTTPS/VPN encryption without touching payloads.
- **Robust Class Balancing:** Implements intelligent stratified sampling on the ~2.83M row CICIDS2017 dataset, retaining all rare/critical attack instances (Infiltration, Heartbleed, Botnets, Web Attacks) while downsampling benign traffic.
- **Dual Baseline Architecture:** Trains, evaluates, and compares **Random Forest** and **XGBoost** classifiers on a held-out test set of 26,400+ flows.
- **Multi-Class & Binary Verdicts:** Outputs a binary risk status (`BENIGN` vs. `MALICIOUS`), threat probability, assigned risk tier (`LOW`, `MODERATE`, `HIGH`, `CRITICAL`), and specific attack categorization.
- **Interactive Streamlit Web Dashboard:**
  - **Single Flow Inspector:** One-click presets to test Benign HTTPS, DDoS, PortScan, DoS, Brute Force, Web Attacks, and Botnet flows.
  - **Batch CSV Flow Scanner:** Upload PCAP-extracted CSV flow logs to scan thousands of connections at once and download threat reports.
  - **Model Explainability:** Interactive visualizations of the top features identifying threats in encrypted traffic.

---

## 📂 Project Structure

```
AIFinal/
│
├── archive/                           # Raw CICIDS2017 CSV flow captures (8 files)
│   ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
│   ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
│   ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
│   ├── Monday-WorkingHours.pcap_ISCX.csv
│   ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
│   ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
│   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   └── Wednesday-workingHours.pcap_ISCX.csv
│
├── data/                              # Clean datasets and demo test presets
│   ├── clean_dataset.csv              # Balanced 132k flow dataset
│   ├── feature_columns.json           # List of 65 observable flow features
│   ├── test_threat_detection_with_ground_truth.csv  # 1,000-flow test dataset (500 Threats vs 500 Benign with labels)
│   ├── test_threat_detection_unlabelled.csv         # 1,000-flow unlabelled dataset for blind threat testing
│   ├── sample_benign_flow.json        # Test profile: Normal HTTPS
│   ├── sample_ddos_flow.json          # Test profile: DDoS flood
│   ├── sample_portscan_flow.json      # Test profile: PortScan probe
│   ├── sample_dos_flow.json           # Test profile: DoS attack
│   ├── sample_bruteforce_flow.json    # Test profile: FTP/SSH Patator
│   ├── sample_webattack_flow.json     # Test profile: SQLi / XSS
│   ├── sample_botnet_flow.json        # Test profile: Bot communication
│   ├── sample_batch.csv               # 32 mixed flows for quick batch testing
│   └── sample_traffic.pcap            # Test PCAP capture with normal & malicious flows
│
├── models/                            # Trained artifacts & benchmark results
│   ├── threat_detector.joblib         # Serialized top model (XGBoost)
│   ├── scaler.joblib                  # Standard scaler fitted on training flows
│   ├── label_encoder.joblib           # Multi-class attack label encoder
│   ├── evaluation_results.json        # Test set metrics & confusion matrix
│   └── feature_importance.json       # Top flow features ranked by importance
│
├── src/                               # Modular source code
│   ├── __init__.py
│   ├── preprocess.py                  # Data cleaning, column normalization & sampling
│   ├── create_test_set.py             # Generates dedicated 50/50 threat test datasets
│   ├── pcap_extractor.py              # Pure-Python PCAP/PCAPNG flow reconstruction engine
│   ├── train.py                       # Model training & benchmarking pipeline
│   └── predict.py                     # CLI & programmatic inference engine
│
├── app.py                             # Interactive Streamlit Web Interface
├── requirements.txt                   # Dependency specifications
└── README.md                          # Full setup & run guide
```

---

## 🚀 Step-by-Step Run Guide

Follow these simple steps to run the project on Windows, macOS, or Linux.

### **Step 1: Install Dependencies**
Open your terminal or PowerShell in the project directory (`AIFinal`) and install the required packages:

```bash
pip install -r requirements.txt
```

*Required packages: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `streamlit`, `joblib`.*

---

### **Step 2: Preprocess & Balance Dataset**
*(Note: The clean dataset and sample presets are already generated in `data/`, but you can re-run this anytime.)*

Run the preprocessing script to clean the raw CSVs from `archive/`, remove zero-variance noise, handle null/infinite values, and create a balanced dataset:

```bash
python src/preprocess.py
```

**What this produces:**
- Cleans 8 raw CSVs (~2.83M rows).
- Eliminates 13 zero-variance or duplicate columns.
- Creates `data/clean_dataset.csv` (132,004 balanced flows).
- Generates `data/feature_columns.json` (65 flow features).
- Exports test presets (`sample_*.json` and `sample_batch.csv`).

---

### **Step 3: Train & Benchmark Baseline Models**
*(Note: Pre-trained models are already saved in `models/`, but you can re-train at any time.)*

Run the training pipeline to train and compare **Random Forest** and **XGBoost**:

```bash
python src/train.py
```

**What this produces:**
- Splits into 80% train (105,603 flows) and 20% test (26,401 flows).
- Scales flow features using `StandardScaler`.
- Trains Random Forest (100 trees) and XGBoost (150 trees).
- Compares metrics and selects the best model.
- Saves model artifacts to `models/threat_detector.joblib`, `models/scaler.joblib`, and `models/label_encoder.joblib`.
- Saves benchmark metrics to `models/evaluation_results.json`.

---

### **Step 4: Launch the Web Dashboard**

Start the interactive web frontend:

```bash
streamlit run app.py
```

Once started, open your web browser at:
👉 **`http://localhost:8501`**

#### How to Use the Dashboard:
1. **🔍 Single Flow Inspector:**
   - Use the **"Quick Load Sample Profile"** dropdown to select a traffic pattern (e.g. *BENIGN*, *DDoS Attack*, *PortScan Probe*, *Web Attack*).
   - Inspect or tweak any flow metadata fields (Destination Port, Flow Duration, Packet Lengths, Flags, Window Sizes).
   - Click **"Inspect Network Flow"** to see the instant verdict, risk level card, threat confidence percentage, and class probability chart.
2. **📁 Batch CSV Flow Scanner:**
   - Click **"Load 1,000 Test Flows (500 Threats vs 500 Benign)"** or upload either `data/test_threat_detection_with_ground_truth.csv` or `data/test_threat_detection_unlabelled.csv`.
   - The AI engine instantly scans all flows, displays overall detection rate, live ground-truth validation percentage, detected attack types breakdown, and a downloadable threat report CSV.
3. **🦈 PCAP Traffic Analyzer:**
   - Upload any raw Wireshark/tcpdump packet capture (`.pcap` or `.pcapng`), or click **"Load Included Sample PCAP (sample_traffic.pcap)"**.
   - The system automatically parses IP/TCP/UDP packets, reconstructs bidirectional network flows, computes 65 statistical flow features, and displays threat verdicts and confidence scores for every connection.
4. **📊 Model Benchmarks & Explainability:**
   - Review the side-by-side comparison table between Random Forest and XGBoost.
   - Explore the bar chart showing the **Top 15 Most Informative Flow Features** that identify threats under encryption.

---

### **Step 5: Quick Command-Line Testing (Optional)**

You can also test predictions and extract PCAPs directly from the command line:

```bash
# Analyze a PCAP capture directly from CLI:
python src/pcap_extractor.py data/sample_traffic.pcap

# Test a normal benign flow:
python src/predict.py --sample data/sample_benign_flow.json

# Test a DDoS attack flow:
python src/predict.py --sample data/sample_ddos_flow.json

# Test a PortScan attack flow:
python src/predict.py --sample data/sample_portscan_flow.json
```

---

## 📊 Benchmark Results (Held-Out Test Set)

Evaluated on **26,401 unseen test network flows** from the CICIDS2017 benchmark:

| Metric | Random Forest Baseline | XGBoost Classifier (Top Model) |
| :--- | :---: | :---: |
| **Threat Detection Rate (Recall)** | **99.74%** | **99.92%** |
| **Threat Precision** | **99.81%** | **99.80%** |
| **Binary F1-Score** | **99.77%** | **99.86%** |
| **Binary ROC-AUC** | **99.98%** | **99.99%** |
| **Multi-Class Accuracy** | **99.74%** | **99.81%** |
| **Multi-Class F1 (Weighted)** | **99.74%** | **99.81%** |

> **Key Takeaway:** Both tree-based models achieve near-perfect threat detection with less than 0.08% missed attacks. XGBoost was chosen as the primary engine for its superior detection rate on edge-case attacks.

---

## 🔍 Why Flow Metadata Detects Encrypted Threats

Even when the packet payload is encrypted by TLS 1.3, HTTPS, or a VPN tunnel, the following metadata features remain completely visible and reveal attack signatures:

1. **Packet Length Distributions (`Bwd Packet Length Min`, `Packet Length Mean`)**:
   - Automated attacks (PortScan, DoS) generate uniform, repetitive packet sizes.
   - Legitimate HTTPS web browsing produces asymmetric, heavily skewed packet size distributions.
2. **Timing Patterns & Inter-Arrival Times (`Idle Mean`, `Active Std`)**:
   - Attack scripts and botnet command-and-control beacons trigger packets in rigid intervals.
   - Real human users generate irregular pauses and bursts ("think time").
3. **TCP Flag & Window Characteristics (`PSH Flag Count`, `Init Win Bytes`)**:
   - Port scans and network probes exhibit distinctive TCP flag distributions during connection establishment before encryption is established.

---

## 🛡️ License & Acknowledgments
- Dataset: Canadian Institute for Cybersecurity (CIC), University of New Brunswick (CICIDS2017).
- Developed for the BSCS AI Final Project.
