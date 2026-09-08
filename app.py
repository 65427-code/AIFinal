"""
AI-Based Encrypted Traffic Threat Detection
Streamlit Interactive Frontend Application v2.0

Four tabs:
1. Single Flow Inspector
2. Batch CSV Scanner (with schema validation & flexible ground truth)
3. PCAP Traffic Analyzer (with truncation reporting & model info)
4. Model Benchmarks & Explainability (per-dataset, held-out family)
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from src.predict import ThreatPredictor
from src.pcap_extractor import extract_flows_from_pcap
from src.features.schema import CANONICAL_FEATURES
from src.features.validator import validate_dataframe
from src.config import (
    MODEL_METADATA_PATH, EVALUATION_RESULTS_PATH, FEATURE_IMPORTANCE_PATH,
    KNOWN_GROUND_TRUTH_COLUMNS, KNOWN_BENIGN_VALUES, KNOWN_MALICIOUS_VALUES,
)

# Page configuration
st.set_page_config(
    page_title="AI Encrypted Traffic Threat Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .verdict-box-safe {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        border-radius: 10px;
        padding: 1.2rem;
        color: #065F46;
        font-weight: bold;
        font-size: 1.3rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .verdict-box-threat {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 10px;
        padding: 1.2rem;
        color: #991B1B;
        font-weight: bold;
        font-size: 1.3rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    return ThreatPredictor(strict_validation=False)

@st.cache_data
def load_evaluation_data():
    metrics = None
    feat_imp = None
    metadata = None
    
    if os.path.exists(EVALUATION_RESULTS_PATH):
        with open(EVALUATION_RESULTS_PATH, "r") as f:
            metrics = json.load(f)
    
    if os.path.exists(FEATURE_IMPORTANCE_PATH):
        with open(FEATURE_IMPORTANCE_PATH, "r") as f:
            feat_imp = json.load(f)
    
    if os.path.exists(MODEL_METADATA_PATH):
        with open(MODEL_METADATA_PATH, "r") as f:
            metadata = json.load(f)
    
    return metrics, feat_imp, metadata

def load_preset(cat_key: str) -> dict:
    preset_files = {
        "BENIGN (Normal HTTPS)": "data/sample_benign_flow.json",
        "DDoS Attack": "data/sample_ddos_flow.json",
        "PortScan Probe": "data/sample_portscan_flow.json",
        "DoS Attack": "data/sample_dos_flow.json",
        "Brute Force Attack": "data/sample_bruteforce_flow.json",
        "Web Attack": "data/sample_webattack_flow.json",
        "Botnet Traffic": "data/sample_botnet_flow.json",
    }
    file_path = preset_files.get(cat_key)
    if file_path and os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return {}

def evaluate_ground_truth(results_df, gt_column, benign_value):
    """Evaluate predictions against ground truth."""
    true_binary = []
    for v in results_df[gt_column]:
        v_str = str(v).strip()
        if v_str.upper() in [bv.upper() for bv in KNOWN_BENIGN_VALUES] or v_str == benign_value:
            true_binary.append("BENIGN")
        else:
            true_binary.append("MALICIOUS")
    
    pred_binary = results_df["Predicted_Verdict"].values
    
    tp = sum(1 for t, p in zip(true_binary, pred_binary) if t == "MALICIOUS" and p == "MALICIOUS")
    tn = sum(1 for t, p in zip(true_binary, pred_binary) if t == "BENIGN" and p == "BENIGN")
    fp = sum(1 for t, p in zip(true_binary, pred_binary) if t == "BENIGN" and p == "MALICIOUS")
    fn = sum(1 for t, p in zip(true_binary, pred_binary) if t == "MALICIOUS" and p == "BENIGN")
    
    total = len(true_binary)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1, "fnr": fnr, "fpr": fpr,
        "total": total,
    }

# Header
st.markdown('<div class="main-title">🛡️ AI-Based Encrypted Traffic Threat Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Threat detection from observable network-flow metadata without payload decryption.</div>', unsafe_allow_html=True)

# Check model availability
try:
    predictor = load_predictor()
    eval_metrics, feat_imp, model_meta = load_evaluation_data()
    model_ready = True
except Exception as e:
    model_ready = False
    st.error(f"⚠️ Model artifacts not found or could not be loaded: {e}")
    if os.path.exists("models/threat_detector.joblib") and not os.path.exists("models/binary_threat_detector.joblib"):
        st.warning("🔄 Legacy model detected. Run `python src/train.py` to retrain with the new multi-dataset architecture.")
    else:
        st.info("Run `python src/prepare_datasets.py` then `python src/train.py` to train the model.")

if model_ready:
    # Sidebar Info
    st.sidebar.markdown("### **System Status**")
    model_info = predictor.get_model_info()
    
    if model_info["is_legacy"]:
        st.sidebar.warning("⚠️ Legacy Model (retrain recommended)")
    else:
        st.sidebar.success("● AI Engine: Active (v2.0)")
    
    st.sidebar.markdown(f"**Model**: {model_info.get('model_type', 'Unknown')}")
    st.sidebar.markdown(f"**Features**: {model_info['feature_count']}")
    st.sidebar.markdown(f"**Threshold**: {model_info['binary_threshold']:.3f}")
    
    if model_info.get("datasets_used"):
        st.sidebar.markdown(f"**Training Datasets**: {', '.join(model_info['datasets_used'])}")
    
    if model_info.get("has_anomaly_detector"):
        st.sidebar.markdown("**OOD Detection**: ✅ Enabled")
    
    if eval_metrics and not model_info["is_legacy"]:
        binary_test = eval_metrics.get("binary_detector", {}).get("internal_test", {})
        if binary_test:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### **Test Set Performance**")
            st.sidebar.metric("Recall", f"{binary_test.get('recall', 0)*100:.2f}%")
            st.sidebar.metric("Precision", f"{binary_test.get('precision', 0)*100:.2f}%")
            st.sidebar.metric("F1-Score", f"{binary_test.get('f1_score', 0)*100:.2f}%")
    elif eval_metrics and model_info["is_legacy"]:
        best_name = eval_metrics.get("best_model", "XGBoost")
        if "models" in eval_metrics and best_name in eval_metrics["models"]:
            bin_metrics = eval_metrics["models"][best_name]["binary_metrics"]
            st.sidebar.markdown("---")
            st.sidebar.markdown("### **Legacy Test Metrics**")
            st.sidebar.metric("Recall", f"{bin_metrics['recall']*100:.2f}%")
            st.sidebar.metric("Precision", f"{bin_metrics['precision']*100:.2f}%")

    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Single Flow Inspector",
        "📁 Batch CSV Scanner",
        "🦈 PCAP Traffic Analyzer",
        "📊 Model Benchmarks & Explainability"
    ])

    # ─────────────────────────────────────────────────────────────────
    # TAB 1: Single Flow Inspector
    # ─────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Inspect a Network Flow")
        st.markdown("Select a preset flow profile or adjust parameters to observe the AI verdict.")

        preset_choice = st.selectbox(
            "⚡ Quick Load Sample Profile:",
            ["BENIGN (Normal HTTPS)", "DDoS Attack", "PortScan Probe",
             "DoS Attack", "Brute Force Attack", "Web Attack", "Botnet Traffic"]
        )
        
        current_data = load_preset(preset_choice)
        
        st.markdown("#### Primary Flow Parameters")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            dest_port = st.number_input("Destination Port", value=int(current_data.get("Destination Port", 443)), min_value=0, max_value=65535)
            flow_duration = st.number_input("Flow Duration (µs)", value=float(current_data.get("Flow Duration", 1000.0)))
            total_fwd_pkts = st.number_input("Total Fwd Packets", value=int(current_data.get("Total Fwd Packets", 10)))
            total_bwd_pkts = st.number_input("Total Bwd Packets", value=int(current_data.get("Total Backward Packets", 10)))

        with col2:
            fwd_pkt_mean = st.number_input("Fwd Packet Length Mean", value=float(current_data.get("Fwd Packet Length Mean", 50.0)))
            bwd_pkt_min = st.number_input("Bwd Packet Length Min", value=float(current_data.get("Bwd Packet Length Min", 0.0)))
            pkt_len_mean = st.number_input("Packet Length Mean", value=float(current_data.get("Packet Length Mean", 60.0)))
            bwd_pkts_s = st.number_input("Bwd Packets/s", value=float(current_data.get("Bwd Packets/s", 10.0)))

        with col3:
            flow_iat_mean = st.number_input("Flow IAT Mean", value=float(current_data.get("Flow IAT Mean", 200.0)))
            idle_mean = st.number_input("Idle Mean", value=float(current_data.get("Idle Mean", 0.0)))
            active_std = st.number_input("Active Std", value=float(current_data.get("Active Std", 0.0)))
            psh_flag = st.number_input("PSH Flag Count", value=int(current_data.get("PSH Flag Count", 0)), min_value=0)

        with col4:
            init_win_fwd = st.number_input("Init Win Bytes Fwd", value=int(current_data.get("Init_Win_bytes_forward", 29200)))
            init_win_bwd = st.number_input("Init Win Bytes Bwd", value=int(current_data.get("Init_Win_bytes_backward", 29200)))
            act_data_pkt = st.number_input("Act Data Pkt Fwd", value=int(current_data.get("act_data_pkt_fwd", 2)))
            tot_bwd_len = st.number_input("Total Length of Bwd Pkts", value=float(current_data.get("Total Length of Bwd Packets", 1000.0)))

        flow_input = current_data.copy()
        flow_input.update({
            "Destination Port": dest_port, "Flow Duration": flow_duration,
            "Total Fwd Packets": total_fwd_pkts, "Total Backward Packets": total_bwd_pkts,
            "Fwd Packet Length Mean": fwd_pkt_mean, "Bwd Packet Length Min": bwd_pkt_min,
            "Packet Length Mean": pkt_len_mean, "Bwd Packets/s": bwd_pkts_s,
            "Flow IAT Mean": flow_iat_mean, "Idle Mean": idle_mean,
            "Active Std": active_std, "PSH Flag Count": psh_flag,
            "Init_Win_bytes_forward": init_win_fwd, "Init_Win_bytes_backward": init_win_bwd,
            "act_data_pkt_fwd": act_data_pkt, "Total Length of Bwd Packets": tot_bwd_len,
        })

        if st.button("🔍 Inspect Network Flow", type="primary", use_container_width=True):
            result = predictor.predict_single(flow_input)
            
            st.markdown("---")
            if result["is_threat"]:
                st.markdown(f'<div class="verdict-box-threat">🚨 VERDICT: {result["verdict"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verdict-box-safe">✅ VERDICT: {result["verdict"]}</div>', unsafe_allow_html=True)
                
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Predicted Category", result["predicted_category"])
            with m2:
                st.metric("Threat Confidence", f"{result['threat_confidence']}%")
            with m3:
                st.metric("Benign Confidence", f"{result['benign_confidence']}%")
            with m4:
                st.metric("Risk Level", result["risk_level"])
                
            if result.get("ood_warning"):
                st.warning("⚠️ This flow has unusual characteristics (Out-of-Distribution). Exercise caution.")
            
            st.markdown("#### Probability Distribution")
            probs_df = pd.DataFrame(
                list(result["class_probabilities"].items()),
                columns=["Class", "Probability"]
            ).set_index("Class")
            st.bar_chart(probs_df)

    # ─────────────────────────────────────────────────────────────────
    # TAB 2: Batch CSV Scanner
    # ─────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Batch Flow Threat Scanner")
        st.markdown("Upload a network flow CSV to scan for encrypted threats.")

        col_left, col_right = st.columns([2, 1])
        with col_left:
            uploaded_file = st.file_uploader("Upload Network Flows CSV", type=["csv"])
        with col_right:
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            use_1k_test = st.button("🧪 Load 1,000 Test Flows", use_container_width=True)
            use_sample = st.button("⚡ Load Mini Batch (32 flows)", use_container_width=True)

        batch_df = None
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(batch_df)} rows from uploaded file.")
        elif use_1k_test:
            test_path = "data/test_threat_detection_with_ground_truth.csv"
            if os.path.exists(test_path):
                batch_df = pd.read_csv(test_path)
                st.info(f"Loaded {len(batch_df)} test flows.")
            else:
                st.warning("Test dataset not found.")
        elif use_sample:
            sample_path = "data/sample_batch.csv"
            if os.path.exists(sample_path):
                batch_df = pd.read_csv(sample_path)
                st.info(f"Loaded {len(batch_df)} sample flows.")

        if batch_df is not None:
            # Schema compatibility check
            validation = validate_dataframe(batch_df, strict=False)
            
            with st.expander("📋 CSV Schema Compatibility", expanded=True):
                if validation.compatible:
                    st.success(f"✅ COMPATIBLE — {validation.present_count}/{validation.expected_count} features detected")
                else:
                    st.error(f"❌ INCOMPATIBLE — {validation.missing_count} features missing")
                    st.text("Missing features:\n" + "\n".join(f"  - {f}" for f in validation.missing[:20]))
                
                if validation.renamed:
                    st.info(f"🔄 Renamed {len(validation.renamed)} columns via alias normalization")
                
                if validation.warnings:
                    for w in validation.warnings:
                        st.warning(w)
            
            with st.spinner("Scanning flows with AI engine..."):
                scanned_results = predictor.predict_batch(batch_df)
                
            threat_count = int((scanned_results["Predicted_Verdict"] == "MALICIOUS").sum())
            benign_count = int((scanned_results["Predicted_Verdict"] == "BENIGN").sum())
            total_count = len(scanned_results)
            threat_pct = (threat_count / total_count * 100) if total_count > 0 else 0.0

            st.markdown("---")
            st.markdown("#### Scan Summary")
            
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Total Flows", total_count)
            kpi2.metric("Threats Detected", threat_count, delta=f"{threat_pct:.1f}% Rate", delta_color="inverse")
            kpi3.metric("Benign Flows", benign_count)
            kpi4.metric("Security Posture", "CRITICAL" if threat_pct > 30 else ("ELEVATED" if threat_pct > 0 else "NORMAL"))

            # Ground truth evaluation
            gt_columns = [c for c in scanned_results.columns if c in KNOWN_GROUND_TRUTH_COLUMNS]
            
            if gt_columns:
                st.markdown("#### 🎯 Ground Truth Evaluation")
                
                gt_col = st.selectbox("Ground Truth Column:", gt_columns)
                unique_vals = sorted(scanned_results[gt_col].dropna().unique().astype(str))
                
                benign_val = st.selectbox(
                    "Benign label value:",
                    [v for v in unique_vals if v.upper() in [bv.upper() for bv in KNOWN_BENIGN_VALUES]] or unique_vals[:1]
                )
                
                gt_results = evaluate_ground_truth(scanned_results, gt_col, benign_val)
                
                g1, g2, g3, g4 = st.columns(4)
                g1.metric("Accuracy", f"{gt_results['accuracy']*100:.2f}%")
                g2.metric("Precision", f"{gt_results['precision']*100:.2f}%")
                g3.metric("Recall", f"{gt_results['recall']*100:.2f}%")
                g4.metric("F1-Score", f"{gt_results['f1']*100:.2f}%")
                
                cm1, cm2, cm3, cm4 = st.columns(4)
                cm1.metric("True Positives", gt_results["tp"])
                cm2.metric("True Negatives", gt_results["tn"])
                cm3.metric("False Positives", gt_results["fp"])
                cm4.metric("False Negatives", gt_results["fn"])
                
                st.metric("False Negative Rate", f"{gt_results['fnr']*100:.2f}%")

            # Category breakdown
            st.markdown("#### Detected Attack Types")
            cat_counts = scanned_results["Predicted_Category"].value_counts()
            st.bar_chart(cat_counts)

            # Results table
            st.markdown("#### Detailed Flow Logs")
            preview_cols = []
            for c in gt_columns:
                preview_cols.append(c)
            preview_cols.extend(["Predicted_Verdict", "Predicted_Category", "Threat_Confidence_%", "Risk_Level"])
            for c in ["Destination Port", "Flow Duration", "Total Fwd Packets", "Packet Length Mean"]:
                if c in scanned_results.columns:
                    preview_cols.append(c)
            if "OOD_Score" in scanned_results.columns:
                preview_cols.append("OOD_Score")
            
            show_cols = [c for c in preview_cols if c in scanned_results.columns]
            st.dataframe(scanned_results[show_cols], use_container_width=True)

            csv_data = scanned_results.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Full Report (CSV)", data=csv_data,
                             file_name="threat_report.csv", mime="text/csv", use_container_width=True)

    # ─────────────────────────────────────────────────────────────────
    # TAB 3: PCAP Traffic Analyzer
    # ─────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### 🦈 PCAP / PCAPNG Network Traffic Threat Analyzer")
        st.markdown("Upload a raw packet capture. The engine reconstructs bidirectional flows and detects threats from metadata without decrypting payloads.")

        col_pcap_left, col_pcap_right = st.columns([2, 1])
        with col_pcap_left:
            uploaded_pcap = st.file_uploader("Upload Packet Capture (.pcap, .pcapng)", type=["pcap", "pcapng", "cap"])
        with col_pcap_right:
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            use_sample_pcap = st.button("🦈 Load Sample PCAP", use_container_width=True)

        pcap_source = None
        pcap_name = ""
        if uploaded_pcap is not None:
            pcap_source = uploaded_pcap
            pcap_name = uploaded_pcap.name
        elif use_sample_pcap:
            sample_path = "data/sample_traffic.pcap"
            if os.path.exists(sample_path):
                pcap_source = sample_path
                pcap_name = "sample_traffic.pcap"
            else:
                st.warning("Sample PCAP not found.")

        if pcap_source is not None:
            with st.spinner(f"Parsing packets & reconstructing flows from {pcap_name}..."):
                pcap_flows_df, pcap_meta = extract_flows_from_pcap(pcap_source)
                
            if pcap_flows_df.empty:
                st.warning("⚠️ No valid TCP/UDP flows found in this capture.")
            else:
                # Get extraction info
                extraction_info = pcap_flows_df.attrs.get("extraction_info", {})
                
                with st.spinner("Classifying flows with AI engine..."):
                    pcap_results = predictor.predict_batch(pcap_flows_df)

                threat_count = int((pcap_results["Predicted_Verdict"] == "MALICIOUS").sum())
                benign_count = int((pcap_results["Predicted_Verdict"] == "BENIGN").sum())
                total_flows = len(pcap_results)
                total_pkts = extraction_info.get("packets_processed", sum(m.get("Packets", 0) for m in pcap_meta))
                total_bytes = sum(m.get("Bytes", 0) for m in pcap_meta)
                threat_pct = (threat_count / total_flows * 100) if total_flows > 0 else 0.0
                was_truncated = extraction_info.get("was_truncated", False)

                st.markdown("---")
                st.markdown(f"#### Capture Summary: `{pcap_name}`")
                
                kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
                kpi1.metric("Packets Processed", f"{total_pkts:,}")
                kpi2.metric("Flows Extracted", f"{total_flows:,}")
                kpi3.metric("Threats Detected", threat_count, delta=f"{threat_pct:.1f}% Rate", delta_color="inverse")
                kpi4.metric("Benign Flows", benign_count)
                kpi5.metric("Total Volume", f"{total_bytes / 1024:.1f} KB")
                
                if was_truncated:
                    st.warning(f"⚠️ Capture was TRUNCATED at {extraction_info.get('max_packets_setting', 'N/A')} packets. "
                              f"Results may not represent the full capture.")
                
                # Model info expander
                with st.expander("ℹ️ Analysis Details"):
                    st.markdown(f"- **Feature Extractor**: v2.0")
                    st.markdown(f"- **Model Version**: {model_info.get('schema_version', '1.0')}")
                    st.markdown(f"- **Binary Threshold**: {model_info['binary_threshold']:.3f}")
                    if model_info.get("datasets_used"):
                        st.markdown(f"- **Trained On**: {', '.join(model_info['datasets_used'])}")
                    if model_info.get("has_anomaly_detector"):
                        st.markdown("- **OOD Detection**: Enabled")

                # Threat distribution
                st.markdown("#### Detected Attack Types")
                cat_counts = pcap_results["Predicted_Category"].value_counts()
                st.bar_chart(cat_counts)

                # Flow table
                st.markdown("#### Extracted Network Flows")
                filter_choice = st.radio("Filter:", ["All Flows", "Threats Only", "Benign Only"], horizontal=True)
                
                display_df = pcap_results.copy()
                if filter_choice == "Threats Only":
                    display_df = display_df[display_df["Predicted_Verdict"] == "MALICIOUS"]
                elif filter_choice == "Benign Only":
                    display_df = display_df[display_df["Predicted_Verdict"] == "BENIGN"]

                pcap_display_cols = [
                    "Source_IP", "Source_Port", "Destination_IP", "Destination_Port",
                    "Protocol", "Packets", "Predicted_Verdict", "Predicted_Category",
                    "Threat_Confidence_%", "Risk_Level"
                ]
                if "OOD_Score" in display_df.columns:
                    pcap_display_cols.append("OOD_Score")
                
                show_cols = [c for c in pcap_display_cols if c in display_df.columns]
                st.dataframe(display_df[show_cols], use_container_width=True)

                pcap_csv = pcap_results.to_csv(index=False).encode("utf-8")
                clean_fname = pcap_name.split()[0].replace(".pcapng", "").replace(".pcap", "")
                st.download_button("📥 Download PCAP Flow Report (CSV)", data=pcap_csv,
                                 file_name=f"{clean_fname}_threat_report.csv", mime="text/csv",
                                 use_container_width=True)

    # ─────────────────────────────────────────────────────────────────
    # TAB 4: Model Benchmarks & Explainability
    # ─────────────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### Model Benchmarks & Feature Explainability")
        
        if model_info["is_legacy"]:
            st.warning("⚠️ Showing LEGACY model metrics. These are from CICIDS2017-only training and may not generalize to external traffic.")
        
        if eval_metrics:
            # v2.0 format
            if "binary_detector" in eval_metrics:
                binary_info = eval_metrics["binary_detector"]
                test_metrics = binary_info.get("internal_test", {})
                
                st.markdown("#### 1. Primary Binary Detector Performance")
                st.markdown(f"**Model**: {binary_info.get('model', 'Unknown')} | "
                          f"**Threshold**: {binary_info.get('threshold', 0.5):.3f}")
                
                if test_metrics:
                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("Recall", f"{test_metrics.get('recall', 0)*100:.2f}%")
                    b2.metric("Precision", f"{test_metrics.get('precision', 0)*100:.2f}%")
                    b3.metric("F1-Score", f"{test_metrics.get('f1_score', 0)*100:.2f}%")
                    b4.metric("ROC-AUC", f"{test_metrics.get('roc_auc', 0)*100:.2f}%")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("FPR", f"{test_metrics.get('false_positive_rate', 0)*100:.2f}%")
                    c2.metric("FNR", f"{test_metrics.get('false_negative_rate', 0)*100:.2f}%")
                    c3.metric("Specificity", f"{test_metrics.get('specificity', 0)*100:.2f}%")
                    c4.metric("Accuracy", f"{test_metrics.get('accuracy', 0)*100:.2f}%")
                    
                    st.markdown("**Confusion Matrix:**")
                    cm_data = {
                        "": ["Actual BENIGN", "Actual MALICIOUS"],
                        "Predicted BENIGN": [test_metrics.get("true_negatives", 0), test_metrics.get("false_negatives", 0)],
                        "Predicted MALICIOUS": [test_metrics.get("false_positives", 0), test_metrics.get("true_positives", 0)],
                    }
                    st.table(pd.DataFrame(cm_data).set_index(""))
                
                # Per-dataset results
                per_dataset = eval_metrics.get("datasets", {})
                if per_dataset:
                    st.markdown("#### 2. Per-Dataset External Performance")
                    
                    rows = []
                    for ds_name, ds_metrics in per_dataset.items():
                        rows.append({
                            "Dataset": ds_name,
                            "Samples": ds_metrics.get("sample_count", "N/A"),
                            "Recall": f"{ds_metrics.get('recall', 0)*100:.1f}%",
                            "Precision": f"{ds_metrics.get('precision', 0)*100:.1f}%",
                            "F1": f"{ds_metrics.get('f1_score', 0)*100:.1f}%",
                            "FPR": f"{ds_metrics.get('false_positive_rate', 0)*100:.1f}%",
                            "FNR": f"{ds_metrics.get('false_negative_rate', 0)*100:.1f}%",
                        })
                    
                    st.table(pd.DataFrame(rows))
                
                # Held-out family results
                held_out = eval_metrics.get("held_out_families", {})
                if held_out:
                    st.markdown("#### 3. Held-Out Malware Family Test (Generalization)")
                    st.markdown("*These families were NOT included in training — this tests unseen threat detection.*")
                    
                    for family, fam_metrics in held_out.items():
                        st.markdown(f"**{family}:**")
                        f1, f2, f3, f4 = st.columns(4)
                        f1.metric("Total Flows", fam_metrics.get("total_malicious", 0) + fam_metrics.get("total_benign", 0))
                        f2.metric("Detected", fam_metrics.get("detected_malicious", 0))
                        f3.metric("Missed", fam_metrics.get("missed_malicious", 0))
                        f4.metric("Recall", f"{fam_metrics.get('recall', 0)*100:.1f}%")
                
                # Training composition
                composition = eval_metrics.get("training_composition", {})
                if composition:
                    st.markdown("#### 4. Training Dataset Composition")
                    comp_df = pd.DataFrame([
                        {"Dataset": k, "Samples": v} for k, v in composition.items()
                    ])
                    st.bar_chart(comp_df.set_index("Dataset"))
            
            # Legacy format
            elif "models" in eval_metrics:
                st.markdown("#### Legacy Model Comparison (CICIDS2017 only)")
                models_dict = eval_metrics.get("models", {})
                
                comparison_rows = []
                for m_name, m_data in models_dict.items():
                    b = m_data["binary_metrics"]
                    comparison_rows.append({
                        "Model": m_name,
                        "Recall": f"{b['recall']*100:.2f}%",
                        "Precision": f"{b['precision']*100:.2f}%",
                        "F1": f"{b['f1_score']*100:.2f}%",
                        "ROC-AUC": f"{b['roc_auc']*100:.2f}%",
                        "Accuracy": f"{m_data['multiclass_metrics']['accuracy']*100:.2f}%",
                    })
                st.table(pd.DataFrame(comparison_rows))

        # Feature Importance
        if feat_imp:
            st.markdown("#### Feature Importance (Top 15)")
            st.markdown("Flow-level statistical features that carry the strongest signal for threat detection:")
            
            top_15 = feat_imp[:15]
            imp_df = pd.DataFrame(top_15)
            imp_df["Importance (%)"] = imp_df["importance"] * 100
            imp_df = imp_df.sort_values(by="Importance (%)", ascending=True)
            
            chart_data = imp_df.set_index("feature")["Importance (%)"]
            st.bar_chart(chart_data)

        # Limitations
        st.markdown("---")
        st.markdown("#### Model Limitations")
        st.markdown("""
        - **No payload decryption**: Detection relies exclusively on observable flow metadata.
        - **Feature compatibility**: CICIDS2017 features come from CICFlowMeter; USTC features from our canonical extractor. Minor distribution differences exist.
        - **Encrypted tunnel detection**: Requires CIRA-CIC-DoHBrw-2020 training data for full coverage.
        - **Unknown threats**: Novel attack patterns outside the training distribution may not be detected.
        - **Threshold sensitivity**: The binary threshold was calibrated on validation data and may need adjustment for specific deployment environments.
        """)

        st.markdown("#### Why Flow Metadata Detects Encrypted Threats")
        st.markdown("""
        - **Packet Length Distributions**: Attacks produce uniform packet sizes; legitimate browsing creates skewed distributions.
        - **Timing Patterns**: Automated tools transmit in rigid intervals; human users exhibit irregular pauses.
        - **TCP State Metadata**: Malicious scanners exhibit anomalous flag combinations before encryption negotiation.
        - **Active/Idle Patterns**: Malware C2 beacons show distinctive activity-inactivity cycles.
        """)
