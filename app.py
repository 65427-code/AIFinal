"""
AI-Based Encrypted Traffic Threat Detection
Streamlit Interactive Frontend Application
"""

import os
import json
import pandas as pd
import streamlit as st
from src.predict import ThreatPredictor
from src.pcap_extractor import extract_flows_from_pcap

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
    return ThreatPredictor()

@st.cache_data
def load_evaluation_data():
    metrics_path = "models/evaluation_results.json"
    feat_imp_path = "models/feature_importance.json"
    
    metrics = None
    feat_imp = None
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    if os.path.exists(feat_imp_path):
        with open(feat_imp_path, "r") as f:
            feat_imp = json.load(f)
    return metrics, feat_imp

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

# Header
st.markdown('<div class="main-title">🛡️ AI-Based Encrypted Traffic Threat Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Detect cyber threats hidden inside encrypted network flows (HTTPS/TLS/VPN) without payload decryption.</div>', unsafe_allow_html=True)

# Check model availability
try:
    predictor = load_predictor()
    eval_metrics, feat_imp = load_evaluation_data()
    model_ready = True
except Exception as e:
    model_ready = False
    st.error(f"⚠️ Model artifacts not found or could not be loaded: {e}. Please run `python src/train.py` first.")

if model_ready:
    # Sidebar Info
    st.sidebar.image("https://img.icons8.com/color/96/shield.png", width=70)
    st.sidebar.markdown("### **System Status**")
    st.sidebar.success("● AI Engine: Active")
    st.sidebar.info(f"**Best Model**: {eval_metrics.get('best_model', 'XGBoost') if eval_metrics else 'XGBoost'}")
    st.sidebar.markdown(f"**Flow Features**: {len(predictor.feature_names)}")
    st.sidebar.markdown(f"**Known Classes**: {len(predictor.classes)}")
    
    if eval_metrics:
        best_name = eval_metrics.get("best_model", "XGBoost")
        bin_metrics = eval_metrics["models"][best_name]["binary_metrics"]
        st.sidebar.markdown("---")
        st.sidebar.markdown("### **Test Set Accuracy**")
        st.sidebar.metric("Detection Rate (Recall)", f"{bin_metrics['recall']*100:.2f}%")
        st.sidebar.metric("Precision", f"{bin_metrics['precision']*100:.2f}%")
        st.sidebar.metric("F1-Score", f"{bin_metrics['f1_score']*100:.2f}%")

    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Single Flow Inspector",
        "📁 Batch CSV Scanner",
        "🦈 PCAP Traffic Analyzer",
        "📊 Model Benchmarks & Explainability"
    ])

    # -------------------------------------------------------------
    # TAB 1: Single Flow Inspector
    # -------------------------------------------------------------
    with tab1:
        st.markdown("### Inspect a Network Flow")
        st.markdown("Select a preset flow profile or adjust network metadata parameters to observe the AI verdict.")

        preset_choice = st.selectbox(
            "⚡ Quick Load Sample Profile:",
            [
                "BENIGN (Normal HTTPS)",
                "DDoS Attack",
                "PortScan Probe",
                "DoS Attack",
                "Brute Force Attack",
                "Web Attack",
                "Botnet Traffic"
            ]
        )
        
        current_data = load_preset(preset_choice)
        
        st.markdown("#### Primary Flow Parameters")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            dest_port = st.number_input(
                "Destination Port",
                value=int(current_data.get("Destination Port", 443)),
                min_value=0, max_value=65535
            )
            flow_duration = st.number_input(
                "Flow Duration (µs)",
                value=float(current_data.get("Flow Duration", 1000.0))
            )
            total_fwd_pkts = st.number_input(
                "Total Fwd Packets",
                value=int(current_data.get("Total Fwd Packets", 10))
            )
            total_bwd_pkts = st.number_input(
                "Total Bwd Packets",
                value=int(current_data.get("Total Backward Packets", 10))
            )

        with col2:
            fwd_pkt_mean = st.number_input(
                "Fwd Packet Length Mean",
                value=float(current_data.get("Fwd Packet Length Mean", 50.0))
            )
            bwd_pkt_min = st.number_input(
                "Bwd Packet Length Min",
                value=float(current_data.get("Bwd Packet Length Min", 0.0))
            )
            pkt_len_mean = st.number_input(
                "Packet Length Mean",
                value=float(current_data.get("Packet Length Mean", 60.0))
            )
            bwd_pkts_s = st.number_input(
                "Bwd Packets/s",
                value=float(current_data.get("Bwd Packets/s", 10.0))
            )

        with col3:
            flow_iat_mean = st.number_input(
                "Flow IAT Mean",
                value=float(current_data.get("Flow IAT Mean", 200.0))
            )
            idle_mean = st.number_input(
                "Idle Mean",
                value=float(current_data.get("Idle Mean", 0.0))
            )
            active_std = st.number_input(
                "Active Std",
                value=float(current_data.get("Active Std", 0.0))
            )
            psh_flag = st.number_input(
                "PSH Flag Count",
                value=int(current_data.get("PSH Flag Count", 0)),
                min_value=0, max_value=1
            )

        with col4:
            init_win_fwd = st.number_input(
                "Init Win Bytes Fwd",
                value=int(current_data.get("Init_Win_bytes_forward", 29200))
            )
            init_win_bwd = st.number_input(
                "Init Win Bytes Bwd",
                value=int(current_data.get("Init_Win_bytes_backward", 29200))
            )
            act_data_pkt = st.number_input(
                "Act Data Pkt Fwd",
                value=int(current_data.get("act_data_pkt_fwd", 2))
            )
            tot_bwd_len = st.number_input(
                "Total Length of Bwd Pkts",
                value=float(current_data.get("Total Length of Bwd Packets", 1000.0))
            )

        # Merge user changes into flow dictionary
        flow_input = current_data.copy()
        flow_input["Destination Port"] = dest_port
        flow_input["Flow Duration"] = flow_duration
        flow_input["Total Fwd Packets"] = total_fwd_pkts
        flow_input["Total Backward Packets"] = total_bwd_pkts
        flow_input["Fwd Packet Length Mean"] = fwd_pkt_mean
        flow_input["Bwd Packet Length Min"] = bwd_pkt_min
        flow_input["Packet Length Mean"] = pkt_len_mean
        flow_input["Bwd Packets/s"] = bwd_pkts_s
        flow_input["Flow IAT Mean"] = flow_iat_mean
        flow_input["Idle Mean"] = idle_mean
        flow_input["Active Std"] = active_std
        flow_input["PSH Flag Count"] = psh_flag
        flow_input["Init_Win_bytes_forward"] = init_win_fwd
        flow_input["Init_Win_bytes_backward"] = init_win_bwd
        flow_input["act_data_pkt_fwd"] = act_data_pkt
        flow_input["Total Length of Bwd Packets"] = tot_bwd_len

        st.markdown("")
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
                st.metric("Assigned Risk Level", result["risk_level"])
                
            st.markdown("#### Threat Category Probability Distribution")
            probs_df = pd.DataFrame(
                list(result["class_probabilities"].items()),
                columns=["Attack Category", "Probability"]
            ).set_index("Attack Category")
            
            st.bar_chart(probs_df)

    # -------------------------------------------------------------
    # TAB 2: Batch CSV Scanner
    # -------------------------------------------------------------
    with tab2:
        st.markdown("### Batch Flow Threat Scanner")
        st.markdown("Upload a network flow CSV file to scan multiple flows simultaneously for encrypted threats.")

        col_left, col_right = st.columns([2, 1])
        with col_left:
            uploaded_file = st.file_uploader("Upload Network Flows CSV", type=["csv"])
        with col_right:
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            use_1k_test = st.button("🧪 Load 1,000 Test Flows (500 Threats vs 500 Benign)", use_container_width=True)
            use_sample = st.button("⚡ Load Mini Batch (32 flows)", use_container_width=True)

        batch_df = None
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(batch_df)} rows from uploaded file.")
        elif use_1k_test:
            test_1k_path = "data/test_threat_detection_with_ground_truth.csv"
            if os.path.exists(test_1k_path):
                batch_df = pd.read_csv(test_1k_path)
                st.info(f"Loaded 1,000 dedicated test flows (500 Threats vs 500 Benign) from {test_1k_path}.")
            else:
                st.warning("1k test dataset not found. Run python src/create_test_set.py first.")
        elif use_sample:
            sample_batch_path = "data/sample_batch.csv"
            if os.path.exists(sample_batch_path):
                batch_df = pd.read_csv(sample_batch_path)
                st.info(f"Loaded {len(batch_df)} sample network flows from {sample_batch_path}.")
            else:
                st.warning("Sample batch file not found. Run preprocessing first.")

        if batch_df is not None:
            with st.spinner("Scanning network flows with AI engine..."):
                scanned_results = predictor.predict_batch(batch_df)
                
            threat_count = int((scanned_results["Predicted_Verdict"] == "MALICIOUS").sum())
            benign_count = int((scanned_results["Predicted_Verdict"] == "BENIGN").sum())
            total_count = len(scanned_results)
            threat_pct = (threat_count / total_count * 100) if total_count > 0 else 0.0

            st.markdown("---")
            st.markdown("#### Scan Summary")
            
            # Ground truth validation banner if labels exist in data
            has_ground_truth = "Expected_Verdict" in scanned_results.columns
            if has_ground_truth:
                true_vals = ["MALICIOUS" if str(v).upper() == "THREAT" else "BENIGN" for v in scanned_results["Expected_Verdict"]]
                pred_vals = scanned_results["Predicted_Verdict"].values
                matched = sum(p == t for p, t in zip(pred_vals, true_vals))
                acc_rate = (matched / total_count) * 100
                st.success(f"🎯 **Ground Truth Comparison**: {matched}/{total_count} ({acc_rate:.2f}%) flows accurately classified against actual labels!")

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Total Flows Scanned", total_count)
            kpi2.metric("Threats Detected", threat_count, delta=f"{threat_pct:.1f}% Threat Rate", delta_color="inverse")
            kpi3.metric("Benign Flows", benign_count)
            kpi4.metric("Security Posture", "CRITICAL" if threat_pct > 30 else ("ELEVATED" if threat_pct > 0 else "NORMAL"))

            # Breakdown chart
            st.markdown("#### Detected Attack Types Breakdown")
            cat_counts = scanned_results["Predicted_Category"].value_counts()
            st.bar_chart(cat_counts)

            # Table of results
            st.markdown("#### Detailed Flow Logs")
            preview_cols = []
            if "Expected_Verdict" in scanned_results.columns:
                preview_cols.append("Expected_Verdict")
            preview_cols.append("Predicted_Verdict")
            if "Actual_Attack_Type" in scanned_results.columns:
                preview_cols.append("Actual_Attack_Type")
            preview_cols.extend(["Predicted_Category", "Threat_Confidence_%", "Risk_Level"])
            
            # include a few raw flow columns if present
            for c in ["Destination Port", "Flow Duration", "Total Fwd Packets", "Packet Length Mean"]:
                if c in scanned_results.columns:
                    preview_cols.append(c)
                    
            st.dataframe(scanned_results[preview_cols], use_container_width=True)

            # CSV Download
            csv_data = scanned_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Full Threat Report (CSV)",
                data=csv_data,
                file_name="encrypted_traffic_threat_report.csv",
                mime="text/csv",
                use_container_width=True
            )

    # -------------------------------------------------------------
    # TAB 3: PCAP Traffic Analyzer
    # -------------------------------------------------------------
    with tab3:
        st.markdown("### 🦈 PCAP / PCAPNG Network Traffic Threat Analyzer")
        st.markdown("Upload a raw packet capture (`.pcap` or `.pcapng`). The engine reconstructs bidirectional TCP/UDP network flows and extracts statistical features to detect threats without decrypting payloads.")

        col_pcap_left, col_pcap_right = st.columns([2, 1])
        with col_pcap_left:
            uploaded_pcap = st.file_uploader("Upload Packet Capture (.pcap, .pcapng)", type=["pcap", "pcapng", "cap"])
        with col_pcap_right:
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            use_sample_pcap = st.button("🦈 Load Included Sample PCAP (sample_traffic.pcap)", use_container_width=True)

        pcap_source = None
        pcap_name = ""
        if uploaded_pcap is not None:
            pcap_source = uploaded_pcap
            pcap_name = uploaded_pcap.name
        elif use_sample_pcap:
            sample_pcap_path = "data/sample_traffic.pcap"
            if os.path.exists(sample_pcap_path):
                pcap_source = sample_pcap_path
                pcap_name = "sample_traffic.pcap (Live Network Capture)"
            else:
                st.warning("Sample PCAP not found in data/.")

        if pcap_source is not None:
            with st.spinner(f"Parsing packets & reconstructing network flows from {pcap_name}..."):
                pcap_flows_df, pcap_meta = extract_flows_from_pcap(pcap_source)
                
            if pcap_flows_df.empty:
                st.warning("⚠️ No valid IPv4/IPv6 TCP or UDP traffic flows were found in this packet capture.")
            else:
                with st.spinner("Classifying flows with AI threat detection engine..."):
                    pcap_results = predictor.predict_batch(pcap_flows_df)

                threat_count = int((pcap_results["Predicted_Verdict"] == "MALICIOUS").sum())
                benign_count = int((pcap_results["Predicted_Verdict"] == "BENIGN").sum())
                total_flows = len(pcap_results)
                total_pkts = sum(m.get("Packets", 0) for m in pcap_meta)
                total_bytes = sum(m.get("Bytes", 0) for m in pcap_meta)
                threat_pct = (threat_count / total_flows * 100) if total_flows > 0 else 0.0

                st.markdown("---")
                st.markdown(f"#### Capture Summary: `{pcap_name}`")
                kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
                kpi1.metric("Packets Processed", f"{total_pkts:,}")
                kpi2.metric("Flows Extracted", f"{total_flows:,}")
                kpi3.metric("Threats Detected", threat_count, delta=f"{threat_pct:.1f}% Rate", delta_color="inverse")
                kpi4.metric("Benign Flows", benign_count)
                kpi5.metric("Total Volume", f"{total_bytes / 1024:.1f} KB")

                # Threat distribution
                st.markdown("#### Detected Attack Types in Capture")
                cat_counts = pcap_results["Predicted_Category"].value_counts()
                st.bar_chart(cat_counts)

                # Filter option
                st.markdown("#### Extracted Network Flows")
                filter_choice = st.radio(
                    "Filter Displayed Flows:",
                    ["All Flows", "Threats Only", "Benign Only"],
                    horizontal=True
                )
                
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
                show_cols = [c for c in pcap_display_cols if c in display_df.columns]
                st.dataframe(display_df[show_cols], use_container_width=True)

                # Download
                pcap_csv = pcap_results.to_csv(index=False).encode("utf-8")
                clean_fname = pcap_name.split()[0].replace(".pcapng", "").replace(".pcap", "")
                st.download_button(
                    "📥 Download Extracted PCAP Flow Report (CSV)",
                    data=pcap_csv,
                    file_name=f"{clean_fname}_threat_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    # -------------------------------------------------------------
    # TAB 4: Model Benchmarks & Explainability
    # -------------------------------------------------------------
    with tab4:
        st.markdown("### Model Benchmarks & Feature Explainability")
        st.markdown("Evaluation of baseline algorithms on the held-out test set (~26,400 network flows).")

        if eval_metrics:
            models_dict = eval_metrics.get("models", {})
            
            # Benchmark Comparison Table
            comparison_rows = []
            for m_name, m_data in models_dict.items():
                b = m_data["binary_metrics"]
                comparison_rows.append({
                    "Model": m_name,
                    "Detection Rate (Recall)": f"{b['recall']*100:.2f}%",
                    "Precision": f"{b['precision']*100:.2f}%",
                    "F1-Score": f"{b['f1_score']*100:.2f}%",
                    "ROC-AUC": f"{b['roc_auc']*100:.2f}%",
                    "Multi-Class Accuracy": f"{m_data['multiclass_metrics']['accuracy']*100:.2f}%"
                })
            
            st.markdown("#### 1. Baseline Model Comparison")
            st.table(pd.DataFrame(comparison_rows))

            # Feature Importance
            if feat_imp:
                st.markdown("#### 2. Top Informative Features (Observable Under Encryption)")
                st.markdown(
                    "These flow-level statistical metrics carry the strongest signal for identifying threats "
                    "without needing to decrypt payload content:"
                )
                
                top_15 = feat_imp[:15]
                imp_df = pd.DataFrame(top_15)
                imp_df["Importance (%)"] = imp_df["importance"] * 100
                imp_df = imp_df.sort_values(by="Importance (%)", ascending=True)
                
                chart_data = imp_df.set_index("feature")["Importance (%)"]
                st.bar_chart(chart_data)

            # Why it works explanation card
            st.markdown("---")
            st.markdown("#### 3. Why Flow Metadata Detects Encrypted Threats")
            st.markdown("""
            - **Packet Length Distributions (`Bwd Packet Length Min`, `Packet Length Mean`)**: Attacks like PortScan and DoS produce uniform, tiny packet lengths, whereas human HTTPS browsing creates heavy, skewed packet distributions.
            - **Timing & Inter-Arrivals (`Idle Mean`, `Active Std`)**: Automated botnets and DoS tools transmit packets in rigid, repetitive clock intervals, unlike real human users who exhibit bursty, irregular pauses.
            - **TCP State Metadata (`PSH Flag Count`, `Init Win Bytes`)**: Malicious scanners and probes exhibit anomalous TCP flag combinations during handshakes before encrypted sessions fully negotiate.
            """)
