"""
AI-Based Encrypted Traffic Threat Detection
PCAP Flow Feature Extractor Module
Parses .pcap / .pcapng files into bidirectional network flows and computes
statistical metadata features matching the CICIDS2017 schema without payload decryption.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scapy.all import rdpcap, PcapReader, IP, IPv6, TCP, UDP

FEATURES_PATH = "data/feature_columns.json"

def get_feature_columns():
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, "r") as f:
            return json.load(f)
    return []

def extract_flows_from_pcap(pcap_source, max_packets=50000):
    """
    Extracts bidirectional network flows and computes flow features from a pcap file or path.
    :param pcap_source: File path (str) or file-like stream / bytes
    :param max_packets: Maximum packets to process to prevent memory overload
    :return: (pandas.DataFrame of flow features, list of flow metadata dicts)
    """
    feature_columns = get_feature_columns()
    
    # Check if pcap_source is a file path or a temporary file object
    temp_file_created = False
    if hasattr(pcap_source, "read"):
        # Save Streamlit UploadedFile to temporary file for scapy PcapReader
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
            pcap_source.seek(0)
            tmp.write(pcap_source.read())
            pcap_path = tmp.name
            temp_file_created = True
    else:
        pcap_path = pcap_source

    try:
        # Group packets into flows
        # Flow key: (min_endpoint, max_endpoint, protocol)
        flows = {}
        pkt_count = 0
        
        with PcapReader(pcap_path) as reader:
            for pkt in reader:
                pkt_count += 1
                if pkt_count > max_packets:
                    break
                    
                if not (IP in pkt or IPv6 in pkt):
                    continue
                    
                ip_layer = pkt[IP] if IP in pkt else pkt[IPv6]
                src_ip = ip_layer.src
                dst_ip = ip_layer.dst
                proto = ip_layer.proto if IP in pkt else ip_layer.nh
                
                if TCP in pkt:
                    transport = pkt[TCP]
                    src_port = transport.sport
                    dst_port = transport.dport
                    is_tcp = True
                elif UDP in pkt:
                    transport = pkt[UDP]
                    src_port = transport.sport
                    dst_port = transport.dport
                    is_tcp = False
                else:
                    continue
                    
                ep1 = (src_ip, src_port)
                ep2 = (dst_ip, dst_port)
                
                # Canonical flow identifier
                if ep1 <= ep2:
                    flow_key = (ep1, ep2, proto)
                else:
                    flow_key = (ep2, ep1, proto)
                    
                ts = float(pkt.time)
                pkt_len = len(pkt)
                
                if flow_key not in flows:
                    flows[flow_key] = {
                        "fwd_src_ip": src_ip,
                        "fwd_dst_ip": dst_ip,
                        "fwd_src_port": src_port,
                        "fwd_dst_port": dst_port,
                        "protocol": proto,
                        "start_time": ts,
                        "last_time": ts,
                        "fwd_packets": [],
                        "bwd_packets": [],
                        "all_packets": []
                    }
                    
                flow = flows[flow_key]
                flow["last_time"] = ts
                
                is_forward = (src_ip == flow["fwd_src_ip"] and src_port == flow["fwd_src_port"])
                
                header_len = 0
                payload_len = 0
                tcp_flags = {"FIN": 0, "SYN": 0, "RST": 0, "PSH": 0, "ACK": 0, "URG": 0}
                win_size = -1
                
                if is_tcp:
                    header_len = (ip_layer.ihl * 4 if IP in pkt else 40) + (transport.dataofs * 4)
                    payload_len = max(0, pkt_len - header_len)
                    win_size = transport.window
                    flags_val = int(transport.flags)
                    tcp_flags["FIN"] = 1 if flags_val & 0x01 else 0
                    tcp_flags["SYN"] = 1 if flags_val & 0x02 else 0
                    tcp_flags["RST"] = 1 if flags_val & 0x04 else 0
                    tcp_flags["PSH"] = 1 if flags_val & 0x08 else 0
                    tcp_flags["ACK"] = 1 if flags_val & 0x10 else 0
                    tcp_flags["URG"] = 1 if flags_val & 0x20 else 0
                else:
                    header_len = (ip_layer.ihl * 4 if IP in pkt else 40) + 8
                    payload_len = max(0, pkt_len - header_len)
                    
                pkt_data = {
                    "time": ts,
                    "len": pkt_len,
                    "header_len": header_len,
                    "payload_len": payload_len,
                    "is_tcp": is_tcp,
                    "tcp_flags": tcp_flags,
                    "win_size": win_size
                }
                
                flow["all_packets"].append(pkt_data)
                if is_forward:
                    flow["fwd_packets"].append(pkt_data)
                else:
                    flow["bwd_packets"].append(pkt_data)

    finally:
        if temp_file_created and os.path.exists(pcap_path):
            try:
                os.remove(pcap_path)
            except Exception:
                pass

    if not flows:
        return pd.DataFrame(), []

    # Calculate flow features for each flow
    flow_rows = []
    metadata_rows = []
    
    for flow_key, f in flows.items():
        all_pkts = f["all_packets"]
        fwd_pkts = f["fwd_packets"]
        bwd_pkts = f["bwd_packets"]
        
        tot_pkts = len(all_pkts)
        tot_fwd = len(fwd_pkts)
        tot_bwd = len(bwd_pkts)
        
        # Duration in microseconds
        duration_sec = f["last_time"] - f["start_time"]
        duration_us = max(0.0, duration_sec * 1e6)
        
        # Packet length statistics
        all_lens = [p["len"] for p in all_pkts]
        fwd_lens = [p["len"] for p in fwd_pkts] if fwd_pkts else [0]
        bwd_lens = [p["len"] for p in bwd_pkts] if bwd_pkts else [0]
        
        tot_len_fwd = sum(fwd_lens)
        tot_len_bwd = sum(bwd_lens) if bwd_pkts else 0
        
        fwd_max = float(np.max(fwd_lens)) if fwd_pkts else 0.0
        fwd_min = float(np.min(fwd_lens)) if fwd_pkts else 0.0
        fwd_mean = float(np.mean(fwd_lens)) if fwd_pkts else 0.0
        fwd_std = float(np.std(fwd_lens)) if len(fwd_pkts) > 1 else 0.0
        
        bwd_max = float(np.max(bwd_lens)) if bwd_pkts else 0.0
        bwd_min = float(np.min(bwd_lens)) if bwd_pkts else 0.0
        bwd_mean = float(np.mean(bwd_lens)) if bwd_pkts else 0.0
        bwd_std = float(np.std(bwd_lens)) if len(bwd_pkts) > 1 else 0.0
        
        pkt_max = float(np.max(all_lens))
        pkt_min = float(np.min(all_lens))
        pkt_mean = float(np.mean(all_lens))
        pkt_std = float(np.std(all_lens)) if tot_pkts > 1 else 0.0
        pkt_var = float(np.var(all_lens)) if tot_pkts > 1 else 0.0
        
        # Rates
        dur_div = duration_sec if duration_sec > 0 else 1e-6
        flow_bytes_s = (tot_len_fwd + tot_len_bwd) / dur_div
        flow_pkts_s = tot_pkts / dur_div
        fwd_pkts_s = tot_fwd / dur_div
        bwd_pkts_s = tot_bwd / dur_div
        
        # Inter-arrival times (microseconds)
        def calc_iat(pkts):
            if len(pkts) < 2:
                return 0.0, 0.0, 0.0, 0.0, 0.0
            times = [p["time"] for p in pkts]
            diffs = [(times[i] - times[i - 1]) * 1e6 for i in range(1, len(times))]
            return sum(diffs), float(np.mean(diffs)), float(np.std(diffs)), float(np.max(diffs)), float(np.min(diffs))
            
        _, flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min = calc_iat(all_pkts)
        fwd_iat_tot, fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min = calc_iat(fwd_pkts)
        bwd_iat_tot, bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min = calc_iat(bwd_pkts)
        
        # Flags
        fin_cnt = sum(p["tcp_flags"]["FIN"] for p in all_pkts)
        syn_cnt = sum(p["tcp_flags"]["SYN"] for p in all_pkts)
        psh_cnt = sum(p["tcp_flags"]["PSH"] for p in all_pkts)
        ack_cnt = sum(p["tcp_flags"]["ACK"] for p in all_pkts)
        urg_cnt = sum(p["tcp_flags"]["URG"] for p in all_pkts)
        fwd_psh_flags = sum(p["tcp_flags"]["PSH"] for p in fwd_pkts)
        
        # Headers & Windows
        fwd_hdr_len = sum(p["header_len"] for p in fwd_pkts)
        bwd_hdr_len = sum(p["header_len"] for p in bwd_pkts)
        
        init_win_fwd = -1
        for p in fwd_pkts:
            if p["win_size"] >= 0:
                init_win_fwd = p["win_size"]
                break
                
        init_win_bwd = -1
        for p in bwd_pkts:
            if p["win_size"] >= 0:
                init_win_bwd = p["win_size"]
                break
                
        act_data_fwd = sum(1 for p in fwd_pkts if p["payload_len"] > 0)
        min_seg_fwd = min((p["header_len"] for p in fwd_pkts), default=20)
        
        down_up_ratio = (tot_bwd / tot_fwd) if tot_fwd > 0 else 0.0
        avg_pkt_size = sum(all_lens) / tot_pkts if tot_pkts > 0 else 0.0
        
        # Idle / Active calculation (threshold = 5.0 seconds)
        idle_thresh_us = 5.0 * 1e6
        all_times = [p["time"] for p in all_pkts]
        diffs = [(all_times[i] - all_times[i - 1]) * 1e6 for i in range(1, len(all_times))]
        idle_times = [d for d in diffs if d >= idle_thresh_us]
        
        if idle_times:
            idle_mean = float(np.mean(idle_times))
            idle_std = float(np.std(idle_times))
            idle_max = float(np.max(idle_times))
            idle_min = float(np.min(idle_times))
        else:
            idle_mean = idle_std = idle_max = idle_min = 0.0
            
        active_mean = active_std = active_max = active_min = 0.0
        
        # Assemble feature dict
        row = {
            "Destination Port": f["fwd_dst_port"],
            "Flow Duration": duration_us,
            "Total Fwd Packets": tot_fwd,
            "Total Backward Packets": tot_bwd,
            "Total Length of Fwd Packets": tot_len_fwd,
            "Total Length of Bwd Packets": tot_len_bwd,
            "Fwd Packet Length Max": fwd_max,
            "Fwd Packet Length Min": fwd_min,
            "Fwd Packet Length Mean": fwd_mean,
            "Fwd Packet Length Std": fwd_std,
            "Bwd Packet Length Max": bwd_max,
            "Bwd Packet Length Min": bwd_min,
            "Bwd Packet Length Mean": bwd_mean,
            "Bwd Packet Length Std": bwd_std,
            "Flow Bytes/s": flow_bytes_s,
            "Flow Packets/s": flow_pkts_s,
            "Flow IAT Mean": flow_iat_mean,
            "Flow IAT Std": flow_iat_std,
            "Flow IAT Max": flow_iat_max,
            "Flow IAT Min": flow_iat_min,
            "Fwd IAT Total": fwd_iat_tot,
            "Fwd IAT Mean": fwd_iat_mean,
            "Fwd IAT Std": fwd_iat_std,
            "Fwd IAT Max": fwd_iat_max,
            "Fwd IAT Min": fwd_iat_min,
            "Bwd IAT Total": bwd_iat_tot,
            "Bwd IAT Mean": bwd_iat_mean,
            "Bwd IAT Std": bwd_iat_std,
            "Bwd IAT Max": bwd_iat_max,
            "Bwd IAT Min": bwd_iat_min,
            "Fwd PSH Flags": fwd_psh_flags,
            "Fwd Header Length": fwd_hdr_len,
            "Bwd Header Length": bwd_hdr_len,
            "Fwd Packets/s": fwd_pkts_s,
            "Bwd Packets/s": bwd_pkts_s,
            "Min Packet Length": pkt_min,
            "Max Packet Length": pkt_max,
            "Packet Length Mean": pkt_mean,
            "Packet Length Std": pkt_std,
            "Packet Length Variance": pkt_var,
            "FIN Flag Count": fin_cnt,
            "SYN Flag Count": syn_cnt,
            "PSH Flag Count": psh_cnt,
            "ACK Flag Count": ack_cnt,
            "URG Flag Count": urg_cnt,
            "Down/Up Ratio": down_up_ratio,
            "Average Packet Size": avg_pkt_size,
            "Avg Fwd Segment Size": fwd_mean,
            "Avg Bwd Segment Size": bwd_mean,
            "Subflow Fwd Packets": tot_fwd,
            "Subflow Fwd Bytes": tot_len_fwd,
            "Subflow Bwd Packets": tot_bwd,
            "Subflow Bwd Bytes": tot_len_bwd,
            "Init_Win_bytes_forward": init_win_fwd,
            "Init_Win_bytes_backward": init_win_bwd,
            "act_data_pkt_fwd": act_data_fwd,
            "min_seg_size_forward": min_seg_fwd,
            "Active Mean": active_mean,
            "Active Std": active_std,
            "Active Max": active_max,
            "Active Min": active_min,
            "Idle Mean": idle_mean,
            "Idle Std": idle_std,
            "Idle Max": idle_max,
            "Idle Min": idle_min
        }
        
        meta = {
            "Source_IP": f["fwd_src_ip"],
            "Destination_IP": f["fwd_dst_ip"],
            "Source_Port": f["fwd_src_port"],
            "Destination_Port": f["fwd_dst_port"],
            "Protocol": "TCP" if f["protocol"] == 6 else ("UDP" if f["protocol"] == 17 else str(f["protocol"])),
            "Packets": tot_pkts,
            "Bytes": tot_len_fwd + tot_len_bwd
        }
        
        flow_rows.append(row)
        metadata_rows.append(meta)

    features_df = pd.DataFrame(flow_rows)
    # Ensure all required features are present
    for col in feature_columns:
        if col not in features_df.columns:
            features_df[col] = 0.0
            
    # Include metadata columns at front for display
    meta_df = pd.DataFrame(metadata_rows)
    combined_df = pd.concat([meta_df, features_df], axis=1)
    
    return combined_df, metadata_rows

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pcap_file = sys.argv[1]
        print(f"Extracting flows from: {pcap_file}")
        df, meta = extract_flows_from_pcap(pcap_file)
        print(f"Extracted {len(df)} network flows.")
        print(df[["Source_IP", "Destination_IP", "Destination_Port", "Protocol", "Packets"]].head())
    else:
        print("Usage: python src/pcap_extractor.py <path_to_pcap>")
