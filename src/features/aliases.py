"""
CICFlowMeter Column Alias Normalization

Maps common column name variations found across different CICFlowMeter outputs
and datasets to the canonical feature names used in this project.
"""

from typing import Dict, List, Optional
import re

# ─── Alias Map ──────────────────────────────────────────────────────────────
# Maps variant column names → canonical column name.
# Keys are lowercase-stripped for matching; values are the exact canonical name.

_ALIAS_MAP: Dict[str, str] = {
    # Destination Port
    "destination port": "Destination Port",
    "dst port": "Destination Port",
    "dst_port": "Destination Port",
    "dport": "Destination Port",
    
    # Flow Duration
    "flow duration": "Flow Duration",
    "flow_duration": "Flow Duration",
    "duration": "Flow Duration",
    
    # Total Fwd Packets
    "total fwd packets": "Total Fwd Packets",
    "total fwd packet": "Total Fwd Packets",
    "total_fwd_packets": "Total Fwd Packets",
    "fwd packets": "Total Fwd Packets",
    
    # Total Backward Packets
    "total backward packets": "Total Backward Packets",
    "total bwd packets": "Total Backward Packets",
    "total_backward_packets": "Total Backward Packets",
    "bwd packets": "Total Backward Packets",
    
    # Total Length of Fwd Packets
    "total length of fwd packets": "Total Length of Fwd Packets",
    "total length of fwd packet": "Total Length of Fwd Packets",
    "totlen fwd pkts": "Total Length of Fwd Packets",
    
    # Total Length of Bwd Packets
    "total length of bwd packets": "Total Length of Bwd Packets",
    "total length of bwd packet": "Total Length of Bwd Packets",
    "totlen bwd pkts": "Total Length of Bwd Packets",
    
    # Fwd Packet Length
    "fwd packet length max": "Fwd Packet Length Max",
    "fwd pkt len max": "Fwd Packet Length Max",
    "fwd packet length min": "Fwd Packet Length Min",
    "fwd pkt len min": "Fwd Packet Length Min",
    "fwd packet length mean": "Fwd Packet Length Mean",
    "fwd pkt len mean": "Fwd Packet Length Mean",
    "fwd packet length std": "Fwd Packet Length Std",
    "fwd pkt len std": "Fwd Packet Length Std",
    
    # Bwd Packet Length
    "bwd packet length max": "Bwd Packet Length Max",
    "bwd pkt len max": "Bwd Packet Length Max",
    "bwd packet length min": "Bwd Packet Length Min",
    "bwd pkt len min": "Bwd Packet Length Min",
    "bwd packet length mean": "Bwd Packet Length Mean",
    "bwd pkt len mean": "Bwd Packet Length Mean",
    "bwd packet length std": "Bwd Packet Length Std",
    "bwd pkt len std": "Bwd Packet Length Std",
    
    # Flow Bytes/s variants
    "flow bytes/s": "Flow Bytes/s",
    "flow byts/s": "Flow Bytes/s",
    "flow_bytes/s": "Flow Bytes/s",
    "flow bytes per second": "Flow Bytes/s",
    
    # Flow Packets/s variants
    "flow packets/s": "Flow Packets/s",
    "flow pkts/s": "Flow Packets/s",
    "flow_packets/s": "Flow Packets/s",
    
    # Flow IAT
    "flow iat mean": "Flow IAT Mean",
    "flow iat std": "Flow IAT Std",
    "flow iat max": "Flow IAT Max",
    "flow iat min": "Flow IAT Min",
    
    # Fwd IAT
    "fwd iat total": "Fwd IAT Total",
    "fwd iat tot": "Fwd IAT Total",
    "fwd iat mean": "Fwd IAT Mean",
    "fwd iat std": "Fwd IAT Std",
    "fwd iat max": "Fwd IAT Max",
    "fwd iat min": "Fwd IAT Min",
    
    # Bwd IAT
    "bwd iat total": "Bwd IAT Total",
    "bwd iat tot": "Bwd IAT Total",
    "bwd iat mean": "Bwd IAT Mean",
    "bwd iat std": "Bwd IAT Std",
    "bwd iat max": "Bwd IAT Max",
    "bwd iat min": "Bwd IAT Min",
    
    # PSH Flags
    "fwd psh flags": "Fwd PSH Flags",
    "fwd_psh_flags": "Fwd PSH Flags",
    
    # Header Length
    "fwd header length": "Fwd Header Length",
    "fwd header len": "Fwd Header Length",
    "fwd_header_length": "Fwd Header Length",
    "bwd header length": "Bwd Header Length",
    "bwd header len": "Bwd Header Length",
    "bwd_header_length": "Bwd Header Length",
    
    # Packets/s
    "fwd packets/s": "Fwd Packets/s",
    "fwd pkts/s": "Fwd Packets/s",
    "bwd packets/s": "Bwd Packets/s",
    "bwd pkts/s": "Bwd Packets/s",
    
    # Packet Length
    "min packet length": "Min Packet Length",
    "pkt len min": "Min Packet Length",
    "max packet length": "Max Packet Length",
    "pkt len max": "Max Packet Length",
    "packet length mean": "Packet Length Mean",
    "pkt len mean": "Packet Length Mean",
    "packet length std": "Packet Length Std",
    "pkt len std": "Packet Length Std",
    "packet length variance": "Packet Length Variance",
    "pkt len var": "Packet Length Variance",
    
    # Flags
    "fin flag count": "FIN Flag Count",
    "fin flag cnt": "FIN Flag Count",
    "syn flag count": "SYN Flag Count",
    "syn flag cnt": "SYN Flag Count",
    "psh flag count": "PSH Flag Count",
    "psh flag cnt": "PSH Flag Count",
    "ack flag count": "ACK Flag Count",
    "ack flag cnt": "ACK Flag Count",
    "urg flag count": "URG Flag Count",
    "urg flag cnt": "URG Flag Count",
    "rst flag count": "RST Flag Count",
    "rst flag cnt": "RST Flag Count",
    
    # Ratios and averages
    "down/up ratio": "Down/Up Ratio",
    "down_up_ratio": "Down/Up Ratio",
    "average packet size": "Average Packet Size",
    "pkt size avg": "Average Packet Size",
    "avg fwd segment size": "Avg Fwd Segment Size",
    "fwd seg size avg": "Avg Fwd Segment Size",
    "avg bwd segment size": "Avg Bwd Segment Size",
    "bwd seg size avg": "Avg Bwd Segment Size",
    
    # Subflow
    "subflow fwd packets": "Subflow Fwd Packets",
    "subflow fwd pkts": "Subflow Fwd Packets",
    "subflow fwd bytes": "Subflow Fwd Bytes",
    "subflow fwd byts": "Subflow Fwd Bytes",
    "subflow bwd packets": "Subflow Bwd Packets",
    "subflow bwd pkts": "Subflow Bwd Packets",
    "subflow bwd bytes": "Subflow Bwd Bytes",
    "subflow bwd byts": "Subflow Bwd Bytes",
    
    # Init Window
    "init_win_bytes_forward": "Init_Win_bytes_forward",
    "init fwd win byts": "Init_Win_bytes_forward",
    "init win bytes fwd": "Init_Win_bytes_forward",
    "init_win_bytes_backward": "Init_Win_bytes_backward",
    "init bwd win byts": "Init_Win_bytes_backward",
    "init win bytes bwd": "Init_Win_bytes_backward",
    
    # act_data_pkt_fwd
    "act_data_pkt_fwd": "act_data_pkt_fwd",
    "fwd act data pkts": "act_data_pkt_fwd",
    
    # min_seg_size_forward
    "min_seg_size_forward": "min_seg_size_forward",
    "fwd seg size min": "min_seg_size_forward",
    
    # Active stats
    "active mean": "Active Mean",
    "active_mean": "Active Mean",
    "active std": "Active Std",
    "active_std": "Active Std",
    "active max": "Active Max",
    "active_max": "Active Max",
    "active min": "Active Min",
    "active_min": "Active Min",
    
    # Idle stats
    "idle mean": "Idle Mean",
    "idle_mean": "Idle Mean",
    "idle std": "Idle Std",
    "idle_std": "Idle Std",
    "idle max": "Idle Max",
    "idle_max": "Idle Max",
    "idle min": "Idle Min",
    "idle_min": "Idle Min",
}


def normalize_column_name(col: str) -> str:
    """
    Normalize a column name to its canonical form.
    
    1. Strip whitespace
    2. Look up in alias map (case-insensitive)
    3. If no alias found, return stripped original
    
    Args:
        col: Original column name
        
    Returns:
        Canonical column name if alias found, else stripped original
    """
    stripped = col.strip()
    key = stripped.lower().strip()
    
    # Direct lookup
    if key in _ALIAS_MAP:
        return _ALIAS_MAP[key]
    
    # Try with underscores replaced by spaces
    key_spaces = key.replace("_", " ")
    if key_spaces in _ALIAS_MAP:
        return _ALIAS_MAP[key_spaces]
    
    return stripped


def normalize_columns(columns: List[str]) -> Dict[str, str]:
    """
    Normalize a list of column names, returning a mapping of original → canonical.
    
    Args:
        columns: List of original column names
        
    Returns:
        Dict mapping original column name → canonical column name
    """
    mapping = {}
    for col in columns:
        mapping[col] = normalize_column_name(col)
    return mapping


def apply_column_normalization(columns: List[str]) -> List[str]:
    """
    Apply normalization to a list of column names.
    
    Args:
        columns: List of original column names
        
    Returns:
        List of normalized column names
    """
    return [normalize_column_name(c) for c in columns]
