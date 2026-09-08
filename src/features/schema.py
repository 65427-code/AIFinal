"""
Canonical Feature Schema for AI-Based Encrypted Traffic Threat Detection

Defines the ordered list of 65 CICFlowMeter-compatible flow features used 
consistently across training and inference. This is the single source of truth
for feature names and ordering.
"""

from typing import List

SCHEMA_VERSION = "2.0"

# ─── Canonical 65 features in exact training order ──────────────────────────
# These match the CICFlowMeter output schema used in CICIDS2017,
# after removing zero-variance/duplicate columns.

CANONICAL_FEATURES: List[str] = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]

# Additional DoHBrw features (different schema — mapped separately)
DOHBRW_FEATURES: List[str] = [
    "Duration",
    "FlowBytesSent",
    "FlowSentRate",
    "FlowBytesReceived",
    "FlowReceivedRate",
    "PacketLengthVariance",
    "PacketLengthStandardDeviation",
    "PacketLengthMean",
    "PacketLengthMedian",
    "PacketLengthMode",
    "PacketLengthSkewFromMedian",
    "PacketLengthSkewFromMode",
    "PacketLengthCoefficientofVariation",
    "PacketTimeVariance",
    "PacketTimeStandardDeviation",
    "PacketTimeMean",
    "PacketTimeMedian",
    "PacketTimeMode",
    "PacketTimeSkewFromMedian",
    "PacketTimeSkewFromMode",
    "PacketTimeCoefficientofVariation",
    "ResponseTimeTimeVariance",
    "ResponseTimeTimeStandardDeviation",
    "ResponseTimeTimeMean",
    "ResponseTimeTimeMedian",
    "ResponseTimeTimeMode",
    "ResponseTimeTimeSkewFromMedian",
    "ResponseTimeTimeSkewFromMode",
    "ResponseTimeTimeCoefficientofVariation",
]

# Mapping from DoHBrw features to closest canonical features (partial mapping)
# Only features with reasonably similar semantics are mapped.
DOHBRW_TO_CANONICAL_MAP = {
    "Duration": "Flow Duration",  # Note: DoHBrw is in seconds, canonical in microseconds
    "PacketLengthVariance": "Packet Length Variance",
    "PacketLengthStandardDeviation": "Packet Length Std",
    "PacketLengthMean": "Packet Length Mean",
}

# ─── Metadata columns (NOT training features) ──────────────────────────────

METADATA_COLUMNS = [
    "Source_IP", "Destination_IP", "Source_Port", "Destination_Port_Meta",
    "Protocol", "Packets", "Bytes",
    "Dataset", "Capture_ID", "Original_Label", "Binary_Label",
    "Attack_Family", "Attack_Category", "Specific_Family",
    "Is_Encrypted", "Flow_ID",
]

# Columns that must NEVER be used as training features
FORBIDDEN_TRAINING_COLUMNS = [
    "Source_IP", "Destination_IP", "Source_Port",
    "Protocol", "Dataset", "Capture_ID", "Original_Label",
    "Binary_Label", "Attack_Family", "Attack_Category",
    "Specific_Family", "Label", "label", "Class", "class",
    "Expected_Verdict", "Is_Malicious", "Is_Encrypted",
    "Flow_ID", "Predicted_Verdict", "Predicted_Category",
    "Threat_Confidence_%", "Risk_Level", "Filename",
]


def get_feature_count() -> int:
    """Return the number of canonical features."""
    return len(CANONICAL_FEATURES)


def get_feature_index(feature_name: str) -> int:
    """Return the index of a feature in the canonical order."""
    return CANONICAL_FEATURES.index(feature_name)


def validate_feature_list(features: List[str]) -> dict:
    """
    Validate a list of features against the canonical schema.
    
    Returns:
        dict with keys: compatible (bool), present, missing, extra, 
        forbidden (features that should never be in training)
    """
    canonical_set = set(CANONICAL_FEATURES)
    feature_set = set(features)
    
    present = canonical_set & feature_set
    missing = canonical_set - feature_set
    extra = feature_set - canonical_set
    forbidden = extra & set(FORBIDDEN_TRAINING_COLUMNS)
    
    return {
        "compatible": len(missing) == 0,
        "present": sorted(present),
        "missing": sorted(missing),
        "extra": sorted(extra),
        "forbidden": sorted(forbidden),
        "expected_count": len(CANONICAL_FEATURES),
        "present_count": len(present),
        "missing_count": len(missing),
    }
