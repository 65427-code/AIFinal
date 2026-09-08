"""
AI-Based Encrypted Traffic Threat Detection
Central Configuration Module

Defines dataset configurations, label mappings, paths, and model settings.
No hardcoded absolute paths — all paths are relative or configurable via CLI.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ─── Project Paths (relative to project root) ───────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
COMBINED_DIR = os.path.join(DATA_DIR, "combined")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Legacy paths for backward compatibility
LEGACY_ARCHIVE_DIR = os.path.join(PROJECT_ROOT, "archive")
LEGACY_USTC_DIR = os.path.join(PROJECT_ROOT, "USTC-TFC2016-master")
LEGACY_DOHBRW_DIR = os.path.join(PROJECT_ROOT, "CIRA-CIC-DoHBrw-2020")

# ─── Feature Schema ─────────────────────────────────────────────────────────

FEATURE_SCHEMA_VERSION = "2.0"
NUM_CANONICAL_FEATURES = 65

# ─── Flow Reconstruction Defaults ───────────────────────────────────────────

DEFAULT_IDLE_TIMEOUT = 120.0        # seconds
DEFAULT_ACTIVE_TIMEOUT = 1800.0     # seconds (30 min)
DEFAULT_UDP_TIMEOUT = 120.0         # seconds
ACTIVE_IDLE_THRESHOLD = 1.0         # seconds — gap above this is "idle"
DEFAULT_MAX_PACKETS = None          # No limit by default

# ─── Model Defaults ─────────────────────────────────────────────────────────

DEFAULT_BINARY_THRESHOLD = 0.5
ATTACK_CATEGORY_CONFIDENCE_THRESHOLD = 0.30
OOD_CONTAMINATION = 0.05

# ─── Model Artifact Paths ───────────────────────────────────────────────────

BINARY_DETECTOR_PATH = os.path.join(MODELS_DIR, "binary_threat_detector.joblib")
BINARY_SCALER_PATH = os.path.join(MODELS_DIR, "binary_scaler.joblib")
ATTACK_CLASSIFIER_PATH = os.path.join(MODELS_DIR, "attack_classifier.joblib")
ATTACK_LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "attack_label_encoder.joblib")
ANOMALY_DETECTOR_PATH = os.path.join(MODELS_DIR, "anomaly_detector.joblib")
FEATURE_SCHEMA_PATH = os.path.join(MODELS_DIR, "feature_schema.json")
MODEL_METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
EVALUATION_RESULTS_PATH = os.path.join(MODELS_DIR, "evaluation_results.json")
FEATURE_IMPORTANCE_PATH = os.path.join(MODELS_DIR, "feature_importance.json")

# Legacy model paths
LEGACY_MODEL_PATH = os.path.join(MODELS_DIR, "threat_detector.joblib")
LEGACY_SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
LEGACY_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")
LEGACY_FEATURES_PATH = os.path.join(DATA_DIR, "feature_columns.json")

# ─── Dataset Configurations ─────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    """Configuration for a single dataset source."""
    name: str
    description: str
    default_path: str
    source_type: str  # "csv", "pcap", "parquet"
    label_column: Optional[str] = None
    benign_labels: List[str] = field(default_factory=list)
    malicious_labels: List[str] = field(default_factory=list)
    label_mapping: Dict[str, dict] = field(default_factory=dict)


CICIDS2017_CONFIG = DatasetConfig(
    name="CICIDS2017",
    description="Canadian Institute for Cybersecurity IDS 2017 dataset (CICFlowMeter CSVs)",
    default_path=LEGACY_ARCHIVE_DIR,
    source_type="csv",
    label_column="Label",
    benign_labels=["BENIGN"],
    malicious_labels=[],  # everything not BENIGN
    label_mapping={
        "BENIGN": {"binary": "BENIGN", "family": None, "category": "Benign"},
        "DDoS": {"binary": "MALICIOUS", "family": "DDoS", "category": "DDoS"},
        "PortScan": {"binary": "MALICIOUS", "family": "PortScan", "category": "PortScan"},
        "DoS Hulk": {"binary": "MALICIOUS", "family": "DoS", "category": "DoS"},
        "DoS GoldenEye": {"binary": "MALICIOUS", "family": "DoS", "category": "DoS"},
        "DoS slowloris": {"binary": "MALICIOUS", "family": "DoS", "category": "DoS"},
        "DoS Slowhttptest": {"binary": "MALICIOUS", "family": "DoS", "category": "DoS"},
        "Heartbleed": {"binary": "MALICIOUS", "family": "DoS", "category": "DoS"},
        "FTP-Patator": {"binary": "MALICIOUS", "family": "Brute Force", "category": "Brute Force"},
        "SSH-Patator": {"binary": "MALICIOUS", "family": "Brute Force", "category": "Brute Force"},
        "Bot": {"binary": "MALICIOUS", "family": "Botnet", "category": "Botnet"},
        "Web Attack – Brute Force": {"binary": "MALICIOUS", "family": "Web Attack", "category": "Web Attack"},
        "Web Attack – XSS": {"binary": "MALICIOUS", "family": "Web Attack", "category": "Web Attack"},
        "Web Attack – Sql Injection": {"binary": "MALICIOUS", "family": "Web Attack", "category": "Web Attack"},
        "Infiltration": {"binary": "MALICIOUS", "family": "Infiltration", "category": "Infiltration"},
    },
)

USTC_TFC2016_CONFIG = DatasetConfig(
    name="USTC-TFC2016",
    description="USTC Traffic Flow Classification 2016 dataset (raw PCAPs)",
    default_path=LEGACY_USTC_DIR,
    source_type="pcap",
    benign_labels=["BitTorrent", "FTP", "Facetime", "Gmail", "MySQL", "Outlook", "Skype", "WorldOfWarcraft"],
    malicious_labels=["Cridex", "Geodo", "Htbot", "Miuref", "Neris", "Nsis-ay", "Shifu", "Tinba", "Virut", "Zeus"],
    label_mapping={},  # Built dynamically from directory structure
)

DOHBRW2020_CONFIG = DatasetConfig(
    name="CIRA-CIC-DoHBrw-2020",
    description="CIRA-CIC DNS-over-HTTPS Browser dataset (Parquet files)",
    default_path=LEGACY_DOHBRW_DIR,
    source_type="parquet",
    label_column="Label",
    benign_labels=["NonDoH", "Benign"],
    malicious_labels=["Malicious"],
    label_mapping={
        "NonDoH": {"binary": "BENIGN", "family": None, "category": "Benign HTTPS"},
        "DoH": {"binary": "BENIGN", "family": None, "category": "Benign DoH"},
        "Benign": {"binary": "BENIGN", "family": None, "category": "Benign DoH"},
        "Malicious": {"binary": "MALICIOUS", "family": "DoH Tunnel", "category": "Encrypted Tunnel"},
    },
)

ALL_DATASET_CONFIGS = {
    "CICIDS2017": CICIDS2017_CONFIG,
    "USTC-TFC2016": USTC_TFC2016_CONFIG,
    "CIRA-CIC-DoHBrw-2020": DOHBRW2020_CONFIG,
}

# ─── Attack Categories ──────────────────────────────────────────────────────

ATTACK_CATEGORIES = [
    "Benign",
    "DDoS",
    "DoS",
    "PortScan",
    "Brute Force",
    "Web Attack",
    "Botnet",
    "Infiltration",
    "Malware",
    "Encrypted Tunnel",
    "Unknown Malicious",
]

# ─── Ground Truth Column Detection ──────────────────────────────────────────

KNOWN_GROUND_TRUTH_COLUMNS = [
    "Label", "label", "Class", "class", "Attack", "Attack_Category",
    "Expected_Verdict", "Is_Malicious", "Binary_Label", "category",
]

KNOWN_BENIGN_VALUES = [
    "BENIGN", "benign", "Benign", "Normal", "normal", "0",
    "NonDoH", "legitimate", "safe",
]

KNOWN_MALICIOUS_VALUES = [
    "MALICIOUS", "malicious", "Malicious", "THREAT", "Threat",
    "threat", "attack", "Attack", "1", "anomaly", "Anomaly",
]
