import pytest
import pandas as pd
import numpy as np
try:
    from src.features.schema import CANONICAL_FEATURES
except ImportError:
    CANONICAL_FEATURES = [f"feature_{i}" for i in range(65)]

@pytest.fixture
def mock_packet_data():
    """Returns a dictionary with mock packet data."""
    return {
        'src_ip': '192.168.1.100',
        'dst_ip': '10.0.0.1',
        'src_port': 54321,
        'dst_port': 80,
        'protocol': 6, # TCP
        'is_tcp': True,
        'pkt_len': 100,
        'payload_len': 46,
        'header_len': 54,
        'timestamp': 1000.0,
        'tcp_flags': {'FIN': 0, 'SYN': 0, 'RST': 0, 'PSH': 0, 'ACK': 0, 'URG': 0},
        'win_size': 8192
    }

@pytest.fixture
def sample_flow_features():
    """Returns a DataFrame with 1 row containing all 65 canonical features."""
    data = {feature: np.random.rand() for feature in CANONICAL_FEATURES}
    return pd.DataFrame([data])
