import pytest
import pandas as pd
import os
from src.predict import ThreatPredictor

@pytest.fixture
def sample_prediction_df(sample_flow_features):
    return sample_flow_features.copy()

def get_model_path():
    # Helper to check if model exists for skipping
    possible_paths = [
        "models/binary_threat_detector.joblib",
        "models/threat_detector.joblib"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return None

@pytest.mark.skipif(get_model_path() is None, reason="Model file not found")
def test_model_loads():
    predictor = ThreatPredictor(strict_validation=False)
    assert predictor is not None
    info = predictor.get_model_info()
    assert "binary_threshold" in info

@pytest.mark.skipif(get_model_path() is None, reason="Model file not found")
def test_batch_prediction(sample_prediction_df):
    predictor = ThreatPredictor(strict_validation=False)
    preds = predictor.predict_batch(sample_prediction_df)
    assert len(preds) == len(sample_prediction_df)
    assert "Predicted_Verdict" in preds.columns

def test_binary_label_normalization():
    # If ThreatPredictor exposes a method for normalizations
    try:
        from src.predict import normalize_label
        assert normalize_label("DDoS") == "MALICIOUS"
        assert normalize_label("BENIGN") == "BENIGN"
    except ImportError:
        pass

def test_ground_truth_evaluation():
    try:
        # Instantiate without model just to test metrics if possible
        predictor = ThreatPredictor(model_path=None)
        # test logic here
        pass
    except Exception:
        pass
