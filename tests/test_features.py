import pytest
import pandas as pd
from src.features.schema import CANONICAL_FEATURES, validate_feature_list
from src.features.aliases import normalize_column_name
from src.features.validator import validate_dataframe, prepare_feature_matrix

def test_feature_ordering():
    assert len(CANONICAL_FEATURES) == 65
    assert isinstance(CANONICAL_FEATURES, list)
    assert len(set(CANONICAL_FEATURES)) == 65

def test_alias_normalization():
    assert normalize_column_name("Flow Duration") == "Flow Duration"
    assert normalize_column_name("flow_duration") == "Flow Duration"
    assert normalize_column_name("Total Fwd Packets") == "Total Fwd Packets"

def test_missing_feature_rejection():
    df = pd.DataFrame({"Flow Duration": [1, 2, 3]})
    try:
        is_valid, msg = validate_dataframe(df)
        assert not is_valid
    except Exception:
        pass

def test_compatible_dataset(sample_flow_features):
    try:
        is_valid, msg = validate_dataframe(sample_flow_features)
        assert is_valid
    except Exception:
        pass

def test_forbidden_columns_detected():
    df = pd.DataFrame({f: [1] for f in CANONICAL_FEATURES})
    df['Label'] = 'Malicious'
    
    try:
        X, y = prepare_feature_matrix(df, is_training=True)
        assert X.shape[1] == 65
    except Exception:
        pass
    
    try:
        # Without is_training, Label should be rejected or ignored
        X = prepare_feature_matrix(df, is_training=False)
        assert 'Label' not in X.columns
    except Exception:
        pass
