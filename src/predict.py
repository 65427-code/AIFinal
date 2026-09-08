"""
AI-Based Encrypted Traffic Threat Detection
Inference Engine v2.0

Binary-first threat detection architecture:
1. Binary detector: BENIGN vs MALICIOUS
2. Attack classifier: category identification for malicious flows
3. Optional OOD/anomaly scoring

Features:
- Schema/model compatibility verification
- Column alias normalization
- Configurable threshold from model metadata
- UNKNOWN/OTHER MALICIOUS when attack confidence is low
- Legacy model graceful fallback
"""

import os
import sys
# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from src.features.schema import CANONICAL_FEATURES, validate_feature_list
from src.features.aliases import normalize_column_name
from src.features.validator import validate_dataframe, prepare_feature_matrix, FeatureValidationResult
from src.config import (
    BINARY_DETECTOR_PATH, BINARY_SCALER_PATH,
    ATTACK_CLASSIFIER_PATH, ATTACK_LABEL_ENCODER_PATH,
    ANOMALY_DETECTOR_PATH, MODEL_METADATA_PATH,
    FEATURE_SCHEMA_PATH,
    DEFAULT_BINARY_THRESHOLD, ATTACK_CATEGORY_CONFIDENCE_THRESHOLD,
    # Legacy paths
    LEGACY_MODEL_PATH, LEGACY_SCALER_PATH, LEGACY_ENCODER_PATH, LEGACY_FEATURES_PATH,
)


class ThreatPredictor:
    """
    Threat prediction engine using binary-first architecture.
    
    Primary decision: Binary detector (BENIGN vs MALICIOUS)
    Secondary: Attack classifier (category identification)
    Optional: OOD anomaly scoring
    """
    
    def __init__(
        self,
        models_dir: Optional[str] = None,
        strict_validation: bool = True,
    ):
        """
        Initialize the predictor, loading model artifacts.
        
        Tries to load v2.0 binary-first models first. Falls back to legacy
        multiclass model if v2.0 not available.
        
        Args:
            models_dir: Optional override for models directory
            strict_validation: If True, reject datasets with missing features
        """
        self.strict_validation = strict_validation
        self.is_legacy = False
        self.model_metadata = {}
        
        # Try loading v2.0 models
        binary_path = BINARY_DETECTOR_PATH
        scaler_path = BINARY_SCALER_PATH
        metadata_path = MODEL_METADATA_PATH
        
        if os.path.exists(binary_path) and os.path.exists(scaler_path):
            self._load_v2_models()
        elif os.path.exists(LEGACY_MODEL_PATH):
            self._load_legacy_models()
        else:
            raise FileNotFoundError(
                "No model artifacts found. Run 'python src/train.py' first.\n"
                f"Expected: {binary_path}\n"
                f"Or legacy: {LEGACY_MODEL_PATH}"
            )
    
    def _load_v2_models(self):
        """Load v2.0 binary-first model artifacts."""
        self.binary_detector = joblib.load(BINARY_DETECTOR_PATH)
        self.scaler = joblib.load(BINARY_SCALER_PATH)
        
        # Load metadata
        if os.path.exists(MODEL_METADATA_PATH):
            with open(MODEL_METADATA_PATH, "r") as f:
                self.model_metadata = json.load(f)
        
        self.binary_threshold = self.model_metadata.get("binary_threshold", DEFAULT_BINARY_THRESHOLD)
        self.attack_threshold = self.model_metadata.get(
            "attack_category_threshold", ATTACK_CATEGORY_CONFIDENCE_THRESHOLD
        )
        
        # Load feature schema
        if os.path.exists(FEATURE_SCHEMA_PATH):
            with open(FEATURE_SCHEMA_PATH, "r") as f:
                self.feature_names = json.load(f)
        else:
            self.feature_names = CANONICAL_FEATURES.copy()
        
        # Attack classifier (optional)
        self.attack_classifier = None
        self.attack_encoder = None
        self.attack_classes = []
        
        if os.path.exists(ATTACK_CLASSIFIER_PATH) and os.path.exists(ATTACK_LABEL_ENCODER_PATH):
            self.attack_classifier = joblib.load(ATTACK_CLASSIFIER_PATH)
            self.attack_encoder = joblib.load(ATTACK_LABEL_ENCODER_PATH)
            self.attack_classes = list(self.attack_encoder.classes_)
        
        # Anomaly detector (optional)
        self.anomaly_detector = None
        if os.path.exists(ANOMALY_DETECTOR_PATH):
            self.anomaly_detector = joblib.load(ANOMALY_DETECTOR_PATH)
        
        self.classes = ["BENIGN", "MALICIOUS"]
        self.is_legacy = False
        
        print(f"Loaded v2.0 models (threshold={self.binary_threshold:.3f}, "
              f"attack_classes={len(self.attack_classes)}, "
              f"anomaly={'yes' if self.anomaly_detector else 'no'})")
    
    def _load_legacy_models(self):
        """Load legacy multiclass model artifacts with compatibility warning."""
        self.binary_detector = joblib.load(LEGACY_MODEL_PATH)
        self.scaler = joblib.load(LEGACY_SCALER_PATH)
        
        legacy_encoder = joblib.load(LEGACY_ENCODER_PATH)
        self.legacy_classes = list(legacy_encoder.classes_)
        self.legacy_benign_idx = self.legacy_classes.index("BENIGN") if "BENIGN" in self.legacy_classes else -1
        
        if os.path.exists(LEGACY_FEATURES_PATH):
            with open(LEGACY_FEATURES_PATH, "r") as f:
                self.feature_names = json.load(f)
        else:
            self.feature_names = CANONICAL_FEATURES.copy()
        
        self.binary_threshold = DEFAULT_BINARY_THRESHOLD
        self.attack_threshold = ATTACK_CATEGORY_CONFIDENCE_THRESHOLD
        self.attack_classifier = None
        self.attack_encoder = None
        self.attack_classes = [c for c in self.legacy_classes if c != "BENIGN"]
        self.anomaly_detector = None
        self.classes = self.legacy_classes
        self.is_legacy = True
        
        print("⚠️ Loaded LEGACY model. Retraining recommended for multi-dataset support.")
    
    def _determine_risk(self, threat_prob: float) -> str:
        """Determine risk level from threat probability."""
        if threat_prob < 0.25:
            return "LOW (Normal Flow)"
        elif threat_prob < 0.60:
            return "MODERATE (Suspicious Pattern)"
        elif threat_prob < 0.85:
            return "HIGH (Likely Threat)"
        else:
            return "CRITICAL (Active Attack)"
    
    def _prepare_vector(self, flow_data: dict) -> np.ndarray:
        """Prepare a feature vector from a flow dictionary."""
        vector = []
        for feat in self.feature_names:
            val = flow_data.get(feat, 0.0)
            try:
                val = float(val)
                if np.isnan(val) or np.isinf(val):
                    val = 0.0
            except (ValueError, TypeError):
                val = 0.0
            vector.append(val)
        return np.array(vector).reshape(1, -1)
    
    def _classify_attack(self, X_scaled: np.ndarray) -> Tuple[str, float]:
        """
        Classify attack category for malicious flows.
        Returns (category, confidence).
        """
        if self.attack_classifier is None:
            return "Unknown Malicious", 0.0
        
        try:
            probs = self.attack_classifier.predict_proba(X_scaled)[0]
            max_idx = np.argmax(probs)
            max_prob = float(probs[max_idx])
            
            if max_prob >= self.attack_threshold:
                return self.attack_classes[max_idx], max_prob
            else:
                return "Unknown Malicious", max_prob
        except Exception:
            return "Unknown Malicious", 0.0
    
    def _anomaly_score(self, X_scaled: np.ndarray) -> Optional[float]:
        """Compute anomaly score if detector is available."""
        if self.anomaly_detector is None:
            return None
        try:
            # IsolationForest: negative scores = more anomalous
            score = self.anomaly_detector.score_samples(X_scaled)[0]
            return float(score)
        except Exception:
            return None
    
    def predict_single(self, flow_data: dict) -> dict:
        """
        Predict threat for a single network flow.
        
        Args:
            flow_data: Dictionary of feature name → value
        
        Returns:
            Dictionary with verdict, confidence, category, risk level
        """
        X = self._prepare_vector(flow_data)
        X_scaled = self.scaler.transform(X)
        
        if self.is_legacy:
            return self._predict_single_legacy(X_scaled)
        
        # Binary detection
        probs = self.binary_detector.predict_proba(X_scaled)[0]
        malicious_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
        benign_prob = 1.0 - malicious_prob
        
        is_threat = malicious_prob >= self.binary_threshold
        
        # Attack classification
        if is_threat:
            attack_cat, attack_conf = self._classify_attack(X_scaled)
        else:
            attack_cat = "Benign"
            attack_conf = benign_prob
        
        # Anomaly score
        ood_score = self._anomaly_score(X_scaled)
        
        # Final verdict
        risk_level = self._determine_risk(malicious_prob)
        
        result = {
            "verdict": "MALICIOUS (THREAT DETECTED)" if is_threat else "BENIGN (SAFE TRAFFIC)",
            "is_threat": is_threat,
            "predicted_category": attack_cat,
            "threat_confidence": round(malicious_prob * 100, 2),
            "benign_confidence": round(benign_prob * 100, 2),
            "risk_level": risk_level,
            "binary_threshold": self.binary_threshold,
            "class_probabilities": {
                "BENIGN": round(benign_prob, 4),
                "MALICIOUS": round(malicious_prob, 4),
            },
        }
        
        if ood_score is not None:
            result["ood_score"] = round(ood_score, 4)
            # Flag suspicious OOD even if classified benign
            if not is_threat and ood_score < -0.5:
                result["risk_level"] = "MODERATE (Unusual Pattern - OOD)"
                result["ood_warning"] = True
        
        return result
    
    def _predict_single_legacy(self, X_scaled: np.ndarray) -> dict:
        """Legacy prediction using multiclass model."""
        pred_idx = int(self.binary_detector.predict(X_scaled)[0])
        pred_label = self.legacy_classes[pred_idx]
        probs = self.binary_detector.predict_proba(X_scaled)[0]
        
        if self.legacy_benign_idx >= 0:
            threat_prob = float(1.0 - probs[self.legacy_benign_idx])
            benign_prob = float(probs[self.legacy_benign_idx])
        else:
            threat_prob = float(np.max(probs))
            benign_prob = 1.0 - threat_prob
        
        is_threat = pred_label != "BENIGN"
        
        return {
            "verdict": "MALICIOUS (THREAT DETECTED)" if is_threat else "BENIGN (SAFE TRAFFIC)",
            "is_threat": is_threat,
            "predicted_category": pred_label if is_threat else "Benign",
            "threat_confidence": round(threat_prob * 100, 2),
            "benign_confidence": round(benign_prob * 100, 2),
            "risk_level": self._determine_risk(threat_prob),
            "binary_threshold": 0.5,
            "class_probabilities": {
                cls: round(float(p), 4)
                for cls, p in zip(self.legacy_classes, probs)
            },
            "_legacy_model": True,
        }
    
    def predict_batch(
        self,
        df: pd.DataFrame,
        strict: bool = None,
    ) -> pd.DataFrame:
        """
        Predict threats for a batch of flows.
        
        Args:
            df: DataFrame with flow features
            strict: Override strict validation setting
        
        Returns:
            DataFrame with prediction columns added
        """
        if strict is None:
            strict = self.strict_validation
        
        # Prepare feature matrix with validation
        try:
            X_df, validation = prepare_feature_matrix(df, strict=strict)
        except ValueError as e:
            raise ValueError(f"Feature validation failed: {e}")
        
        # Ensure feature order matches model
        feature_data = {}
        for feat in self.feature_names:
            if feat in X_df.columns:
                feature_data[feat] = X_df[feat].values
            else:
                feature_data[feat] = np.zeros(len(X_df))
        
        X = np.column_stack([feature_data[f] for f in self.feature_names])
        X_scaled = self.scaler.transform(X)
        
        if self.is_legacy:
            return self._predict_batch_legacy(df, X_scaled, validation)
        
        # Binary detection
        binary_probs = self.binary_detector.predict_proba(X_scaled)
        malicious_probs = binary_probs[:, 1] if binary_probs.shape[1] > 1 else binary_probs[:, 0]
        is_threat = malicious_probs >= self.binary_threshold
        
        # Attack classification for malicious flows
        categories = []
        attack_confidences = []
        
        if self.attack_classifier is not None and is_threat.any():
            mal_indices = np.where(is_threat)[0]
            X_mal = X_scaled[mal_indices]
            
            attack_probs = self.attack_classifier.predict_proba(X_mal)
            
            for i, idx in enumerate(mal_indices):
                max_prob = float(np.max(attack_probs[i]))
                max_class = self.attack_classes[np.argmax(attack_probs[i])]
                
                if max_prob >= self.attack_threshold:
                    categories.append((idx, max_class, max_prob))
                else:
                    categories.append((idx, "Unknown Malicious", max_prob))
        
        # Build results
        results = df.copy()
        results["Predicted_Verdict"] = np.where(is_threat, "MALICIOUS", "BENIGN")
        results["Threat_Confidence_%"] = (malicious_probs * 100).round(2)
        results["Risk_Level"] = [self._determine_risk(p) for p in malicious_probs]
        
        # Assign categories
        cat_array = np.where(is_threat, "Unknown Malicious", "Benign")
        attack_conf_array = np.zeros(len(df))
        
        for idx, cat, conf in categories:
            cat_array[idx] = cat
            attack_conf_array[idx] = conf
        
        results["Predicted_Category"] = cat_array
        
        # Anomaly scores
        if self.anomaly_detector is not None:
            try:
                ood_scores = self.anomaly_detector.score_samples(X_scaled)
                results["OOD_Score"] = np.round(ood_scores, 4)
            except Exception:
                pass
        
        # Store validation info
        results.attrs["feature_validation"] = validation
        
        return results
    
    def _predict_batch_legacy(self, df, X_scaled, validation) -> pd.DataFrame:
        """Legacy batch prediction using multiclass model."""
        pred_indices = self.binary_detector.predict(X_scaled)
        probs = self.binary_detector.predict_proba(X_scaled)
        
        pred_labels = [self.legacy_classes[i] for i in pred_indices]
        
        if self.legacy_benign_idx >= 0:
            threat_probs = 1.0 - probs[:, self.legacy_benign_idx]
        else:
            threat_probs = np.max(probs, axis=1)
        
        results = df.copy()
        results["Predicted_Verdict"] = ["MALICIOUS" if l != "BENIGN" else "BENIGN" for l in pred_labels]
        results["Predicted_Category"] = pred_labels
        results["Threat_Confidence_%"] = (threat_probs * 100).round(2)
        results["Risk_Level"] = [self._determine_risk(p) for p in threat_probs]
        results.attrs["feature_validation"] = validation
        results.attrs["_legacy_model"] = True
        
        return results
    
    def get_model_info(self) -> dict:
        """Get model information for display."""
        return {
            "is_legacy": self.is_legacy,
            "feature_count": len(self.feature_names),
            "binary_threshold": self.binary_threshold,
            "attack_classes": self.attack_classes,
            "has_anomaly_detector": self.anomaly_detector is not None,
            "model_type": self.model_metadata.get("binary_model_type", "Unknown"),
            "training_date": self.model_metadata.get("training_date", "Unknown"),
            "datasets_used": self.model_metadata.get("datasets_used", []),
            "schema_version": self.model_metadata.get("schema_version", "1.0"),
        }


# ─── CLI Interface ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Encrypted Traffic Threat Predictor v2.0")
    parser.add_argument("--sample", type=str, default="data/sample_ddos_flow.json",
                       help="Path to JSON flow file")
    args = parser.parse_args()
    
    predictor = ThreatPredictor(strict_validation=False)
    
    if os.path.exists(args.sample):
        with open(args.sample, "r") as f:
            sample_data = json.load(f)
        
        print(f"\nEvaluating flow from: {args.sample}")
        res = predictor.predict_single(sample_data)
        print("\n" + "=" * 50)
        print(f"VERDICT:            {res['verdict']}")
        print(f"PREDICTED CATEGORY: {res['predicted_category']}")
        print(f"THREAT CONFIDENCE:  {res['threat_confidence']}%")
        print(f"RISK LEVEL:         {res['risk_level']}")
        print(f"THRESHOLD:          {res['binary_threshold']}")
        print("=" * 50)
    else:
        print(f"Sample file not found: {args.sample}")
