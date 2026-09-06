"""
AI-Based Encrypted Traffic Threat Detection
Inference Engine Module
Provides fast single-flow and batch-flow threat predictions without payload decryption.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

DEFAULT_MODEL_PATH = "models/threat_detector.joblib"
DEFAULT_SCALER_PATH = "models/scaler.joblib"
DEFAULT_ENCODER_PATH = "models/label_encoder.joblib"
DEFAULT_FEATURES_PATH = "data/feature_columns.json"

class ThreatPredictor:
    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        scaler_path=DEFAULT_SCALER_PATH,
        encoder_path=DEFAULT_ENCODER_PATH,
        features_path=DEFAULT_FEATURES_PATH
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found. Run src/train.py first.")
            
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)
        
        with open(features_path, "r") as f:
            self.feature_names = json.load(f)
            
        self.classes = list(self.label_encoder.classes_)
        self.benign_idx = self.classes.index("BENIGN") if "BENIGN" in self.classes else -1

    def _determine_risk(self, threat_prob: float, category: str) -> str:
        if threat_prob < 0.25:
            return "LOW (Normal Flow)"
        elif threat_prob < 0.60:
            return "MODERATE (Suspicious Pattern)"
        elif threat_prob < 0.85:
            return "HIGH (Likely Threat)"
        else:
            return "CRITICAL (Active Attack)"

    def predict_single(self, flow_data: dict) -> dict:
        """
        Predict threat for a single network flow given feature dictionary.
        """
        # Prepare vector in strict order of training features
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
            
        X = np.array(vector).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        pred_idx = int(self.model.predict(X_scaled)[0])
        pred_label = self.classes[pred_idx]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Calculate threat probability (1.0 - benign prob)
        if self.benign_idx >= 0:
            threat_prob = float(1.0 - probabilities[self.benign_idx])
            benign_prob = float(probabilities[self.benign_idx])
        else:
            threat_prob = float(np.max(probabilities))
            benign_prob = 1.0 - threat_prob
            
        is_threat = (pred_label != "BENIGN")
        risk_level = self._determine_risk(threat_prob, pred_label)
        
        # Breakdown of probabilities across all attack categories
        prob_breakdown = {
            cls_name: round(float(prob), 4)
            for cls_name, prob in zip(self.classes, probabilities)
        }
        
        return {
            "verdict": "MALICIOUS (THREAT DETECTED)" if is_threat else "BENIGN (SAFE TRAFFIC)",
            "is_threat": is_threat,
            "predicted_category": pred_label,
            "threat_confidence": round(threat_prob * 100, 2),
            "benign_confidence": round(benign_prob * 100, 2),
            "risk_level": risk_level,
            "class_probabilities": prob_breakdown
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict threats for a batch of flows in a pandas DataFrame.
        """
        df_clean = df.copy()
        df_clean.columns = [c.strip() for c in df_clean.columns]
        
        # Fill missing features with 0.0
        X_df = pd.DataFrame(index=df_clean.index)
        for feat in self.feature_names:
            if feat in df_clean.columns:
                X_df[feat] = pd.to_numeric(df_clean[feat], errors="coerce").fillna(0.0)
            else:
                X_df[feat] = 0.0
                
        # Replace infs
        X_df.replace([np.inf, -np.inf], 0.0, inplace=True)
        
        X_scaled = self.scaler.transform(X_df.values)
        pred_indices = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        pred_labels = [self.classes[i] for i in pred_indices]
        
        if self.benign_idx >= 0:
            threat_probs = 1.0 - probabilities[:, self.benign_idx]
        else:
            threat_probs = np.max(probabilities, axis=1)
            
        results = df.copy()
        results["Predicted_Verdict"] = ["MALICIOUS" if lbl != "BENIGN" else "BENIGN" for lbl in pred_labels]
        results["Predicted_Category"] = pred_labels
        results["Threat_Confidence_%"] = (threat_probs * 100).round(2)
        results["Risk_Level"] = [self._determine_risk(p, l) for p, l in zip(threat_probs, pred_labels)]
        
        return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Encrypted Traffic Threat Predictor")
    parser.add_argument("--sample", type=str, default="data/sample_ddos_flow.json", help="Path to JSON flow file")
    args = parser.parse_args()
    
    predictor = ThreatPredictor()
    
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
        print("=" * 50)
        print("Class Probabilities:")
        for cls_name, prob in res["class_probabilities"].items():
            print(f"  {cls_name:16s}: {prob*100:5.2f}%")
    else:
        print(f"Sample file not found: {args.sample}")
