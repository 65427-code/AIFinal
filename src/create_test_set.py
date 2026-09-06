"""
AI-Based Encrypted Traffic Threat Detection
Test Dataset Generator
Extracts an independent, balanced test dataset (500 Benign vs 500 Threats)
specifically designed to test 'Threat vs. No Threat' detection.
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import normalize_label, DROP_COLUMNS
from src.predict import ThreatPredictor

OUTPUT_WITH_LABELS = "data/test_threat_detection_with_ground_truth.csv"
OUTPUT_UNLABELLED = "data/test_threat_detection_unlabelled.csv"
FEATURES_PATH = "data/feature_columns.json"

def generate_test_dataset(random_seed=777):
    print("=" * 60)
    print("GENERATING DEDICATED THREAT VS. NO-THREAT TEST DATASET")
    print("=" * 60)
    
    with open(FEATURES_PATH, "r") as f:
        feature_cols = json.load(f)

    threat_quotas = {
        "DDoS": 100,
        "PortScan": 100,
        "DoS": 120,
        "Brute Force": 80,
        "Web Attack": 50,
        "Botnet": 45,
        "Infiltration": 5
    }
    
    csv_files = glob.glob("archive/*.csv")
    if not csv_files:
        raise FileNotFoundError("Raw archive CSV files not found.")
        
    collected_benign = []
    collected_threats = {cat: [] for cat in threat_quotas}
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        print(f"Sampling from: {filename}...")
        try:
            df = pd.read_csv(filepath, encoding="utf-8", low_memory=False)
        except Exception:
            df = pd.read_csv(filepath, encoding="latin-1", low_memory=False)
            
        df.columns = [c.strip() for c in df.columns]
        label_col = [c for c in df.columns if c.lower() == "label"][0]
        df["Category"] = df[label_col].apply(normalize_label)
        
        # Clean inf/nan
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        cols_needed = [c for c in feature_cols if c in df.columns]
        df.dropna(subset=cols_needed + ["Category"], inplace=True)
        
        # Collect Benign
        b_df = df[df["Category"] == "BENIGN"]
        if len(b_df) > 0 and len(collected_benign) < 1000:
            collected_benign.append(b_df.sample(n=min(len(b_df), 150), random_state=random_seed))
            
        # Collect Threats
        for cat, quota in threat_quotas.items():
            curr_len = sum(len(x) for x in collected_threats[cat])
            if curr_len < quota:
                c_df = df[df["Category"] == cat]
                if len(c_df) > 0:
                    take = min(quota - curr_len, len(c_df))
                    collected_threats[cat].append(c_df.sample(n=take, random_state=random_seed))

    # Combine benign and downsample to exactly 500
    all_benign_df = pd.concat(collected_benign, ignore_index=True).sample(n=500, random_state=random_seed)
    all_benign_df["Expected_Verdict"] = "BENIGN"
    all_benign_df["Actual_Attack_Type"] = "BENIGN"
    
    # Combine threats
    threat_dfs = []
    for cat, chunks in collected_threats.items():
        if chunks:
            tdf = pd.concat(chunks, ignore_index=True)
            tdf["Expected_Verdict"] = "THREAT"
            tdf["Actual_Attack_Type"] = cat
            threat_dfs.append(tdf)
            
    all_threats_df = pd.concat(threat_dfs, ignore_index=True)
    
    # Merge and shuffle
    final_test_df = pd.concat([all_benign_df, all_threats_df], ignore_index=True)
    final_test_df = final_test_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    
    # Order columns: metadata first, then features
    ordered_cols_with_labels = ["Expected_Verdict", "Actual_Attack_Type"] + feature_cols
    df_with_labels = final_test_df[ordered_cols_with_labels]
    df_unlabelled = final_test_df[feature_cols]
    
    # Save files
    os.makedirs("data", exist_ok=True)
    df_with_labels.to_csv(OUTPUT_WITH_LABELS, index=False)
    df_unlabelled.to_csv(OUTPUT_UNLABELLED, index=False)
    
    print("\n" + "=" * 60)
    print("TEST DATASET GENERATED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Total rows: {len(final_test_df):,} flows")
    print(f"Benign flows (No Threat): 500 (50.0%)")
    print(f"Threat flows:             500 (50.0%)")
    print("\nThreat Breakdown:")
    for cat, count in final_test_df[final_test_df['Expected_Verdict'] == 'THREAT']['Actual_Attack_Type'].value_counts().items():
        print(f"  - {cat:16s}: {count} flows")
        
    print(f"\n1. With Ground Truth Labels : {OUTPUT_WITH_LABELS}")
    print(f"2. Unlabelled (Pure Features): {OUTPUT_UNLABELLED}")
    
    # Run verification test with ThreatPredictor
    print("\n" + "=" * 60)
    print("VALIDATING ON TRAINED AI MODEL...")
    print("=" * 60)
    predictor = ThreatPredictor()
    scanned = predictor.predict_batch(df_unlabelled)
    
    pred_verdicts = scanned["Predicted_Verdict"].values
    true_verdicts = ["MALICIOUS" if v == "THREAT" else "BENIGN" for v in df_with_labels["Expected_Verdict"]]
    
    correct = sum(p == t for p, t in zip(pred_verdicts, true_verdicts))
    total = len(true_verdicts)
    accuracy = (correct / total) * 100
    
    # Threat detection rate (recall)
    threat_mask = [t == "MALICIOUS" for t in true_verdicts]
    threat_correct = sum(p == "MALICIOUS" and t == "MALICIOUS" for p, t in zip(pred_verdicts, true_verdicts))
    detection_rate = (threat_correct / sum(threat_mask)) * 100
    
    # False positive rate
    benign_mask = [t == "BENIGN" for t in true_verdicts]
    false_positives = sum(p == "MALICIOUS" and t == "BENIGN" for p, t in zip(pred_verdicts, true_verdicts))
    fpr = (false_positives / sum(benign_mask)) * 100
    
    print(f"Overall Accuracy:       {accuracy:.2f}% ({correct}/{total})")
    print(f"Threat Detection Rate:  {detection_rate:.2f}% ({threat_correct}/500 threats caught)")
    print(f"False Positive Rate:    {fpr:.2f}% ({false_positives}/500 benign flagged)")
    print("=" * 60)

if __name__ == "__main__":
    generate_test_dataset()
