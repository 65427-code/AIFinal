"""
CLI tool for evaluating PCAP files against the trained ThreatPredictor model.
Used for regression testing.
"""

import argparse
import sys
import os

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List

from src.predict import ThreatPredictor
from src.pcap_extractor import extract_flows_from_pcap

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PCAP against trained model.")
    parser.add_argument("--pcap", required=True, help="Path to PCAP file")
    parser.add_argument("--expected", required=True, choices=["malicious", "benign"], help="Expected overall label")
    parser.add_argument("--dataset", default="Unknown", help="Dataset name for reporting")
    parser.add_argument("--family", default="Unknown", help="Malware family name")
    parser.add_argument("--output", help="Output CSV path")
    parser.add_argument("--max-packets", type=int, default=None, help="Max packets to process")
    
    args = parser.parse_args()
    
    try:
        print(f"Loading ThreatPredictor...")
        predictor = ThreatPredictor(strict_validation=False)
        
        print(f"Extracting flows from {args.pcap}...")
        flows, _meta = extract_flows_from_pcap(args.pcap, max_packets=args.max_packets)
        
        if flows is None or flows.empty:
            print("No flows extracted from PCAP.")
            sys.exit(0)
            
        print("Running predictions...")
        df_results = predictor.predict_batch(flows)
        
        extraction_info = flows.attrs.get("extraction_info", {})
        total_packets = extraction_info.get("packets_processed", len(flows))
        truncated = extraction_info.get("was_truncated", False)
        total_flows = len(df_results)
        
        # Check verdict column
        if 'Predicted_Verdict' in df_results.columns:
            malicious_count = int((df_results['Predicted_Verdict'] == 'MALICIOUS').sum())
        elif 'is_threat' in df_results.columns:
            malicious_count = int((df_results['is_threat'] == True).sum())
        else:
            malicious_count = 0
            
        benign_count = total_flows - malicious_count
        
        malicious_pct = (malicious_count / total_flows) * 100 if total_flows > 0 else 0.0
        benign_pct = (benign_count / total_flows) * 100 if total_flows > 0 else 0.0
        
        if 'Threat_Confidence_%' in df_results.columns:
            confidences = df_results['Threat_Confidence_%'].values
            mean_conf = float(np.mean(confidences)) if len(confidences) > 0 else 0.0
            median_conf = float(np.median(confidences)) if len(confidences) > 0 else 0.0
            max_conf = float(np.max(confidences)) if len(confidences) > 0 else 0.0
            min_conf = float(np.min(confidences)) if len(confidences) > 0 else 0.0
        elif 'threat_confidence' in df_results.columns:
            confidences = df_results['threat_confidence'].values
            mean_conf = float(np.mean(confidences)) if len(confidences) > 0 else 0.0
            median_conf = float(np.median(confidences)) if len(confidences) > 0 else 0.0
            max_conf = float(np.max(confidences)) if len(confidences) > 0 else 0.0
            min_conf = float(np.min(confidences)) if len(confidences) > 0 else 0.0
        else:
            mean_conf = median_conf = max_conf = min_conf = 0.0
            
        print("\n" + "=" * 62)
        print("PCAP EVALUATION REPORT")
        print("=" * 62)
        print(f"File:     {args.pcap}")
        print(f"Dataset:  {args.dataset}")
        print(f"Family:   {args.family}")
        print(f"Expected: {args.expected}")
        print("\nCapture Summary:")
        print(f"  Packets processed:    {total_packets:,}")
        print(f"  Truncated:            {'Yes' if truncated else 'No'}")
        print(f"  Flows extracted:      {total_flows:,}")
        print("\nPrediction Results:")
        print(f"  Predicted MALICIOUS:  {malicious_count:,}  ({malicious_pct:.1f}%)")
        print(f"  Predicted BENIGN:     {benign_count:,}  ({benign_pct:.1f}%)")
        print("")
        
        if args.expected == "malicious":
            print(f"  Detection Rate:       {malicious_pct:.1f}%")
        else:
            print(f"  False Positive Rate:  {malicious_pct:.1f}%")
            
        print("\nThreat Confidence Statistics:")
        print(f"  Mean:    {mean_conf:.1f}%")
        print(f"  Median:  {median_conf:.1f}%")
        print(f"  Max:     {max_conf:.1f}%")
        print(f"  Min:     {min_conf:.1f}%")
        
        print("\nPredicted Categories:")
        cat_col = 'Predicted_Category' if 'Predicted_Category' in df_results.columns else ('predicted_category' if 'predicted_category' in df_results.columns else None)
        if cat_col:
            category_counts = df_results[cat_col].value_counts()
            for cat, count in category_counts.items():
                print(f"  {cat}: {count:,} flows")
        else:
            print("  Category information not available in predictions.")
            
        print("=" * 62)
        
        if args.output:
            df_results.to_csv(args.output, index=False)
            print(f"\nDetailed results saved to {args.output}")
            
    except Exception as e:
        print(f"Error evaluating PCAP: {e}")
        sys.exit(1)
        
    sys.exit(0)

if __name__ == "__main__":
    main()
