"""
AI-Based Encrypted Traffic Threat Detection
Model Evaluation and Cross-Dataset Benchmark Module

Evaluates the trained binary threat detector and secondary attack classifier
across multiple datasets and produces:
- Internal test set metrics
- Per-dataset generalization metrics (CICIDS2017, USTC-TFC2016, CIRA-CIC-DoHBrw-2020)
- Held-out family evaluation (e.g. Cridex)
- Cross-dataset generalization matrix
- Confusion matrices and detailed statistics
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)

from src.predict import ThreatPredictor
from src.features.schema import CANONICAL_FEATURES
from src.config import (
    COMBINED_DIR, PROCESSED_DIR, EVALUATION_RESULTS_PATH,
    MODEL_METADATA_PATH,
)


def compute_comprehensive_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Compute all evaluation metrics specified in requirements."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    elif cm.shape == (1, 1):
        if y_true[0] == 0:
            tn, fp, fn, tp = len(y_true), 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, len(y_true)
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    total_benign = int(tn + fp)
    total_malicious = int(tp + fn)
    total = len(y_true)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / total_benign) if total_benign > 0 else 0.0,
        "false_positive_rate": float(fp / total_benign) if total_benign > 0 else 0.0,
        "false_negative_rate": float(fn / total_malicious) if total_malicious > 0 else 0.0,
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        },
        "counts": {
            "total_flows": total,
            "benign_flow_count": total_benign,
            "malicious_flow_count": total_malicious,
            "detected_malicious_count": int(tp),
            "missed_malicious_count": int(fn),
        }
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = 0.0
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            metrics["pr_auc"] = 0.0

    return metrics


def print_metrics_table(dataset_name: str, metrics: dict):
    """Pretty print metrics table."""
    print(f"\n{'=' * 60}")
    print(f"EVALUATION: {dataset_name}")
    print(f"{'=' * 60}")
    counts = metrics.get("counts", {})
    cm = metrics.get("confusion_matrix", {})
    print(f"Total Flows:        {counts.get('total_flows', 0):,}")
    print(f"Benign Flows:       {counts.get('benign_flow_count', 0):,}")
    print(f"Malicious Flows:    {counts.get('malicious_flow_count', 0):,}")
    print(f"Detected Threats:   {counts.get('detected_malicious_count', 0):,}")
    print(f"Missed Threats:     {counts.get('missed_malicious_count', 0):,}")
    print(f"{'-' * 60}")
    print(f"Accuracy:           {metrics.get('accuracy', 0)*100:6.2f}%")
    print(f"Precision:          {metrics.get('precision', 0)*100:6.2f}%")
    print(f"Recall (Detection): {metrics.get('recall', 0)*100:6.2f}%")
    print(f"F1-Score:           {metrics.get('f1_score', 0)*100:6.2f}%")
    print(f"Specificity:        {metrics.get('specificity', 0)*100:6.2f}%")
    print(f"False Positive Rate:{metrics.get('false_positive_rate', 0)*100:6.2f}%")
    print(f"False Negative Rate:{metrics.get('false_negative_rate', 0)*100:6.2f}%")
    if "roc_auc" in metrics:
        print(f"ROC-AUC:            {metrics.get('roc_auc', 0)*100:6.2f}%")
    if "pr_auc" in metrics:
        print(f"PR-AUC:             {metrics.get('pr_auc', 0)*100:6.2f}%")
    print(f"{'-' * 60}")
    print(f"Confusion Matrix:   TP={cm.get('TP', 0)}  TN={cm.get('TN', 0)}  FP={cm.get('FP', 0)}  FN={cm.get('FN', 0)}")


def evaluate_dataset(predictor: ThreatPredictor, df: pd.DataFrame, dataset_name: str) -> dict:
    """Evaluate predictor on a DataFrame with ground truth labels."""
    # Determine ground truth
    if "Binary_Label" in df.columns:
        y_true = (df["Binary_Label"].str.upper() == "MALICIOUS").astype(int).values
    elif "Expected_Verdict" in df.columns:
        y_true = (df["Expected_Verdict"].astype(str).str.upper() != "BENIGN").astype(int).values
    elif "Is_Malicious" in df.columns:
        y_true = df["Is_Malicious"].astype(int).values
    elif "Label" in df.columns:
        y_true = (df["Label"].astype(str).str.upper() != "BENIGN").astype(int).values
    elif "label" in df.columns:
        y_true = (df["label"].astype(str).str.upper() != "BENIGN").astype(int).values
    else:
        raise ValueError(f"No ground truth column found in {dataset_name}")

    results = predictor.predict_batch(df, strict=False)
    y_pred = (results["Predicted_Verdict"] == "MALICIOUS").astype(int).values
    y_prob = (results["Threat_Confidence_%"] / 100.0).values

    metrics = compute_comprehensive_metrics(y_true, y_pred, y_prob)
    print_metrics_table(dataset_name, metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate AI Encrypted Traffic Threat Detector")
    parser.add_argument("--test-file", type=str, default=None, help="Path to specific test CSV or Parquet file")
    parser.add_argument("--processed-dir", type=str, default=PROCESSED_DIR, help="Path to processed datasets dir")
    parser.add_argument("--output", type=str, default=EVALUATION_RESULTS_PATH, help="Path to save evaluation results JSON")
    args = parser.parse_args()

    print("=" * 60)
    print("AI-BASED ENCRYPTED TRAFFIC THREAT DETECTION: MODEL EVALUATION")
    print("=" * 60)

    predictor = ThreatPredictor(strict_validation=False)
    info = predictor.get_model_info()
    print(f"Model Type:        {info.get('model_type')}")
    print(f"Schema Version:    {info.get('schema_version')}")
    print(f"Features:          {info.get('feature_count')}")
    print(f"Binary Threshold:  {info.get('binary_threshold'):.3f}")

    eval_summary = {
        "model_info": info,
        "datasets": {},
        "held_out_families": {},
    }

    # Evaluate specific test file if provided
    if args.test_file and os.path.exists(args.test_file):
        print(f"\nEvaluating single test file: {args.test_file}")
        if args.test_file.endswith(".parquet"):
            df = pd.read_parquet(args.test_file)
        else:
            df = pd.read_csv(args.test_file)
        metrics = evaluate_dataset(predictor, df, os.path.basename(args.test_file))
        eval_summary["datasets"][os.path.basename(args.test_file)] = metrics

    # Evaluate existing standard test files in data/
    standard_1k_gt = "data/test_threat_detection_with_ground_truth.csv"
    if os.path.exists(standard_1k_gt):
        df_1k = pd.read_csv(standard_1k_gt)
        metrics_1k = evaluate_dataset(predictor, df_1k, "Internal-1k-Test (500 Benign vs 500 Threats)")
        eval_summary["internal_test"] = metrics_1k

    # Check for processed datasets in processed_dir
    dataset_files = {
        "CICIDS2017": os.path.join(args.processed_dir, "cicids2017.parquet"),
        "USTC-TFC2016": os.path.join(args.processed_dir, "ustc_tfc2016.parquet"),
        "CIRA-CIC-DoHBrw-2020": os.path.join(args.processed_dir, "dohbrw2020.parquet"),
    }

    for name, path in dataset_files.items():
        if os.path.exists(path):
            df_ds = pd.read_parquet(path)
            # Sample if extremely large to prevent OOM
            if len(df_ds) > 50000:
                df_ds = df_ds.sample(50000, random_state=42)
            metrics = evaluate_dataset(predictor, df_ds, name)
            eval_summary["datasets"][name] = metrics

            # If USTC dataset, check for held out families
            if name == "USTC-TFC2016" and "Specific_Family" in df_ds.columns:
                for fam in ["Cridex", "Zeus", "Tinba", "Virut", "Neris"]:
                    fam_df = df_ds[df_ds["Specific_Family"] == fam]
                    if len(fam_df) > 0:
                        fam_metrics = evaluate_dataset(predictor, fam_df, f"USTC Family: {fam}")
                        eval_summary["held_out_families"][fam] = fam_metrics
        else:
            print(f"\n[INFO] {name}: Not found in {path}")

    # Cross-dataset generalization matrix table
    if eval_summary["datasets"]:
        print("\n" + "=" * 80)
        print("CROSS-DATASET GENERALIZATION MATRIX")
        print("=" * 80)
        headers = ["Dataset", "Flows", "Accuracy", "Recall", "Precision", "F1-Score", "FPR", "FNR"]
        print(f"{headers[0]:<25} {headers[1]:<8} {headers[2]:<10} {headers[3]:<10} {headers[4]:<10} {headers[5]:<10} {headers[6]:<8} {headers[7]:<8}")
        print("-" * 95)
        for ds_name, m in eval_summary["datasets"].items():
            acc = f"{m.get('accuracy', 0)*100:.2f}%"
            rec = f"{m.get('recall', 0)*100:.2f}%"
            prec = f"{m.get('precision', 0)*100:.2f}%"
            f1 = f"{m.get('f1_score', 0)*100:.2f}%"
            fpr = f"{m.get('false_positive_rate', 0)*100:.2f}%"
            fnr = f"{m.get('false_negative_rate', 0)*100:.2f}%"
            flows = str(m.get('counts', {}).get('total_flows', 0))
            print(f"{ds_name:<25} {flows:<8} {acc:<10} {rec:<10} {prec:<10} {f1:<10} {fpr:<8} {fnr:<8}")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"\nSaved evaluation summary to: {args.output}")


if __name__ == "__main__":
    main()
