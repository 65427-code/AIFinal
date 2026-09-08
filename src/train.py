"""
AI-Based Encrypted Traffic Threat Detection
Model Training & Benchmarking Pipeline v2.0

Trains a binary-first threat detection architecture:
1. Primary binary detector (BENIGN vs MALICIOUS)
2. Secondary attack family classifier
3. Optional IsolationForest anomaly detector

Supports:
- Multi-dataset training (CICIDS2017, USTC-TFC2016, CIRA-CIC-DoHBrw-2020)
- Group-aware train/test splitting
- Threshold calibration on validation set
- Per-dataset evaluation
- Held-out family evaluation
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
import time
import joblib
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Tuple

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    precision_recall_curve, roc_curve
)
from xgboost import XGBClassifier

from src.features.schema import CANONICAL_FEATURES, SCHEMA_VERSION, FORBIDDEN_TRAINING_COLUMNS
from src.config import (
    MODELS_DIR, COMBINED_DIR, PROCESSED_DIR,
    BINARY_DETECTOR_PATH, BINARY_SCALER_PATH,
    ATTACK_CLASSIFIER_PATH, ATTACK_LABEL_ENCODER_PATH,
    ANOMALY_DETECTOR_PATH, FEATURE_SCHEMA_PATH,
    MODEL_METADATA_PATH, EVALUATION_RESULTS_PATH,
    FEATURE_IMPORTANCE_PATH,
    DEFAULT_BINARY_THRESHOLD, ATTACK_CATEGORY_CONFIDENCE_THRESHOLD,
    ATTACK_CATEGORIES,
)


def compute_binary_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Compute comprehensive binary classification metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "total_benign": int(tn + fp),
        "total_malicious": int(tp + fn),
        "detected_malicious": int(tp),
        "missed_malicious": int(fn),
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


def calibrate_threshold(y_true, y_prob, method="f1") -> Tuple[float, dict]:
    """
    Calibrate binary classification threshold using validation data.
    
    Args:
        y_true: True binary labels (0/1)
        y_prob: Predicted probabilities for class 1
        method: 'f1' for max F1, 'recall_fpr' for recall-FPR tradeoff
    
    Returns:
        Tuple of (optimal_threshold, threshold_info)
    """
    precisions, recalls, thresholds_pr = precision_recall_curve(y_true, y_prob)
    
    # Calculate F1 for each threshold
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    
    if method == "f1":
        # Look for threshold with balanced recall and precision
        viable = recalls[:-1] >= 0.70
        if viable.any():
            viable_f1 = f1_scores[viable]
            best_idx = np.where(viable)[0][np.argmax(viable_f1)]
            best_threshold = float(thresholds_pr[best_idx])
        else:
            best_idx = np.argmax(f1_scores)
            best_threshold = float(thresholds_pr[best_idx]) if len(thresholds_pr) > 0 else 0.5
    elif method == "recall_fpr":
        # Find threshold where recall >= 0.90 and FPR is minimized
        fpr_arr, tpr_arr, thresholds_roc = roc_curve(y_true, y_prob)
        valid = tpr_arr >= 0.90
        if valid.any():
            best_idx = np.argmin(fpr_arr[valid])
            actual_idx = np.where(valid)[0][best_idx]
            best_threshold = float(thresholds_roc[min(actual_idx, len(thresholds_roc) - 1)])
        else:
            best_threshold = 0.5
    else:
        best_threshold = 0.5
    
    # Clamp to reasonable operational range (do NOT set artificially low or high)
    best_threshold = max(0.35, min(0.65, best_threshold))
    
    info = {
        "method": method,
        "threshold": best_threshold,
        "f1_at_threshold": float(f1_scores[np.argmin(np.abs(thresholds_pr - best_threshold))]) if len(thresholds_pr) > 0 else 0.0,
    }
    
    return best_threshold, info


def evaluate_per_dataset(y_true, y_pred, y_prob, datasets) -> dict:
    """Evaluate metrics separately for each dataset."""
    results = {}
    unique_datasets = sorted(set(datasets))
    
    for ds_name in unique_datasets:
        mask = np.array(datasets) == ds_name
        if mask.sum() == 0:
            continue
        
        ds_true = y_true[mask]
        ds_pred = y_pred[mask]
        ds_prob = y_prob[mask] if y_prob is not None else None
        
        results[ds_name] = compute_binary_metrics(ds_true, ds_pred, ds_prob)
        results[ds_name]["sample_count"] = int(mask.sum())
    
    return results


def main(
    data_path: Optional[str] = None,
    test_size: float = 0.20,
    val_size: float = 0.10,
    threshold_method: str = "f1",
    enable_anomaly: bool = True,
    held_out_families: Optional[List[str]] = None,
):
    """
    Main training pipeline.
    
    Args:
        data_path: Path to combined training dataset (parquet or CSV)
        test_size: Fraction for test set
        val_size: Fraction for validation set
        threshold_method: 'f1' or 'recall_fpr'
        enable_anomaly: Train IsolationForest on benign flows
        held_out_families: Malware families to exclude from training for generalization test
    """
    print("=" * 70)
    print("AI-BASED ENCRYPTED TRAFFIC THREAT DETECTION: MODEL TRAINING v2.0")
    print("=" * 70)
    
    # ─── Load Data ──────────────────────────────────────────────────────
    
    if data_path is None:
        # Try default paths
        for candidate in [
            os.path.join(COMBINED_DIR, "training_dataset.parquet"),
            os.path.join(COMBINED_DIR, "training_dataset.csv"),
        ]:
            if os.path.exists(candidate):
                data_path = candidate
                break
    
    if data_path is None or not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Training dataset not found. Run 'python src/prepare_datasets.py' first.\n"
            f"Searched: {COMBINED_DIR}"
        )
    
    print(f"\nLoading dataset from: {data_path}")
    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, low_memory=False)
    
    print(f"Loaded {len(df):,} samples")
    
    # ─── Validate Features ──────────────────────────────────────────────
    
    available_features = [f for f in CANONICAL_FEATURES if f in df.columns]
    missing_features = [f for f in CANONICAL_FEATURES if f not in df.columns]
    
    if missing_features:
        print(f"\n[WARN] {len(missing_features)} canonical features missing from dataset:")
        for mf in missing_features[:10]:
            print(f"  - {mf}")
        print("  These will be filled with 0.0")
    
    # ─── Prepare Features ───────────────────────────────────────────────
    
    # Ensure no forbidden columns leak in
    feature_cols = [f for f in CANONICAL_FEATURES if f not in FORBIDDEN_TRAINING_COLUMNS]
    
    X_df = pd.DataFrame(index=df.index)
    for feat in feature_cols:
        if feat in df.columns:
            X_df[feat] = pd.to_numeric(df[feat], errors="coerce").fillna(0.0)
        else:
            X_df[feat] = 0.0
    X_df.replace([np.inf, -np.inf], 0.0, inplace=True)
    
    # ─── Prepare Labels ────────────────────────────────────────────────
    
    if "Binary_Label" not in df.columns:
        raise ValueError("Dataset must contain 'Binary_Label' column (BENIGN/MALICIOUS)")
    
    y_binary = (df["Binary_Label"] == "MALICIOUS").astype(int).values
    
    # Attack category for secondary classifier
    if "Attack_Category" in df.columns:
        cats_series = df["Attack_Category"].fillna("Unknown Malicious").replace({"BENIGN": "Benign"})
        attack_cats = cats_series.values
    else:
        attack_cats = np.where(y_binary == 1, "Unknown Malicious", "Benign")
    
    # Dataset source for group-aware splitting
    if "Capture_ID" in df.columns:
        default_ds = df["Dataset"] if "Dataset" in df.columns else "unknown"
        groups = df["Capture_ID"].fillna(default_ds).astype(str).values
    elif "Dataset" in df.columns:
        groups = df["Dataset"].astype(str).values
    else:
        groups = np.arange(len(df)).astype(str)
    
    dataset_names = df["Dataset"].astype(str).values if "Dataset" in df.columns else np.full(len(df), "Unknown")
    
    # ─── Held-out Family Separation ─────────────────────────────────────
    
    held_out_data = None
    if held_out_families and "Specific_Family" in df.columns:
        held_out_mask = df["Specific_Family"].isin(held_out_families)
        held_out_count = held_out_mask.sum()
        
        if held_out_count > 0:
            print(f"\n[HELD-OUT] Holding out {held_out_count:,} samples from families: {held_out_families}")
            
            held_out_data = {
                "X": X_df[held_out_mask].values,
                "y": y_binary[held_out_mask],
                "families": df.loc[held_out_mask, "Specific_Family"].values,
                "datasets": dataset_names[held_out_mask],
            }
            
            # Remove held-out from training data
            train_mask = ~held_out_mask
            X_df = X_df[train_mask]
            y_binary = y_binary[train_mask.values]
            attack_cats = attack_cats[train_mask.values]
            groups = groups[train_mask.values]
            dataset_names = dataset_names[train_mask.values]
    
    X = X_df.values
    
    # ─── Dataset Summary ────────────────────────────────────────────────
    
    print(f"\nTraining Dataset Summary:")
    print(f"  Total samples: {len(X):,}")
    print(f"  Features: {X.shape[1]}")
    print(f"  BENIGN: {(y_binary == 0).sum():,}")
    print(f"  MALICIOUS: {(y_binary == 1).sum():,}")
    
    if "Dataset" in df.columns:
        print(f"\n  Per-dataset composition:")
        for ds_name in sorted(set(dataset_names)):
            ds_mask = dataset_names == ds_name
            ds_benign = (y_binary[ds_mask] == 0).sum()
            ds_malicious = (y_binary[ds_mask] == 1).sum()
            print(f"    {ds_name}: {ds_mask.sum():,} ({ds_benign:,} benign, {ds_malicious:,} malicious)")
    
    print(f"\n  Attack categories:")
    unique_cats = sorted(set(attack_cats))
    for cat in unique_cats:
        cnt = (attack_cats == cat).sum()
        print(f"    {cat}: {cnt:,}")
    
    # ─── Group-Aware Train/Val/Test Split ───────────────────────────────
    
    print(f"\nSplitting data (dataset-stratified with capture-level grouping for PCAPs)...")
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for ds in sorted(set(dataset_names)):
        ds_mask = np.where(dataset_names == ds)[0]
        y_ds = y_binary[ds_mask]
        groups_ds = groups[ds_mask]
        unique_ds_groups = np.unique(groups_ds)
        
        # Check if dataset has distinct PCAP groups for benign and malicious
        has_benign_groups = len(np.unique(groups_ds[y_ds == 0])) >= 2
        has_malicious_groups = len(np.unique(groups_ds[y_ds == 1])) >= 2
        
        if ds == "USTC-TFC2016" and has_benign_groups and has_malicious_groups:
            # Capture-level split separately for benign PCAPs and malware PCAPs
            # to guarantee both classes appear in train, val, and test without leaking PCAPs
            for label_val in [0, 1]:
                label_mask = np.where((dataset_names == ds) & (y_binary == label_val))[0]
                label_groups = groups[label_mask]
                
                gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
                tr_val_local, test_local = next(gss1.split(label_mask, groups=label_groups))
                
                groups_tr_val = label_groups[tr_val_local]
                gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=42)
                train_local_idx, val_local_idx = next(gss2.split(tr_val_local, groups=groups_tr_val))
                
                train_indices.extend(label_mask[tr_val_local[train_local_idx]])
                val_indices.extend(label_mask[tr_val_local[val_local_idx]])
                test_indices.extend(label_mask[test_local])
        else:
            # Stratified split for CSV/tabular datasets
            tr_val_idx, test_idx = train_test_split(
                ds_mask, test_size=test_size, random_state=42, stratify=y_ds
            )
            y_tr_val = y_binary[tr_val_idx]
            train_idx, val_idx = train_test_split(
                tr_val_idx, test_size=val_size / (1 - test_size), random_state=42, stratify=y_tr_val
            )
            train_indices.extend(train_idx)
            val_indices.extend(val_idx)
            test_indices.extend(test_idx)
            
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    
    X_train, y_train, cats_train = X[train_indices], y_binary[train_indices], attack_cats[train_indices]
    X_val, y_val = X[val_indices], y_binary[val_indices]
    X_test, y_test = X[test_indices], y_binary[test_indices]
    cats_test = attack_cats[test_indices]
    ds_val = dataset_names[val_indices]
    ds_test_names = dataset_names[test_indices]

    print(f"  Train: {len(X_train):,} ({(y_train == 1).sum():,} malicious)")
    print(f"  Validation: {len(X_val):,} ({(y_val == 1).sum():,} malicious)")
    print(f"  Test: {len(X_test):,} ({(y_test == 1).sum():,} malicious)")
    
    # ─── Scale Features ─────────────────────────────────────────────────
    
    print("\nFitting StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # ─── Train Binary Detector ──────────────────────────────────────────
    
    # Calculate class weights
    n_benign = (y_train == 0).sum()
    n_malicious = (y_train == 1).sum()
    weight_ratio = n_benign / max(n_malicious, 1)
    sample_weights = np.where(y_train == 1, weight_ratio, 1.0)
    
    print("\n" + "=" * 70)
    print("TRAINING BINARY THREAT DETECTOR")
    print("=" * 70)
    
    # Random Forest
    print("\nTraining Random Forest...")
    rf_binary = RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    rf_binary.fit(X_train_scaled, y_train)
    rf_val_prob = rf_binary.predict_proba(X_val_scaled)[:, 1]
    rf_val_pred = (rf_val_prob >= 0.5).astype(int)
    rf_val_metrics = compute_binary_metrics(y_val, rf_val_pred, rf_val_prob)
    print(f"  RF Validation - Recall: {rf_val_metrics['recall']*100:.2f}%, "
          f"Precision: {rf_val_metrics['precision']*100:.2f}%, "
          f"F1: {rf_val_metrics['f1_score']*100:.2f}%")
    
    # XGBoost
    print("\nTraining XGBoost...")
    xgb_binary = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        scale_pos_weight=weight_ratio,
        n_jobs=-1,
        random_state=42,
        eval_metric="logloss",
    )
    xgb_binary.fit(X_train_scaled, y_train)
    xgb_val_prob = xgb_binary.predict_proba(X_val_scaled)[:, 1]
    xgb_val_pred = (xgb_val_prob >= 0.5).astype(int)
    xgb_val_metrics = compute_binary_metrics(y_val, xgb_val_pred, xgb_val_prob)
    print(f"  XGB Validation - Recall: {xgb_val_metrics['recall']*100:.2f}%, "
          f"Precision: {xgb_val_metrics['precision']*100:.2f}%, "
          f"F1: {xgb_val_metrics['f1_score']*100:.2f}%")
    
    # Select best binary model
    if xgb_val_metrics["f1_score"] >= rf_val_metrics["f1_score"]:
        best_binary_name = "XGBoost"
        best_binary = xgb_binary
        best_val_prob = xgb_val_prob
    else:
        best_binary_name = "Random Forest"
        best_binary = rf_binary
        best_val_prob = rf_val_prob
    
    print(f"\n[OK] Best binary model: {best_binary_name}")
    
    # ─── Calibrate Threshold ────────────────────────────────────────────
    
    print(f"\nCalibrating threshold (method: {threshold_method})...")
    optimal_threshold, threshold_info = calibrate_threshold(y_val, best_val_prob, threshold_method)
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    print(f"  F1 at threshold: {threshold_info['f1_at_threshold']*100:.2f}%")
    
    # ─── Evaluate Binary on Test Set ────────────────────────────────────
    
    print("\n" + "=" * 70)
    print("EVALUATING BINARY DETECTOR ON TEST SET")
    print("=" * 70)
    
    test_prob = best_binary.predict_proba(X_test_scaled)[:, 1]
    test_pred = (test_prob >= optimal_threshold).astype(int)
    test_metrics = compute_binary_metrics(y_test, test_pred, test_prob)
    
    print(f"  Accuracy:  {test_metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {test_metrics['precision']*100:.2f}%")
    print(f"  Recall:    {test_metrics['recall']*100:.2f}%")
    print(f"  F1:        {test_metrics['f1_score']*100:.2f}%")
    print(f"  ROC-AUC:   {test_metrics.get('roc_auc', 0)*100:.2f}%")
    print(f"  FPR:       {test_metrics['false_positive_rate']*100:.2f}%")
    print(f"  FNR:       {test_metrics['false_negative_rate']*100:.2f}%")
    print(f"  TP: {test_metrics['true_positives']}, TN: {test_metrics['true_negatives']}, "
          f"FP: {test_metrics['false_positives']}, FN: {test_metrics['false_negatives']}")
    
    # Per-dataset evaluation
    per_dataset_results = evaluate_per_dataset(y_test, test_pred, test_prob, ds_test_names)
    
    if per_dataset_results:
        print(f"\n  Per-dataset results:")
        for ds_name, ds_metrics in per_dataset_results.items():
            print(f"    {ds_name}: Recall={ds_metrics['recall']*100:.1f}%, "
                  f"FPR={ds_metrics['false_positive_rate']*100:.1f}%, "
                  f"F1={ds_metrics['f1_score']*100:.1f}% "
                  f"(n={ds_metrics.get('sample_count', 'N/A')})")
    
    # ─── Train Attack Classifier ────────────────────────────────────────
    
    print("\n" + "=" * 70)
    print("TRAINING SECONDARY ATTACK CLASSIFIER")
    print("=" * 70)
    
    # Only train on malicious samples
    mal_train_mask = y_train == 1
    X_train_mal = X_train_scaled[mal_train_mask]
    cats_train_mal = cats_train[mal_train_mask]
    
    # Filter out 'Benign' category if present
    valid_cats = cats_train_mal != "Benign"
    X_train_mal = X_train_mal[valid_cats]
    cats_train_mal = cats_train_mal[valid_cats]
    
    if len(X_train_mal) > 0:
        attack_encoder = LabelEncoder()
        y_attack = attack_encoder.fit_transform(cats_train_mal)
        
        print(f"  Attack categories: {list(attack_encoder.classes_)}")
        print(f"  Training samples: {len(X_train_mal):,}")
        
        n_classes = len(attack_encoder.classes_)
        attack_classifier = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            n_jobs=-1,
            random_state=42,
            eval_metric="mlogloss",
            num_class=n_classes if n_classes > 2 else None,
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
        )
        attack_classifier.fit(X_train_mal, y_attack)
        
        # Evaluate on malicious test samples
        mal_test_mask = y_test == 1
        if mal_test_mask.sum() > 0:
            X_test_mal = X_test_scaled[mal_test_mask]
            cats_test_mal = cats_test[mal_test_mask]
            
            # Only evaluate categories the model knows
            known_mask = np.isin(cats_test_mal, attack_encoder.classes_)
            if known_mask.sum() > 0:
                X_test_known = X_test_mal[known_mask]
                cats_test_known = cats_test_mal[known_mask]
                y_attack_test = attack_encoder.transform(cats_test_known)
                
                attack_pred = attack_classifier.predict(X_test_known)
                attack_acc = accuracy_score(y_attack_test, attack_pred)
                attack_f1_macro = f1_score(y_attack_test, attack_pred, average="macro", zero_division=0)
                attack_f1_weighted = f1_score(y_attack_test, attack_pred, average="weighted", zero_division=0)
                
                print(f"\n  Attack classifier test accuracy: {attack_acc*100:.2f}%")
                print(f"  Macro F1: {attack_f1_macro*100:.2f}%")
                print(f"  Weighted F1: {attack_f1_weighted*100:.2f}%")
        
        has_attack_classifier = True
    else:
        print("  [WARN] No malicious training samples available for attack classifier")
        attack_classifier = None
        attack_encoder = None
        has_attack_classifier = False
    
    # ─── Train Anomaly Detector ─────────────────────────────────────────
    
    has_anomaly_detector = False
    anomaly_detector = None
    
    if enable_anomaly:
        print("\n" + "=" * 70)
        print("TRAINING ANOMALY DETECTOR (IsolationForest)")
        print("=" * 70)
        
        benign_train_mask = y_train == 0
        X_train_benign = X_train_scaled[benign_train_mask]
        
        if len(X_train_benign) > 100:
            # Subsample if too large
            max_anomaly_samples = min(len(X_train_benign), 50000)
            if len(X_train_benign) > max_anomaly_samples:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(X_train_benign), max_anomaly_samples, replace=False)
                X_train_benign = X_train_benign[idx]
            
            anomaly_detector = IsolationForest(
                n_estimators=100,
                contamination=0.05,
                random_state=42,
                n_jobs=-1,
            )
            anomaly_detector.fit(X_train_benign)
            has_anomaly_detector = True
            print(f"  Trained on {len(X_train_benign):,} benign samples")
        else:
            print("  [WARN] Not enough benign samples for anomaly detector")
    
    # ─── Feature Importance ─────────────────────────────────────────────
    
    print("\nComputing feature importance...")
    if hasattr(best_binary, "feature_importances_"):
        importances = best_binary.feature_importances_
    else:
        importances = np.zeros(len(feature_cols))
    
    feat_imp = sorted(
        [{"feature": f, "importance": round(float(imp), 6)}
         for f, imp in zip(feature_cols, importances)],
        key=lambda x: -x["importance"]
    )
    
    print("  Top 10 features:")
    for i, item in enumerate(feat_imp[:10], 1):
        print(f"    {i:2d}. {item['feature']:35s}: {item['importance']*100:6.2f}%")
    
    # ─── Held-Out Family Evaluation ─────────────────────────────────────
    
    held_out_results = {}
    if held_out_data is not None:
        print("\n" + "=" * 70)
        print("HELD-OUT FAMILY EVALUATION (GENERALIZATION TEST)")
        print("=" * 70)
        
        X_held = scaler.transform(held_out_data["X"])
        y_held = held_out_data["y"]
        families_held = held_out_data["families"]
        
        held_prob = best_binary.predict_proba(X_held)[:, 1]
        held_pred = (held_prob >= optimal_threshold).astype(int)
        
        for family in sorted(set(families_held)):
            fam_mask = families_held == family
            fam_true = y_held[fam_mask]
            fam_pred = held_pred[fam_mask]
            fam_prob = held_prob[fam_mask]
            
            fam_metrics = compute_binary_metrics(fam_true, fam_pred, fam_prob)
            held_out_results[family] = fam_metrics
            
            print(f"\n  {family}:")
            print(f"    Total flows: {fam_mask.sum()}")
            print(f"    Detected malicious: {fam_metrics['detected_malicious']}")
            print(f"    Missed malicious: {fam_metrics['missed_malicious']}")
            print(f"    Recall: {fam_metrics['recall']*100:.1f}%")
            print(f"    Precision: {fam_metrics['precision']*100:.2f}%")
    
    # ─── Save Artifacts ─────────────────────────────────────────────────
    
    print("\n" + "=" * 70)
    print("SAVING MODEL ARTIFACTS")
    print("=" * 70)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    joblib.dump(best_binary, BINARY_DETECTOR_PATH)
    print(f"  [SAVED] Binary detector -> {BINARY_DETECTOR_PATH}")
    
    joblib.dump(scaler, BINARY_SCALER_PATH)
    print(f"  [SAVED] Scaler -> {BINARY_SCALER_PATH}")
    
    if has_attack_classifier:
        joblib.dump(attack_classifier, ATTACK_CLASSIFIER_PATH)
        print(f"  [SAVED] Attack classifier -> {ATTACK_CLASSIFIER_PATH}")
        joblib.dump(attack_encoder, ATTACK_LABEL_ENCODER_PATH)
        print(f"  [SAVED] Attack encoder -> {ATTACK_LABEL_ENCODER_PATH}")
    
    if has_anomaly_detector:
        joblib.dump(anomaly_detector, ANOMALY_DETECTOR_PATH)
        print(f"  [SAVED] Anomaly detector -> {ANOMALY_DETECTOR_PATH}")
    
    # Feature schema
    with open(FEATURE_SCHEMA_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"  [SAVED] Feature schema -> {FEATURE_SCHEMA_PATH}")
    
    # Model metadata
    metadata = {
        "training_date": datetime.now().isoformat(),
        "schema_version": SCHEMA_VERSION,
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "binary_threshold": optimal_threshold,
        "threshold_method": threshold_method,
        "attack_category_threshold": ATTACK_CATEGORY_CONFIDENCE_THRESHOLD,
        "binary_model_type": best_binary_name,
        "has_attack_classifier": has_attack_classifier,
        "has_anomaly_detector": has_anomaly_detector,
        "attack_classes": list(attack_encoder.classes_) if has_attack_classifier else [],
        "datasets_used": sorted(set(dataset_names)),
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "held_out_families": held_out_families or [],
        "class_distribution": {
            "train_benign": int((y_train == 0).sum()),
            "train_malicious": int((y_train == 1).sum()),
        },
    }
    
    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  [SAVED] Metadata -> {MODEL_METADATA_PATH}")
    
    # Evaluation results
    evaluation = {
        "schema_version": SCHEMA_VERSION,
        "binary_detector": {
            "model": best_binary_name,
            "threshold": optimal_threshold,
            "internal_test": test_metrics,
            "models_compared": {
                "Random Forest": rf_val_metrics,
                "XGBoost": xgb_val_metrics,
            },
        },
        "datasets": per_dataset_results,
        "held_out_families": held_out_results,
        "training_composition": {
            ds: int((dataset_names == ds).sum())
            for ds in sorted(set(dataset_names))
        },
    }
    
    if has_attack_classifier:
        evaluation["attack_classifier"] = {
            "classes": list(attack_encoder.classes_),
            "accuracy": float(attack_acc) if 'attack_acc' in dir() else 0.0,
        }
    
    with open(EVALUATION_RESULTS_PATH, "w") as f:
        json.dump(evaluation, f, indent=2)
    print(f"  [SAVED] Evaluation results -> {EVALUATION_RESULTS_PATH}")
    
    # Feature importance
    with open(FEATURE_IMPORTANCE_PATH, "w") as f:
        json.dump(feat_imp, f, indent=2)
    print(f"  [SAVED] Feature importance -> {FEATURE_IMPORTANCE_PATH}")
    
    # Also save to legacy paths for backward compatibility
    with open(os.path.join("data", "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    return metadata, evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI Encrypted Traffic Threat Detection models v2.0")
    parser.add_argument("--data", type=str, default=None,
                       help="Path to combined training dataset")
    parser.add_argument("--test-size", type=float, default=0.20,
                       help="Test set fraction (default: 0.20)")
    parser.add_argument("--val-size", type=float, default=0.10,
                       help="Validation set fraction (default: 0.10)")
    parser.add_argument("--threshold-method", type=str, default="f1",
                       choices=["f1", "recall_fpr"],
                       help="Threshold calibration method")
    parser.add_argument("--no-anomaly", action="store_true",
                       help="Disable IsolationForest anomaly detector")
    parser.add_argument("--held-out-families", type=str, nargs="+", default=None,
                       help="Malware families to hold out for generalization testing")
    
    args = parser.parse_args()
    
    main(
        data_path=args.data,
        test_size=args.test_size,
        val_size=args.val_size,
        threshold_method=args.threshold_method,
        enable_anomaly=not args.no_anomaly,
        held_out_families=args.held_out_families,
    )
