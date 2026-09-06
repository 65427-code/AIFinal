"""
AI-Based Encrypted Traffic Threat Detection
Model Training & Benchmarking Module
Trains Random Forest and XGBoost models, benchmarks security metrics, and exports artifacts.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier

DATA_PATH = "data/clean_dataset.csv"
FEATURE_COLS_PATH = "data/feature_columns.json"
MODELS_DIR = "models"

def evaluate_model(name, model, X_test, y_test, label_encoder, is_binary=False):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    
    benign_idx = list(label_encoder.classes_).index("BENIGN")
    
    # Binary metrics: Threat vs Benign
    y_test_binary = (y_test != benign_idx).astype(int)
    preds_binary = (preds != benign_idx).astype(int)
    threat_probs = 1.0 - probs[:, benign_idx]
    
    bin_acc = float(accuracy_score(y_test_binary, preds_binary))
    bin_prec = float(precision_score(y_test_binary, preds_binary, zero_division=0))
    bin_rec = float(recall_score(y_test_binary, preds_binary, zero_division=0))
    bin_f1 = float(f1_score(y_test_binary, preds_binary, zero_division=0))
    try:
        bin_roc_auc = float(roc_auc_score(y_test_binary, threat_probs))
    except Exception:
        bin_roc_auc = 0.0

    # Multi-class metrics
    multi_acc = float(accuracy_score(y_test, preds))
    multi_f1_macro = float(f1_score(y_test, preds, average="macro", zero_division=0))
    multi_f1_weighted = float(f1_score(y_test, preds, average="weighted", zero_division=0))
    
    cm = confusion_matrix(y_test, preds).tolist()
    cls_report = classification_report(
        y_test,
        preds,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )
    
    print(f"\n[{name}] Evaluation Summary:")
    print(f"  - Threat Detection Rate (Recall): {bin_rec * 100:.2f}%")
    print(f"  - Threat Precision:               {bin_prec * 100:.2f}%")
    print(f"  - Binary F1-Score:                {bin_f1 * 100:.2f}%")
    print(f"  - Binary ROC-AUC:                 {bin_roc_auc * 100:.2f}%")
    print(f"  - Multi-Class Accuracy:           {multi_acc * 100:.2f}%")
    print(f"  - Multi-Class F1 (Weighted):      {multi_f1_weighted * 100:.2f}%")
    
    return {
        "model_name": name,
        "binary_metrics": {
            "accuracy": round(bin_acc, 4),
            "precision": round(bin_prec, 4),
            "recall": round(bin_rec, 4),
            "f1_score": round(bin_f1, 4),
            "roc_auc": round(bin_roc_auc, 4)
        },
        "multiclass_metrics": {
            "accuracy": round(multi_acc, 4),
            "f1_macro": round(multi_f1_macro, 4),
            "f1_weighted": round(multi_f1_weighted, 4)
        },
        "confusion_matrix": cm,
        "classification_report": cls_report
    }

def main():
    print("=" * 60)
    print("AI-BASED ENCRYPTED TRAFFIC THREAT DETECTION: MODEL TRAINING")
    print("=" * 60)
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Clean dataset not found at '{DATA_PATH}'. Run src/preprocess.py first.")
        
    print(f"Loading dataset from: {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    with open(FEATURE_COLS_PATH, "r") as f:
        feature_cols = json.load(f)
        
    print(f"Loaded {len(df):,} samples with {len(feature_cols)} features.")
    
    X = df[feature_cols].values
    y_raw = df["Attack_Category"].values
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    print("\nClasses to train:")
    for idx, cls_name in enumerate(label_encoder.classes_):
        cnt = np.sum(y == idx)
        print(f"  [{idx}] {cls_name:16s}: {cnt:6,d} samples")
        
    print("\nSplitting into 80% train and 20% test sets (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train samples: {len(X_train):,}, Test samples: {len(X_test):,}")
    
    print("\nFitting StandardScaler on flow features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # -------------------------------------------------------------
    # 1. Random Forest Baseline
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Training Model 1: Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_eval = evaluate_model("Random Forest", rf_model, X_test_scaled, y_test, label_encoder)
    
    # -------------------------------------------------------------
    # 2. XGBoost Baseline
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Training Model 2: XGBoost Classifier...")
    xgb_model = XGBClassifier(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42,
        eval_metric="mlogloss"
    )
    xgb_model.fit(X_train_scaled, y_train)
    xgb_eval = evaluate_model("XGBoost", xgb_model, X_test_scaled, y_test, label_encoder)
    
    # Compare models by F1-score & Recall
    rf_score = rf_eval["binary_metrics"]["f1_score"]
    xgb_score = xgb_eval["binary_metrics"]["f1_score"]
    
    if xgb_score >= rf_score:
        best_model_name = "XGBoost"
        best_model = xgb_model
    else:
        best_model_name = "Random Forest"
        best_model = rf_model
        
    print("\n" + "=" * 60)
    print(f"Top Model Selected: {best_model_name} (F1: {max(rf_score, xgb_score)*100:.2f}%)")
    
    # Feature Importance (from best model or Random Forest)
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    else:
        importances = rf_model.feature_importances_
        
    feat_imp = sorted(
        [{"feature": f, "importance": round(float(imp), 6)} for f, imp in zip(feature_cols, importances)],
        key=lambda x: -x["importance"]
    )
    
    print("\nTop 10 Most Informative Encrypted Flow Features:")
    for i, item in enumerate(feat_imp[:10], 1):
        print(f"  {i:2d}. {item['feature']:30s}: {item['importance']*100:6.2f}%")
        
    # Save artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    model_path = os.path.join(MODELS_DIR, "threat_detector.joblib")
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    encoder_path = os.path.join(MODELS_DIR, "label_encoder.joblib")
    metrics_path = os.path.join(MODELS_DIR, "evaluation_results.json")
    feat_imp_path = os.path.join(MODELS_DIR, "feature_importance.json")
    
    print(f"\nSaving best model to: {model_path}...")
    joblib.dump(best_model, model_path)
    
    print(f"Saving scaler to: {scaler_path}...")
    joblib.dump(scaler, scaler_path)
    
    print(f"Saving label encoder to: {encoder_path}...")
    joblib.dump(label_encoder, encoder_path)
    
    evaluation_payload = {
        "best_model": best_model_name,
        "classes": list(label_encoder.classes_),
        "feature_count": len(feature_cols),
        "test_samples": len(X_test),
        "models": {
            "Random Forest": rf_eval,
            "XGBoost": xgb_eval
        }
    }
    
    with open(metrics_path, "w") as f:
        json.dump(evaluation_payload, f, indent=2)
    print(f"Saved evaluation metrics to: {metrics_path}")
    
    with open(feat_imp_path, "w") as f:
        json.dump(feat_imp, f, indent=2)
    print(f"Saved feature importances to: {feat_imp_path}")
    
    print("\n" + "=" * 60)
    print("Training and evaluation completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
