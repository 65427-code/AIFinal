"""
AI-Based Encrypted Traffic Threat Detection
Data Preprocessing & Feature Engineering Module
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ARCHIVE_DIR = "archive"
DATA_DIR = "data"

# Zero variance or duplicate columns in CICIDS2017 to remove
DROP_COLUMNS = [
    "Fwd Header Length.1",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "RST Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
]

# Attack label normalization mapping
def normalize_label(raw_label: str) -> str:
    cleaned = str(raw_label).strip().encode("ascii", "ignore").decode("ascii")
    if cleaned == "BENIGN":
        return "BENIGN"
    elif "DDoS" in cleaned:
        return "DDoS"
    elif "PortScan" in cleaned:
        return "PortScan"
    elif "DoS" in cleaned or "Heartbleed" in cleaned:
        return "DoS"
    elif "Patator" in cleaned:
        return "Brute Force"
    elif "Bot" in cleaned:
        return "Botnet"
    elif "Web Attack" in cleaned:
        return "Web Attack"
    elif "Infiltration" in cleaned:
        return "Infiltration"
    return "Other Attack"

def load_and_clean_data(archive_dir=ARCHIVE_DIR, max_per_class=18000, random_state=42):
    print("=" * 60)
    print("Step 1: Discovering CSV files in archive/...")
    csv_files = glob.glob(os.path.join(archive_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory '{archive_dir}'")
    print(f"Found {len(csv_files)} CSV files.")

    sampled_dfs = []
    
    for idx, filepath in enumerate(csv_files, 1):
        filename = os.path.basename(filepath)
        print(f"[{idx}/{len(csv_files)}] Reading: {filename}...")
        
        try:
            df = pd.read_csv(filepath, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding="latin-1", low_memory=False)
            
        # Strip whitespaces from column headers
        df.columns = [c.strip() for c in df.columns]
        
        # Ensure Label column exists
        label_col = [c for c in df.columns if c.lower() == "label"]
        if not label_col:
            continue
        label_name = label_col[0]
        
        # Drop columns if present
        cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
        df.drop(columns=cols_to_drop, inplace=True)
        
        # Normalize labels
        df["Attack_Category"] = df[label_name].apply(normalize_label)
        df["Is_Malicious"] = (df["Attack_Category"] != "BENIGN").astype(int)
        df.drop(columns=[label_name], inplace=True)
        
        # Replace inf and -inf with NaN, then drop NaNs
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        # Convert feature columns to numeric
        feature_cols = [c for c in df.columns if c not in ["Attack_Category", "Is_Malicious"]]
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(inplace=True)
        
        # Sample per category to balance and avoid RAM exhaustion
        class_samples = []
        for cat, group in df.groupby("Attack_Category"):
            if len(group) > max_per_class:
                class_samples.append(group.sample(n=max_per_class, random_state=random_state))
            else:
                class_samples.append(group)
                
        if class_samples:
            sampled_dfs.append(pd.concat(class_samples, ignore_index=True))

    combined = pd.concat(sampled_dfs, ignore_index=True)
    print(f"Combined sampled rows before final balancing: {len(combined):,}")
    
    # Final balancing across categories
    final_dfs = []
    # Limit benign to balance with total attack flows
    benign_df = combined[combined["Attack_Category"] == "BENIGN"]
    attack_df = combined[combined["Attack_Category"] != "BENIGN"]
    
    # Keep up to 60,000 benign flows
    max_benign = min(len(benign_df), 60000)
    final_dfs.append(benign_df.sample(n=max_benign, random_state=random_state))
    
    # For attack classes, cap abundant ones to 18,000 each and keep all rare ones
    for cat, group in attack_df.groupby("Attack_Category"):
        if len(group) > max_per_class:
            final_dfs.append(group.sample(n=max_per_class, random_state=random_state))
        else:
            final_dfs.append(group)
            
    final_dataset = pd.concat(final_dfs, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    return final_dataset

def export_samples_and_metadata(df, feature_cols, output_dir=DATA_DIR):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save feature columns list
    feature_path = os.path.join(output_dir, "feature_columns.json")
    with open(feature_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Saved feature columns list: {feature_path} ({len(feature_cols)} features)")
    
    # Export representative sample flows for frontend presets
    categories_to_export = {
        "BENIGN": "sample_benign_flow.json",
        "DDoS": "sample_ddos_flow.json",
        "PortScan": "sample_portscan_flow.json",
        "DoS": "sample_dos_flow.json",
        "Brute Force": "sample_bruteforce_flow.json",
        "Web Attack": "sample_webattack_flow.json",
        "Botnet": "sample_botnet_flow.json",
    }
    
    for cat, filename in categories_to_export.items():
        sample_row = df[df["Attack_Category"] == cat]
        if not sample_row.empty:
            flow_dict = sample_row.iloc[0][feature_cols].to_dict()
            out_path = os.path.join(output_dir, filename)
            with open(out_path, "w") as f:
                json.dump(flow_dict, f, indent=2)
            print(f"Exported preset flow: {out_path} ({cat})")
            
    # Export a small batch of 25 mixed flows for batch upload testing
    sample_batch = df.groupby("Attack_Category", group_keys=False).apply(
        lambda g: g.head(4)
    ).reset_index(drop=True)
    batch_path = os.path.join(output_dir, "sample_batch.csv")
    sample_batch.to_csv(batch_path, index=False)
    print(f"Exported sample batch CSV: {batch_path} ({len(sample_batch)} rows)")

def main():
    print("=" * 60)
    print("AI-BASED ENCRYPTED TRAFFIC THREAT DETECTION: PREPROCESSING")
    print("=" * 60)
    
    dataset = load_and_clean_data(archive_dir=ARCHIVE_DIR, max_per_class=18000)
    
    feature_cols = [c for c in dataset.columns if c not in ["Attack_Category", "Is_Malicious"]]
    
    print("\nDataset Summary:")
    print(f"Total rows: {len(dataset):,}")
    print(f"Total features: {len(feature_cols)}")
    print("\nClass Distribution (Attack_Category):")
    for cat, cnt in dataset["Attack_Category"].value_counts().items():
        pct = (cnt / len(dataset)) * 100
        print(f"  - {cat:16s}: {cnt:6,d} ({pct:5.2f}%)")
        
    print("\nBinary Target (Is_Malicious):")
    for val, cnt in dataset["Is_Malicious"].value_counts().items():
        name = "Threat (Malicious)" if val == 1 else "Benign (Normal)"
        pct = (cnt / len(dataset)) * 100
        print(f"  - {name:20s}: {cnt:6,d} ({pct:5.2f}%)")
        
    # Save full cleaned dataset
    os.makedirs(DATA_DIR, exist_ok=True)
    clean_path = os.path.join(DATA_DIR, "clean_dataset.csv")
    print(f"\nSaving clean balanced dataset to: {clean_path}...")
    dataset.to_csv(clean_path, index=False)
    print("Dataset saved successfully.")
    
    # Export feature list & demo presets
    export_samples_and_metadata(dataset, feature_cols, DATA_DIR)
    print("\nPreprocessing complete!")

if __name__ == "__main__":
    main()
