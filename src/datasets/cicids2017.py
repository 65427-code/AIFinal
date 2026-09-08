"""
CICIDS2017 dataset loader.
"""

import os
import glob
import pandas as pd
import numpy as np
import logging

from src.features.aliases import normalize_column_name

logger = logging.getLogger(__name__)

# Map labels to Binary_Label (BENIGN/MALICIOUS) and Attack_Category
LABEL_MAPPING = {
    'BENIGN': ('BENIGN', 'Benign'),
    'DDoS': ('MALICIOUS', 'DDoS'),
    'PortScan': ('MALICIOUS', 'PortScan'),
    'DoS Hulk': ('MALICIOUS', 'DoS'),
    'DoS GoldenEye': ('MALICIOUS', 'DoS'),
    'DoS slowloris': ('MALICIOUS', 'DoS'),
    'DoS Slowhttptest': ('MALICIOUS', 'DoS'),
    'Heartbleed': ('MALICIOUS', 'DoS'),
    'FTP-Patator': ('MALICIOUS', 'Brute Force'),
    'SSH-Patator': ('MALICIOUS', 'Brute Force'),
    'Bot': ('MALICIOUS', 'Botnet'),
    'Infiltration': ('MALICIOUS', 'Infiltration'),
}

ZERO_VARIANCE_COLS = [
    'Fwd Header Length.1', 'Bwd PSH Flags', 'Fwd URG Flags', 
    'Bwd URG Flags', 'RST Flag Count', 'CWE Flag Count', 'ECE Flag Count'
]

def load_cicids2017(archive_dir: str, max_per_class: int = None) -> pd.DataFrame:
    """
    Load CICIDS2017 dataset from CSV files or a single preprocessed CSV.
    """
    if os.path.isfile(archive_dir):
        csv_files = [archive_dir]
    elif os.path.isdir(archive_dir):
        csv_files = glob.glob(os.path.join(archive_dir, '*.csv'))
    else:
        csv_files = []
        
    if not csv_files:
        logger.warning(f"No CSV files found at {archive_dir}")
        return pd.DataFrame()
        
    dfs = []
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        logger.info(f"Processing {filename}...")
        try:
            df = pd.read_csv(file_path, encoding='cp1252', low_memory=False)
            
            # Strip whitespace
            df.columns = [str(c).strip() for c in df.columns]
            
            # Check if this is already preprocessed (has Attack_Category or Is_Malicious)
            if 'Attack_Category' in df.columns:
                df['Dataset'] = 'CICIDS2017'
                df['Capture_ID'] = filename
                df['Original_Label'] = df['Attack_Category']
                df['Binary_Label'] = df['Attack_Category'].apply(lambda x: 'BENIGN' if str(x).upper() == 'BENIGN' else 'MALICIOUS')
                dfs.append(df)
                continue
                
            # Find Label column
            label_col = None
            for c in df.columns:
                if c.lower() == 'label':
                    label_col = c
                    break
            
            if not label_col:
                logger.warning(f"No label column found in {filename}, skipping.")
                continue
                
            # Normalize column names except label
            rename_map = {}
            for c in df.columns:
                if c != label_col:
                    rename_map[c] = normalize_column_name(c)
            df = df.rename(columns=rename_map)
            
            # Metadata
            df['Dataset'] = 'CICIDS2017'
            df['Capture_ID'] = filename
            
            # Drop the original label column to avoid duplication
            original_labels = df[label_col].copy()
            df = df.drop(columns=[label_col])
            df['Original_Label'] = original_labels
            
            def map_label(lbl):
                lbl_str = str(lbl).strip()
                if lbl_str in LABEL_MAPPING:
                    return LABEL_MAPPING[lbl_str]
                # Try partial match for web attacks
                if 'Web Attack' in lbl_str:
                    return ('MALICIOUS', 'Web Attack')
                if lbl_str.lower() == 'benign':
                    return ('BENIGN', 'Benign')
                return ('MALICIOUS', 'Other Attack')
                
            mapped = df['Original_Label'].apply(map_label)
            df['Binary_Label'] = [x[0] for x in mapped]
            df['Attack_Category'] = [x[1] for x in mapped]
            
            # Clean numeric
            feature_cols = [c for c in df.columns if c not in ['Dataset', 'Capture_ID', 'Original_Label', 'Binary_Label', 'Attack_Category']]
            for c in feature_cols:
                df[c] = pd.to_numeric(df[c], errors='coerce')
                
            df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
            df = df.dropna(subset=feature_cols)
            
            # Remove zero-variance and bulk
            cols_to_drop = [c for c in ZERO_VARIANCE_COLS if c in df.columns]
            bulk_cols = [c for c in df.columns if 'Bulk' in c or 'bulk' in c.lower()]
            cols_to_drop.extend([c for c in bulk_cols if c not in cols_to_drop])
            
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                
            dfs.append(df)
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            
    if not dfs:
        return pd.DataFrame()
        
    combined = pd.concat(dfs, ignore_index=True)
    
    if max_per_class is not None and max_per_class > 0:
        logger.info(f"Sampling max {max_per_class} per class...")
        sampled_dfs = []
        for label, group in combined.groupby('Original_Label'):
            if len(group) > max_per_class:
                sampled_dfs.append(group.sample(n=max_per_class, random_state=42))
            else:
                sampled_dfs.append(group)
        combined = pd.concat(sampled_dfs, ignore_index=True)
        
    return combined
