"""
CIRA-CIC-DoHBrw-2020 dataset loader.
"""

import os
import pandas as pd
import numpy as np
import logging

from src.features.schema import CANONICAL_FEATURES

logger = logging.getLogger(__name__)

def load_dohbrw2020(base_dir: str, max_per_class: Optional[int] = 25000) -> pd.DataFrame:
    """
    Load CIRA-CIC-DoHBrw-2020 dataset and map to canonical features.
    """
    l1_path = os.path.join(base_dir, 'L1-DoH-NonDoH.parquet')
    l2_path = os.path.join(base_dir, 'L2-BenignDoH-MaliciousDoH.parquet')
    
    dfs = []
    
    if os.path.exists(l1_path):
        try:
            logger.info(f"Loading {l1_path}...")
            df_l1 = pd.read_parquet(l1_path)
            # Filter for NonDoH (benign HTTPS)
            if 'Label' in df_l1.columns:
                df_l1 = df_l1[df_l1['Label'] == 'NonDoH'].copy()
                df_l1['Original_Label'] = 'NonDoH'
                df_l1['Binary_Label'] = 'BENIGN'
                if max_per_class and len(df_l1) > max_per_class:
                    df_l1 = df_l1.sample(n=max_per_class, random_state=42)
                dfs.append(df_l1)
                logger.info(f"  Sampled {len(df_l1)} NonDoH benign flows")
            else:
                logger.warning("No Label column in L1 dataset.")
        except Exception as e:
            logger.error(f"Failed to load {l1_path}: {e}")
    else:
        logger.warning(f"Path not found: {l1_path}")
        
    if os.path.exists(l2_path):
        try:
            logger.info(f"Loading {l2_path}...")
            df_l2 = pd.read_parquet(l2_path)
            if 'Label' in df_l2.columns:
                df_l2['Original_Label'] = df_l2['Label']
                df_l2['Binary_Label'] = df_l2['Label'].apply(lambda x: 'MALICIOUS' if str(x).lower() == 'malicious' else 'BENIGN')
                if max_per_class:
                    sampled_l2 = []
                    for lbl, grp in df_l2.groupby('Binary_Label'):
                        if len(grp) > max_per_class:
                            sampled_l2.append(grp.sample(n=max_per_class, random_state=42))
                        else:
                            sampled_l2.append(grp)
                    df_l2 = pd.concat(sampled_l2, ignore_index=True)
                dfs.append(df_l2)
                logger.info(f"  Sampled {len(df_l2)} L2 DoH flows")
            else:
                logger.warning("No Label column in L2 dataset.")
        except Exception as e:
            logger.error(f"Failed to load {l2_path}: {e}")
    else:
        logger.warning(f"Path not found: {l2_path}")
        
    if not dfs:
        return pd.DataFrame()
        
    combined = pd.concat(dfs, ignore_index=True)
    
    # Map features
    mapped_df = pd.DataFrame(index=combined.index)
    
    # Default all canonical to 0.0
    for col in CANONICAL_FEATURES:
        mapped_df[col] = 0.0
        
    # Semantic mapping
    mapping = {
        'Duration': ('Flow Duration', lambda x: x * 1e6), # seconds to microseconds
        'PacketLengthVariance': ('Packet Length Variance', None),
        'PacketLengthStandardDeviation': ('Packet Length Std', None),
        'PacketLengthMean': ('Packet Length Mean', None),
        'FlowBytesSent': ('Total Length of Fwd Packets', None),
        'FlowBytesReceived': ('Total Length of Bwd Packets', None),
        'FlowSentRate': ('Flow Bytes/s', None), # approximate
    }
    
    for doh_col, (can_col, transform) in mapping.items():
        if doh_col in combined.columns and can_col in CANONICAL_FEATURES:
            if transform:
                mapped_df[can_col] = transform(combined[doh_col])
            else:
                mapped_df[can_col] = combined[doh_col]
                
    # Metadata
    mapped_df['Dataset'] = 'CIRA-CIC-DoHBrw-2020'
    mapped_df['Original_Label'] = combined['Original_Label']
    mapped_df['Binary_Label'] = combined['Binary_Label']
    
    attack_cat = []
    for lbl in combined['Binary_Label']:
        if lbl == 'MALICIOUS':
            attack_cat.append('Encrypted Tunnel')
        else:
            attack_cat.append('Benign')
            
    mapped_df['Attack_Category'] = attack_cat
    mapped_df['Is_Encrypted'] = True
    
    return mapped_df
