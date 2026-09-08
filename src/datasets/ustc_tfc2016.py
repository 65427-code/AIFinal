"""
USTC-TFC2016 PCAP dataset loader.
"""

import os
import logging
import pandas as pd
from typing import List, Optional

from src.pcap_extractor import extract_flows_from_pcap

logger = logging.getLogger(__name__)

BENIGN_APPS = ['BitTorrent', 'FTP', 'Facetime', 'Gmail', 'MySQL', 'Outlook', 'Skype', 'WorldOfWarcraft']
MALWARE_FAMILIES = ['Cridex', 'Geodo', 'Htbot', 'Miuref', 'Neris', 'Nsis-ay', 'Shifu', 'Tinba', 'Virut', 'Zeus']

def load_ustc_tfc2016(base_dir: str, cache_dir: Optional[str] = None, exclude_families: Optional[List[str]] = None, max_packets: Optional[int] = None) -> pd.DataFrame:
    """
    Load USTC-TFC2016 dataset from PCAPs.
    """
    if exclude_families is None:
        exclude_families = []
        
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    dfs = []
    
    # Check Benign and Malware subdirs
    for category, is_malicious in [('Benign', False), ('Malware', True)]:
        cat_dir = os.path.join(base_dir, category)
        if not os.path.exists(cat_dir):
            logger.warning(f"Directory not found: {cat_dir}")
            continue
            
        # Discover pcaps
        for root, _, files in os.walk(cat_dir):
            for file in files:
                if not file.endswith('.pcap') and not file.endswith('.pcapng'):
                    continue
                    
                pcap_path = os.path.join(root, file)
                filename = file
                
                # Try to guess family from directory or filename
                rel_path = os.path.relpath(root, cat_dir)
                family_name = rel_path if rel_path != '.' else filename.split('.')[0]
                
                # Check exclusions
                if family_name in exclude_families:
                    logger.info(f"Skipping excluded family: {family_name}")
                    continue
                    
                cache_file = None
                if cache_dir:
                    cache_file = os.path.join(cache_dir, f"ustc_{category}_{family_name}_{filename}.parquet")
                    
                df = None
                if cache_file and os.path.exists(cache_file):
                    try:
                        df = pd.read_parquet(cache_file)
                        logger.info(f"Loaded cached {cache_file}")
                    except Exception as e:
                        logger.warning(f"Failed to load cache {cache_file}: {e}")
                        
                if df is None:
                    logger.info(f"Extracting flows from {pcap_path}...")
                    try:
                        df, _meta = extract_flows_from_pcap(pcap_path, max_packets=max_packets, report_progress=True)
                    except Exception as e:
                        logger.error(f"Failed to extract from {pcap_path}: {e}")
                        continue
                if df is not None and not df.empty:
                    df['Dataset'] = 'USTC-TFC2016'
                    df['Capture_ID'] = filename
                    df['Original_Label'] = family_name
                    df['Binary_Label'] = 'MALICIOUS' if is_malicious else 'BENIGN'
                    df['Attack_Category'] = 'Malware' if is_malicious else 'Benign'
                    if is_malicious:
                        df['Attack_Family'] = 'Malware'
                        df['Specific_Family'] = family_name
                    else:
                        df['Attack_Family'] = 'Benign'
                        df['Specific_Family'] = family_name
                        
                    if cache_file and not os.path.exists(cache_file):
                        try:
                            df.to_parquet(cache_file)
                            logger.info(f"  Cached flows to {cache_file}")
                        except Exception as ce:
                            logger.warning(f"  Could not cache: {ce}")
                            
                    dfs.append(df)
                    logger.info(f"  {family_name}: {len(df)} flows extracted")
                    
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)
