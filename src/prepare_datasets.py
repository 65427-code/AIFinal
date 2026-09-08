"""
CLI tool to prepare all datasets.
"""

import os
import sys
# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import logging
import pandas as pd

from src.datasets.cicids2017 import load_cicids2017
from src.datasets.ustc_tfc2016 import load_ustc_tfc2016
from src.datasets.dohbrw2020 import load_dohbrw2020

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for training")
    parser.add_argument('--cicids', type=str, help="Path to CICIDS2017 directory")
    parser.add_argument('--ustc', type=str, help="Path to USTC-TFC2016 directory")
    parser.add_argument('--dohbrw', type=str, help="Path to DoHBrw directory")
    parser.add_argument('--exclude-families', nargs='*', help="Malware families to exclude for held-out testing")
    parser.add_argument('--max-packets-per-pcap', type=int, default=None, help="Max packets to read per PCAP (default: unlimited)")
    parser.add_argument('--max-per-class', type=int, default=None, help="Max samples per class for balancing")
    parser.add_argument('--output-dir', type=str, default='data', help="Output directory base")
    
    args = parser.parse_args()
    
    processed_dir = os.path.join(args.output_dir, 'processed')
    combined_dir = os.path.join(args.output_dir, 'combined')
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)
    
    dfs = []
    
    if args.cicids:
        logger.info(f"Processing CICIDS2017 from {args.cicids}...")
        df_cic = load_cicids2017(args.cicids, max_per_class=args.max_per_class)
        if not df_cic.empty:
            out_path = os.path.join(processed_dir, 'cicids2017.parquet')
            df_cic.to_parquet(out_path)
            dfs.append(df_cic)
            logger.info(f"CICIDS2017 processed: {len(df_cic)} rows")
        else:
            logger.warning("CICIDS2017 returned empty DataFrame")
            
    if args.ustc:
        logger.info(f"Processing USTC-TFC2016 from {args.ustc}...")
        cache_dir = os.path.join(processed_dir, '.ustc_cache')
        df_ustc = load_ustc_tfc2016(
            args.ustc,
            cache_dir=cache_dir,
            exclude_families=args.exclude_families,
            max_packets=args.max_packets_per_pcap
        )
        if not df_ustc.empty:
            out_path = os.path.join(processed_dir, 'ustc_tfc2016.parquet')
            df_ustc.to_parquet(out_path)
            dfs.append(df_ustc)
            logger.info(f"USTC-TFC2016 processed: {len(df_ustc)} rows")
        else:
            logger.warning("USTC-TFC2016 returned empty DataFrame")
            logger.warning("USTC-TFC2016 returned empty DataFrame")
            
    if args.dohbrw:
        logger.info(f"Processing CIRA-CIC-DoHBrw-2020 from {args.dohbrw}...")
        df_doh = load_dohbrw2020(args.dohbrw)
        if not df_doh.empty:
            out_path = os.path.join(processed_dir, 'dohbrw2020.parquet')
            df_doh.to_parquet(out_path)
            dfs.append(df_doh)
            logger.info(f"DoHBrw processed: {len(df_doh)} rows")
        else:
            logger.warning("DoHBrw returned empty DataFrame")
            
    if dfs:
        logger.info("Combining all datasets...")
        
        # Check that we don't have completely disjoint columns between sets
        # by aligning them correctly, pd.concat handles this but missing will be NaN
        combined = pd.concat(dfs, ignore_index=True)
        out_path = os.path.join(combined_dir, 'training_dataset.parquet')
        combined.to_parquet(out_path)
        logger.info(f"Combined dataset saved to {out_path} with {len(combined)} rows")
        
        # Summary
        logger.info("\nDataset Composition Summary:")
        summary = combined.groupby(['Dataset', 'Binary_Label', 'Attack_Category']).size().reset_index(name='Count')
        logger.info("\n" + summary.to_string(index=False))
    else:
        logger.warning("No datasets were successfully processed.")

if __name__ == '__main__':
    main()
