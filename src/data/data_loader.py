import os
import pandas as pd
from typing import Tuple, Dict, Any
from src.utils.helper import get_logger, get_project_root

logger = get_logger("data_loader")

def load_raw_data(config: dict, load_sample: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads raw main, diagnosis, and procedure parquet files.
    If load_sample is True and sample_fraction < 1.0, subsamples the main dataframe.
    """
    root = get_project_root()
    
    # Resolve paths
    main_path = os.path.join(root, config["data"]["raw_main_path"])
    diag_path = os.path.join(root, config["data"]["raw_diagnosa_path"])
    proc_path = os.path.join(root, config["data"]["raw_procedure_path"])
    
    logger.info(f"Loading main dataset from: {main_path}")
    df_main = pd.read_parquet(main_path)
    
    logger.info(f"Loading diagnosis dataset from: {diag_path}")
    df_diagnosa = pd.read_parquet(diag_path)
    
    logger.info(f"Loading procedure dataset from: {proc_path}")
    df_proc = pd.read_parquet(proc_path)
    
    logger.info(f"Main shape: {df_main.shape}, Diagnosa shape: {df_diagnosa.shape}, Procedure shape: {df_proc.shape}")
    
    # Subsampling for speed during training/testing
    sample_frac = config["data"]["sample_fraction"]
    seed = config["data"]["random_seed"]
    if load_sample and sample_frac < 1.0:
        logger.info(f"Subsampling main dataset with fraction: {sample_frac} (seed={seed})")
        df_main = df_main.sample(frac=sample_frac, random_state=seed).reset_index(drop=True)
        # Filter diag and proc to include only IDs that exist in the sampled main
        sampled_ids = set(df_main["id"].unique())
        df_diagnosa = df_diagnosa[df_diagnosa["id"].isin(sampled_ids)].reset_index(drop=True)
        df_proc = df_proc[df_proc["id"].isin(sampled_ids)].reset_index(drop=True)
        logger.info(f"Sampled Main shape: {df_main.shape}, Sampled Diagnosa: {df_diagnosa.shape}, Sampled Procedure: {df_proc.shape}")
        
    return df_main, df_diagnosa, df_proc

def validate_dataframe(df: pd.DataFrame, df_name: str) -> Dict[str, Any]:
    """
    Performs data quality checks: missing values, duplicates, types.
    """
    stats = {}
    stats["num_rows"] = df.shape[0]
    stats["num_cols"] = df.shape[1]
    
    # Missing values
    missing = df.isna().sum()
    stats["missing_values"] = missing[missing > 0].to_dict()
    
    # Duplicates
    num_dups = df.duplicated().sum()
    stats["duplicate_records"] = int(num_dups)
    
    # Data types
    stats["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    logger.info(f"Validation summary for {df_name}: {stats['num_rows']} rows, {stats['num_cols']} cols, {stats['duplicate_records']} duplicates.")
    return stats
