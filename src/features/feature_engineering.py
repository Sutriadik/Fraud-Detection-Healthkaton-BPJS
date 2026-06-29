import pandas as pd
import numpy as np
from typing import Tuple
from src.utils.helper import get_logger

logger = get_logger("feature_engineering")

def extract_stay_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes stay duration in days (lama_rawat) from tgldatang and tglpulang,
    then drops the date columns.
    """
    df = df.copy()
    tgldatang = pd.to_datetime(df["tgldatang"])
    tglpulang = pd.to_datetime(df["tglpulang"])
    df["lama_rawat"] = (tglpulang - tgldatang).dt.days
    df.drop(columns=["tgldatang", "tglpulang"], inplace=True, errors="ignore")
    return df

def split_cbg_code(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Splits INA-CBG/KDS codes separated by '-' into 4 parts:
    CMG, tipekasus, spesifikkasus, and severity.
    Optimized version of the notebook's pisah_cbg using vectorized operations.
    """
    # Force convert to string, fill na, split by '-'
    splits = series.astype(str).str.split('-')
    
    def parse_split(val):
        if not isinstance(val, list):
            return '-', '-', '-', '-'
        n = len(val)
        if n == 4:
            return val[0], val[1], val[2], val[3]
        elif n == 3:
            return val[0], '-', val[1], val[2]
        else:
            return '-', '-', '-', '-'
            
    parsed = splits.apply(parse_split)
    
    # Unpack into 4 series
    CMG = parsed.apply(lambda x: x[0])
    tipekasus = parsed.apply(lambda x: x[1])
    spesifikkasus = parsed.apply(lambda x: x[2])
    severity = parsed.apply(lambda x: x[3])
    
    return CMG, tipekasus, spesifikkasus, severity

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies stay duration extraction and splits CBG/KDS codes as done in the notebooks.
    """
    logger.info("Applying feature engineering pipeline...")
    df = df.copy()
    
    # 1. Stay duration
    df = extract_stay_duration(df)
    
    # 2. Split CBG
    logger.info("Splitting CBG codes...")
    cbg_CMG, cbg_tipekasus, cbg_spesifikkasus, cbg_severity = split_cbg_code(df["cbg"])
    df["cbg_CMG"] = cbg_CMG
    df["cbg_tipekasus"] = cbg_tipekasus
    df["cbg_spesifikkasus"] = cbg_spesifikkasus
    df["cbg_severity"] = cbg_severity
    df.drop(columns=["cbg"], inplace=True, errors="ignore")
    
    # 3. Split KDSA
    logger.info("Splitting KDSA codes...")
    kdsa_CMG, kdsa_tipekasus, kdsa_spesifikkasus, kdsa_severity = split_cbg_code(df["kdsa"])
    df["kdsa_CMG"] = kdsa_CMG
    df["kdsa_tipekasus"] = kdsa_tipekasus
    df["kdsa_spesifikkasus"] = kdsa_spesifikkasus
    df["kdsa_severity"] = kdsa_severity
    # Align unknown kdsa severity to III, as done in the notebook
    df.loc[~df["kdsa_severity"].isin(["-", "I", "II", "III"]), "kdsa_severity"] = "III"
    df.drop(columns=["kdsa"], inplace=True, errors="ignore")
    
    # 4. Split KDSP
    logger.info("Splitting KDSP codes...")
    _, _, kdsp_spesifikkasus, kdsp_severity = split_cbg_code(df["kdsp"])
    df["kdsp_spesifikkasus"] = kdsp_spesifikkasus
    df["kdsp_severity"] = kdsp_severity
    df.drop(columns=["kdsp"], inplace=True, errors="ignore")
    
    # 5. Split KDSR
    logger.info("Splitting KDSR codes...")
    _, _, kdsr_spesifikkasus, kdsr_severity = split_cbg_code(df["kdsr"])
    df["kdsr_spesifikkasus"] = kdsr_spesifikkasus
    df["kdsr_severity"] = kdsr_severity
    df.drop(columns=["kdsr"], inplace=True, errors="ignore")
    
    # 6. Split KDSI
    logger.info("Splitting KDSI codes...")
    _, _, kdsi_spesifikkasus, _ = split_cbg_code(df["kdsi"])
    df["kdsi_spesifikkasus"] = kdsi_spesifikkasus
    df.drop(columns=["kdsi"], inplace=True, errors="ignore")
    
    # 7. Split KDSD
    logger.info("Splitting KDSD codes...")
    _, _, kdsd_spesifikkasus, kdsd_severity = split_cbg_code(df["kdsd"])
    df["kdsd_spesifikkasus"] = kdsd_spesifikkasus
    df["kdsd_severity"] = kdsd_severity
    df.drop(columns=["kdsd"], inplace=True, errors="ignore")
    
    # 8. Drop dati2
    df.drop(columns=["dati2"], inplace=True, errors="ignore")
    
    return df
