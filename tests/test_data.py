import os
import pytest
import pandas as pd
from src.utils.helper import load_config, get_project_root
from src.data.data_loader import load_raw_data, validate_dataframe

def test_config_loading():
    config = load_config()
    assert config is not None
    assert "data" in config
    assert "model" in config
    assert "sample_fraction" in config["data"]

def test_data_ingestion():
    config = load_config()
    # Force a very small sample fraction just to test loading quickly
    config_test = config.copy()
    config_test["data"]["sample_fraction"] = 0.001 # 0.1%
    
    df_main, df_diagnosa, df_proc = load_raw_data(config_test, load_sample=True)
    
    assert isinstance(df_main, pd.DataFrame)
    assert isinstance(df_diagnosa, pd.DataFrame)
    assert isinstance(df_proc, pd.DataFrame)
    
    assert df_main.shape[0] > 0
    assert "id" in df_main.columns
    assert "label" in df_main.columns

def test_data_validation():
    config = load_config()
    config_test = config.copy()
    config_test["data"]["sample_fraction"] = 0.001
    
    df_main, _, _ = load_raw_data(config_test, load_sample=True)
    stats = validate_dataframe(df_main, "df_main_test")
    
    assert "num_rows" in stats
    assert "num_cols" in stats
    assert "missing_values" in stats
    assert "duplicate_records" in stats
    assert stats["num_rows"] > 0
