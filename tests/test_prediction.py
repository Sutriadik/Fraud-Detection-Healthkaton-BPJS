import os
import pytest
import pandas as pd
from src.prediction.predict import InferencePipeline

@pytest.fixture
def mock_claim():
    claim_data = {
        "id": "claim_test_101",
        "id_peserta": "peserta_test_101",
        "jenispulang": "1",
        "jenkel": "P",
        "pisat": "1",
        "diagfktp": "E11.9",
        "biaya": 4500000.0,
        "dati2": "291",
        "typefaskes": "B",
        "usia": 55,
        "tgldatang": "2022-06-10",
        "tglpulang": "2022-06-14",
        "jenispel": "1",
        "cbg": "E-4-10-I",
        "kelasrawat": 2,
        "kdsa": "-",
        "kdsp": "-",
        "kdsr": "-",
        "kdsi": "-",
        "kdsd": "-"
    }
    diagnoses = [
        {"diag": "E11.9", "levelid": 1},
        {"diag": "I10", "levelid": 2}
    ]
    procedures = ["99.04", "88.78"]
    return claim_data, diagnoses, procedures

def test_inference_pipeline_single(mock_claim):
    # Only test if model has been successfully trained and saved
    if not os.path.exists("models/final_model.joblib"):
        pytest.skip("Model not trained yet.")
        
    pipeline = InferencePipeline()
    claim_data, diagnoses, procedures = mock_claim
    
    result = pipeline.predict_single(claim_data, diagnoses, procedures)
    
    assert "id" in result
    assert "is_fraud" in result
    assert "fraud_probability" in result
    assert "confidence_score" in result
    assert "explanation" in result
    assert "model_info" in result
    
    assert result["is_fraud"] in (0, 1)
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert 0.5 <= result["confidence_score"] <= 1.0
    assert isinstance(result["explanation"], str)

def test_inference_pipeline_batch(mock_claim):
    if not os.path.exists("models/final_model.joblib"):
        pytest.skip("Model not trained yet.")
        
    pipeline = InferencePipeline()
    claim_data, diagnoses, procedures = mock_claim
    
    df_main = pd.DataFrame([claim_data])
    
    diag_list = [{"id": claim_data["id"], "diag": d["diag"], "levelid": d["levelid"]} for d in diagnoses]
    df_diagnosa = pd.DataFrame(diag_list)
    
    proc_list = [{"id": claim_data["id"], "proc": p} for p in procedures]
    df_proc = pd.DataFrame(proc_list)
    
    results = pipeline.predict_batch(df_main, df_diagnosa, df_proc)
    
    assert isinstance(results, pd.DataFrame)
    assert results.shape[0] == 1
    assert "id" in results.columns
    assert "fraud_probability" in results.columns
    assert "is_fraud" in results.columns
    assert "confidence_score" in results.columns
