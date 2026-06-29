import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union
from src.utils.helper import get_logger, load_config, get_project_root
from src.preprocessing.preprocessor import BPJSDataPreprocessor

logger = get_logger("predict")

class InferencePipeline:
    """
    End-to-End inference pipeline for predicting health insurance claim fraud.
    """
    def __init__(self, model_dir: str = None):
        config = load_config()
        root = get_project_root()
        
        # Paths
        if model_dir is None:
            model_dir = os.path.join(root, config["model"]["save_dir"])
            
        model_path = os.path.join(root, config["model"]["model_path"])
        prep_path = os.path.join(root, config["model"]["preprocessor_path"])
        meta_path = os.path.join(root, config["model"]["metadata_path"])
        
        logger.info(f"Loading preprocessor from {prep_path}")
        self.preprocessor = joblib.load(prep_path)
        
        logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)
        
        logger.info(f"Loading metadata from {meta_path}")
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)
            
        self.threshold = self.metadata.get("classification_threshold", 0.7868109)
        logger.info(f"Loaded successfully. Classification threshold is: {self.threshold}")

    def predict_batch(self, df_main: pd.DataFrame, df_diagnosa: pd.DataFrame, df_proc: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts fraud labels and probabilities for batch datasets.
        Returns a DataFrame with columns: id, fraud_probability, is_fraud, confidence_score.
        """
        df_main = df_main.copy()
        # Clean missing values to 'NAN' string to align with preprocessor fit expectations
        str_cols = ['kdsa', 'kdsp', 'kdsr', 'kdsi', 'kdsd', 'diagfktp']
        for col in str_cols:
            if col in df_main.columns:
                df_main[col] = df_main[col].astype(str).str.strip().replace({
                    '-': 'NAN', 'nan': 'NAN', 'None': 'NAN', 'NAN': 'NAN', '': 'NAN'
                })
                
        logger.info(f"Transforming batch claims (size={df_main.shape[0]})...")
        X_prepared = self.preprocessor.transform(df_main, df_diagnosa, df_proc)
        
        logger.info("Generating predictions...")
        probs = self.model.predict_proba(X_prepared)[:, 1]
        preds = (probs >= self.threshold).astype(int)
        
        # Calculate confidence score
        # Since it is a binary classifier, confidence is higher when probability is close to 0 or 1.
        # However, for fraud flagging, we can display the fraud probability directly or map it.
        # Let's define confidence score as:
        # If predicted as fraud (prob >= threshold): scale from threshold..1.0 to 50%..100%
        # If predicted as non-fraud (prob < threshold): scale from 0..threshold to 100%..50%
        confidence = []
        for p, pred in zip(probs, preds):
            if pred == 1:
                # scale p from [threshold, 1.0] to [0.5, 1.0]
                conf = 0.5 + 0.5 * (p - self.threshold) / (1.0 - self.threshold + 1e-6)
            else:
                # scale p from [0, threshold] to [1.0, 0.5] (reverse)
                conf = 1.0 - 0.5 * p / (self.threshold + 1e-6)
            confidence.append(float(conf))
            
        result = pd.DataFrame({
            "id": df_main["id"].values,
            "fraud_probability": probs,
            "is_fraud": preds,
            "confidence_score": confidence
        })
        return result

    def predict_single(self, claim_data: Dict[str, Any], diagnoses: List[Dict[str, Any]], procedures: List[str]) -> Dict[str, Any]:
        """
        Predicts a single claim query.
        
        claim_data: dictionary containing claim details (excluding dates/IDs if passed as fields, but must have 'id').
        diagnoses: list of dictionaries, e.g. [{'diag': 'K29.7', 'levelid': 1}, ...]
        procedures: list of procedure codes, e.g. ['99.04', '88.76']
        """
        claim_id = claim_data.get("id", "single_query")
        
        # 1. Format main claim dictionary as a 1-row DataFrame
        main_dict = claim_data.copy()
        main_dict["id"] = claim_id
        df_main = pd.DataFrame([main_dict])
        
        # 2. Format diagnoses list as a DataFrame
        diag_list = []
        for item in diagnoses:
            diag_list.append({
                "id": claim_id,
                "diag": item.get("diag", ""),
                "levelid": int(item.get("levelid", 1))
            })
        df_diagnosa = pd.DataFrame(diag_list) if diag_list else pd.DataFrame(columns=["id", "diag", "levelid"])
        
        # 3. Format procedures list as a DataFrame
        proc_list = []
        for code in procedures:
            proc_list.append({
                "id": claim_id,
                "proc": code
            })
        df_proc = pd.DataFrame(proc_list) if proc_list else pd.DataFrame(columns=["id", "proc"])
        
        # 4. Predict
        results_df = self.predict_batch(df_main, df_diagnosa, df_proc)
        
        row = results_df.iloc[0]
        prob = float(row["fraud_probability"])
        is_fraud = int(row["is_fraud"])
        conf = float(row["confidence_score"])
        
        # Generate prediction explanation / comments
        explanation = self._generate_explanation(claim_data, diagnoses, procedures, prob, is_fraud)
        
        return {
            "id": claim_id,
            "is_fraud": is_fraud,
            "fraud_probability": prob,
            "confidence_score": conf,
            "explanation": explanation,
            "model_info": {
                "name": self.metadata.get("model_name", "XGBoost"),
                "version": "1.0.0",
                "threshold": self.threshold
            }
        }
        
    def _generate_explanation(self, claim_data: Dict[str, Any], diagnoses: List[Dict[str, Any]], procedures: List[str], prob: float, is_fraud: int) -> str:
        """Generates a text explanation for the fraud prediction based on domain rules."""
        explanations = []
        
        # Rule-based highlights
        biaya = float(claim_data.get("biaya", 0))
        usia = int(claim_data.get("usia", 0))
        
        # Calculate length of stay if dates are present
        tgldatang = pd.to_datetime(claim_data.get("tgldatang", None))
        tglpulang = pd.to_datetime(claim_data.get("tglpulang", None))
        stay_days = 0
        if tgldatang is not None and tglpulang is not None and not pd.isna(tgldatang) and not pd.isna(tglpulang):
            stay_days = (tglpulang - tgldatang).days
            
        primary_diag = next((d.get("diag", "") for d in diagnoses if d.get("levelid", 1) == 1), "")
        num_sec_diag = sum(1 for d in diagnoses if d.get("levelid", 1) == 2)
        num_proc = len(procedures)
        
        if is_fraud == 1:
            explanations.append(f"Klaim ini ditandai sebagai **FRAUD** dengan probabilitas **{prob*100:.1f}%**.")
            
            # Sub-risk factors
            reasons = []
            if biaya > 10000000:
                reasons.append(f"Biaya klaim sangat tinggi (Rp {biaya:,.2f})")
            if stay_days > 10:
                reasons.append(f"Durasi rawat inap lama ({stay_days} hari)")
            if num_sec_diag > 4:
                reasons.append(f"Jumlah diagnosis sekunder banyak ({num_sec_diag} diagnosis)")
            if num_proc > 5:
                reasons.append(f"Jumlah tindakan medis tinggi ({num_proc} tindakan)")
                
            if reasons:
                explanations.append("Faktor risiko utama meliputi: " + ", ".join(reasons) + ".")
            else:
                explanations.append("Model memprediksi fraud berdasarkan interaksi pola CBG/KDS dan diagnosis.")
        else:
            explanations.append(f"Klaim ini diklasifikasikan sebagai **TIDAK FRAUD** (aman) dengan probabilitas fraud sebesar **{prob*100:.1f}%**.")
            explanations.append("Klaim ini dinilai memiliki profil risiko rendah oleh model.")
            
        return " ".join(explanations)
import json
