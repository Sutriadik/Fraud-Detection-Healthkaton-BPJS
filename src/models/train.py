import os
import time
import json
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as imbpipeline

from src.utils.helper import get_logger, load_config, get_project_root
from src.data.data_loader import load_raw_data, validate_dataframe
from src.preprocessing.preprocessor import BPJSDataPreprocessor
from src.models.evaluate import calculate_metrics

logger = get_logger("train")

def run_training():
    start_time = time.time()
    config = load_config()
    root = get_project_root()
    
    # 1. Load data
    logger.info("Loading raw Parquet datasets...")
    df_main, df_diagnosa, df_proc = load_raw_data(config, load_sample=True)
    
    # Validate raw data
    validate_dataframe(df_main, "df_main")
    
    # Check labels distribution
    if "label" in df_main.columns:
        logger.info(f"Target distribution in loaded sample:\n{df_main['label'].value_counts(normalize=True)}")
    
    # 2. Fit Preprocessor
    logger.info("Initializing and fitting preprocessor...")
    preprocessor = BPJSDataPreprocessor()
    preprocessor.fit(df_main, df_diagnosa, df_proc)
    
    # Save preprocessor
    prep_path = os.path.join(root, config["model"]["preprocessor_path"])
    os.makedirs(os.path.dirname(prep_path), exist_ok=True)
    joblib.dump(preprocessor, prep_path)
    logger.info(f"Preprocessor saved to {prep_path}")
    
    # Transform data
    logger.info("Transforming datasets...")
    X_prepared = preprocessor.transform(df_main, df_diagnosa, df_proc)
    y = df_main["label"].values.astype(int)
    
    logger.info(f"Prepared shape: {X_prepared.shape}")
    
    # 3. Train-Val-Test Split
    seed = config["data"]["random_seed"]
    logger.info("Splitting data into Train (60%), Val (20%), and Test (20%)...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_prepared, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=seed, stratify=y_train_full
    )
    
    logger.info(f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")
    
    # 4. Construct imbalanced-learn Pipeline (SMOTE + RUS + XGBoost)
    logger.info("Setting up balancing and XGBoost training pipeline...")
    
    smote_cfg = config["model"]["smote_params"]
    rus_cfg = config["model"]["rus_params"]
    xgb_cfg = config["model"]["xgboost_params"]
    
    over = SMOTE(
        sampling_strategy=smote_cfg["sampling_strategy"],
        k_neighbors=smote_cfg["k_neighbors"],
        random_state=seed
    )
    
    under = RandomUnderSampler(
        sampling_strategy=rus_cfg["sampling_strategy"],
        random_state=seed
    )
    
    # Calculate scale_pos_weight as in notebook: 1 / rus_strategy
    scale_pos_weight = 1.0 / rus_cfg["sampling_strategy"]
    
    # Construct XGBoost Classifier
    model = xgb.XGBClassifier(
        n_estimators=xgb_cfg["n_estimators"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        gamma=xgb_cfg["gamma"],
        learning_rate=xgb_cfg["learning_rate"],
        max_depth=xgb_cfg["max_depth"],
        reg_alpha=xgb_cfg["reg_alpha"],
        reg_lambda=xgb_cfg["reg_lambda"],
        subsample=xgb_cfg["subsample"],
        min_child_weight=xgb_cfg["min_child_weight"],
        scale_pos_weight=scale_pos_weight,
        verbosity=xgb_cfg["verbosity"],
        n_jobs=xgb_cfg["n_jobs"],
        tree_method=xgb_cfg["tree_method"],
        random_state=seed,
        early_stopping_rounds=100,
        eval_metric=["auc", "logloss"]
    )
    
    pipeline = imbpipeline([
        ('over', over),
        ('under', under),
        ('xgb', model)
    ])
    
    # 5. Fit pipeline with early stopping
    logger.info("Fitting model pipeline with early stopping on validation set...")
    # Evaluate on both train and validation sets to capture train learning curves
    pipeline.fit(X_train, y_train, xgb__eval_set=[(X_train, y_train), (X_val, y_val)])
    
    # Extract fitted model
    fitted_xgb = pipeline.named_steps['xgb']
    logger.info(f"Best iteration: {fitted_xgb.best_iteration}, Best score: {fitted_xgb.best_score}")
    
    # Save the standalone XGBoost classifier (keeps inference lightweight)
    model_path = os.path.join(root, config["model"]["model_path"])
    joblib.dump(fitted_xgb, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # 6. Evaluation and Threshold selection
    logger.info("Evaluating model performance on Val and Test sets...")
    
    # Predict probabilities
    y_val_proba = fitted_xgb.predict_proba(X_val)[:, 1]
    y_test_proba = fitted_xgb.predict_proba(X_test)[:, 1]
    
    # Calculate metrics with default threshold (0.5) and target threshold (0.7868)
    th_target = config["model"]["threshold"]
    metrics_val_default = calculate_metrics(y_val, y_val_proba, threshold=0.5)
    metrics_val_target = calculate_metrics(y_val, y_val_proba, threshold=th_target)
    metrics_test_target = calculate_metrics(y_test, y_test_proba, threshold=th_target)
    
    logger.info(f"Val metrics at 0.5 threshold: {metrics_val_default}")
    logger.info(f"Val metrics at {th_target} threshold: {metrics_val_target}")
    logger.info(f"Test metrics at {th_target} threshold: {metrics_test_target}")
    
    # Calculate train vs val curves across iterations for the 4-panel dashboard
    evals = fitted_xgb.evals_result()
    train_loss = evals['validation_0'].get('logloss', [])
    val_loss = evals['validation_1'].get('logloss', [])
    train_auc = evals['validation_0'].get('auc', [])
    val_auc = evals['validation_1'].get('auc', [])
    
    num_iterations = len(train_loss)
    # Calculate around 50 points to keep training fast and generate a smooth curve
    step = max(1, num_iterations // 50)
    iterations_range = list(range(1, num_iterations + 1, step))
    if num_iterations not in iterations_range:
        iterations_range.append(num_iterations)
        
    logger.info("Pre-calculating Precision and Recall curves across iterations...")
    train_prec = []
    train_rec = []
    val_prec = []
    val_rec = []
    
    # Subsample to keep evaluation fast (~10,000 rows is statistically identical to full set)
    eval_tr_size = min(10000, len(X_train))
    idx_tr = np.random.choice(len(X_train), eval_tr_size, replace=False)
    X_tr_eval = X_train[idx_tr]
    y_tr_eval = y_train.values[idx_tr] if hasattr(y_train, 'values') else y_train[idx_tr]
    
    eval_val_size = min(10000, len(X_val))
    idx_val = np.random.choice(len(X_val), eval_val_size, replace=False)
    X_val_eval = X_val[idx_val]
    y_val_eval = y_val.values[idx_val] if hasattr(y_val, 'values') else y_val[idx_val]
    
    for i in iterations_range:
        p_tr = fitted_xgb.predict_proba(X_tr_eval, iteration_range=(0, i))[:, 1]
        p_val = fitted_xgb.predict_proba(X_val_eval, iteration_range=(0, i))[:, 1]
        
        # Train
        pred_tr = (p_tr >= th_target).astype(int)
        tp_tr = np.sum((y_tr_eval == 1) & (pred_tr == 1))
        fp_tr = np.sum((y_tr_eval == 0) & (pred_tr == 1))
        fn_tr = np.sum((y_tr_eval == 1) & (pred_tr == 0))
        train_prec.append(tp_tr / (tp_tr + fp_tr + 1e-10))
        train_rec.append(tp_tr / (tp_tr + fn_tr + 1e-10))
        
        # Val
        pred_val = (p_val >= th_target).astype(int)
        tp_val = np.sum((y_val_eval == 1) & (pred_val == 1))
        fp_val = np.sum((y_val_eval == 0) & (pred_val == 1))
        fn_val = np.sum((y_val_eval == 1) & (pred_val == 0))
        val_prec.append(tp_val / (tp_val + fp_val + 1e-10))
        val_rec.append(tp_val / (tp_val + fn_val + 1e-10))
        
    # 7. Write metadata
    metadata = {
        "model_name": "XGBoost",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": seed,
        "sample_fraction": config["data"]["sample_fraction"],
        "num_features": X_train.shape[1],
        "best_iteration": int(fitted_xgb.best_iteration),
        "classification_threshold": th_target,
        "val_metrics_default_0.5": metrics_val_default,
        "val_metrics_target_threshold": metrics_val_target,
        "test_metrics_target_threshold": metrics_test_target
    }
    
    meta_path = os.path.join(root, config["model"]["metadata_path"])
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Metadata saved to {meta_path}")
    
    # Save evaluation predictions for the Streamlit dashboard
    val_pred_path = os.path.join(root, config["model"]["save_dir"], "val_eval.joblib")
    joblib.dump({
        "y_val": y_val, 
        "y_val_proba": y_val_proba, 
        "y_test": y_test, 
        "y_test_proba": y_test_proba,
        "curve_iterations": iterations_range,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_auc": train_auc,
        "val_auc": val_auc,
        "train_prec": train_prec,
        "train_rec": train_rec,
        "val_prec": val_prec,
        "val_rec": val_rec
    }, val_pred_path)
    logger.info(f"Evaluation predictions saved to {val_pred_path}")
    
    elapsed = time.time() - start_time
    logger.info(f"Training completed successfully in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    run_training()
