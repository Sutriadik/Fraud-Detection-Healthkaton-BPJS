# Healthcare Insurance Fraud Detection Production System

This repository contains a production-grade, end-to-end Machine Learning system to verify healthcare insurance claims and detect potential fraud (kecurangan). The project utilizes claim datasets from the National Healthcare Insurance (BPJS Kesehatan) and implements domain-specific feature engineering, balancing techniques, and threshold tuning to optimize prediction precision.

This system is built for deployment on **Streamlit Cloud** and is structured according to professional software engineering best practices.

---

## 🧭 Project Structure

```
health-fraud-detection/
├── app/
│   ├── Home.py                             # Main landing page for Streamlit Dashboard
│   ├── pages/
│   │   ├── 1_📊_Dataset_Overview.py        # Descriptive statistics, costs, demographics
│   │   ├── 2_📈_Model_Performance.py       # Tuning strategy, PR/ROC curves, Confusion Matrix
│   │   ├── 3_🔍_Fraud_Prediction.py        # Claims verification (Manual Form & Bulk CSV)
│   │   └── 4_ℹ️_About_Project.py           # Technical architecture, INA-CBG, fraud typologies
│   └── assets/                             # Saved UI assets or custom CSS
│
├── configs/
│   └── config.yaml                         # Centralized configuration (hyperparameters, paths, sample rates)
│
├── data/
│   ├── raw/                                # Raw Parquet datasets (ignored in git)
│   └── processed/                          # Intermediate/engineered data
│
├── models/
│   ├── final_model.joblib                  # Serialized XGBoost model binary
│   ├── preprocessor_pipeline.joblib         # ColumnTransformer preprocessor (One-Hot, Ordinal)
│   ├── val_eval.joblib                     # Raw validation predictions for visualization
│   └── model_metadata.json                 # Metrics, training timestamps, parameters
│
├── notebooks/
│   ├── Healthkathon_Classification_Model.ipynb  # Primary development notebook
│   └── bpjs_fraud_detection.ipynb               # Alternative neural network experiments
│
├── src/                                    # Source code package
│   ├── data/
│   │   └── data_loader.py                  # Ingestion & validation of raw Parquet tables
│   ├── preprocessing/
│   │   └── preprocessor.py                 # ICD-10 chapter mappings, diagnosis cleaning
│   ├── features/
│   │   └── feature_engineering.py          # Stay duration, INA-CBG / KDS splitting
│   ├── models/
│   │   ├── train.py                        # Model training (SMOTE + RUS + XGBoost)
│   │   └── evaluate.py                     # Metric calculation helper
│   └── prediction/
│       └── predict.py                      # End-to-End inference pipeline wrapper
│
├── tests/                                  # Automated tests
│   ├── test_data.py                        # Ingestion, validation, config tests
│   └── test_prediction.py                  # Single & batch inference tests
│
├── requirements.txt                        # Python dependencies
├── README.md                               # Project documentation
└── .gitignore                              # Git exclusion rules
```

---

## ⚡ Quick Start

### 1. Installation

Clone this repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Automated Tests

Execute the test suite using `pytest` to verify the pipeline's integrity:
```bash
PYTHONPATH=. pytest tests/
```

### 3. Run Training Pipeline

If you wish to retrain the model, configure `configs/config.yaml` to set your desired dataset size fraction (e.g., `sample_fraction: 0.02` for 2% sample, or `1.0` for full training) and run:
```bash
PYTHONPATH=. python3 src/models/train.py
```

### 4. Run Streamlit Web Application Locally

Launch the user interface to verify claims manually or in batches:
```bash
streamlit run app/Home.py
```

---

## 🧠 Machine Learning & Domain Engineering Details

### 1. INA-CBG & ICD-10 Feature Engineering
* **CBG / KDS Splitting:** We decompose complex INA-CBG and KDS codes into separate variables: *CMG (Case-Mix Main Group)*, *Case Type*, *Specific Case*, and *Severity Level* according to Ministry of Health (*Permenkes No. 27 Tahun 2014*).
* **ICD-10 Chapter Mapping:** We parse the first letter and numbers of diagnosis codes and group them into their official ICD-10 chapters (I to XXII), allowing the tree model to learn high-level organ-system correlations.
* **Stay Duration:** Computes hospitalization length in days (`lama_rawat`) from admission dates.

### 2. Handling Imbalanced Datasets
Healthcare fraud datasets are highly imbalanced (~1.4% fraud cases). To address this, the pipeline applies:
1. **SMOTE (Synthetic Minority Over-sampling Technique):** Oversamples the minority fraud cases to a ratio of `0.197`.
2. **Random Under-Sampler (RUS):** Downsamples the majority non-fraud class to reach a ratio of `0.885`.
3. **Scale Pos Weight:** Sets the XGBoost positive class scaling weight to `1 / 0.885 = 1.13` to balance tree gains.

### 3. High-Precision Threshold Tuning
* In healthcare audits, false positives (flagging an honest claim as fraud) delay hospital payments and damage stakeholder relationships.
* Therefore, the pipeline tunes the decision threshold of the probability predictions. By shifting the threshold from the default `0.50` to **`0.7868`**, we achieve **~90.0% precision** on the validation set while preserving **~40.0% recall** of actual fraud cases.

---

## 🖥️ Streamlit App Features

* **🏠 Home:** Dashboard landing page displaying high-level system summaries and metadata.
* **📊 Dataset Overview:** Interactive visualization of demographic distribution, claim costs, and imbalance ratios.
* **📈 Model Performance:** Displays ROC/PR curves, a dynamic Confusion Matrix, and explains the business logic of threshold tuning.
* **🔍 Fraud Prediction:**
  * **Manual Form:** Fill out a claims ticket with drop-down menus and get instant predictions, confidence scores, and recommendations.
  * **Bulk CSV Upload:** Drag-and-drop a CSV claims sheet containing raw columns to predict batch claims, download prediction summaries, and export reports.
* **ℹ️ About Project:** Explains regulatory INA-CBG structures and details fraud typologies (upcoding, phantom billing, unbundling).
