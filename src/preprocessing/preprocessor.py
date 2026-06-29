import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.utils.helper import get_logger, get_project_root
from src.features.feature_engineering import apply_feature_engineering

logger = get_logger("preprocessor")

def clean_diagnosis_code(series: pd.Series) -> pd.Series:
    """
    Normalizes diagnosis codes. Capitalizes, strips whitespace, corrects typos
    and maps invalid codes to Z09.8, matching the notebook logic.
    """
    s = series.astype(str).str.strip().str.upper()
    s = s.replace("H8.14", "H81.4")
    s = s.replace("ZO3.9", "Z03.9")
    
    # Fast apply helper
    def clean_code(val):
        if val in ('H81.4', 'Z03.9'):
            return val
        if len(val) >= 3:
            if val[0].isdigit():
                return 'Z09.8'
            if not val[1:3].isdigit():
                return 'Z09.8'
        elif len(val) > 0:
            if val[0].isdigit():
                return 'Z09.8'
        return val
        
    s_cleaned = s.apply(clean_code)
    return s_cleaned.str.slice(0, 3)

def map_icd10_chapter(letters: pd.Series, nums: pd.Series) -> pd.Series:
    """
    Vectorized mapping of ICD-10 letters and numbers to chapters (I to XXII).
    Matches the logic of Code Cell 92.
    """
    chapters = pd.Series('XXII', index=letters.index)
    
    chapters.loc[letters.isin(['A', 'B'])] = 'I'
    chapters.loc[(letters == 'C') | ((letters == 'D') & (nums <= 48))] = 'II'
    chapters.loc[(letters == 'D') & (nums >= 50)] = 'III'
    chapters.loc[letters == 'E'] = 'IV'
    chapters.loc[letters == 'F'] = 'V'
    chapters.loc[letters == 'G'] = 'VI'
    chapters.loc[(letters == 'H') & (nums <= 59)] = 'VII'
    chapters.loc[(letters == 'H') & (nums >= 60)] = 'VIII'
    chapters.loc[letters == 'I'] = 'IX'
    chapters.loc[letters == 'J'] = 'X'
    chapters.loc[letters == 'K'] = 'XI'
    chapters.loc[letters == 'L'] = 'XII'
    chapters.loc[letters == 'M'] = 'XIII'
    chapters.loc[letters == 'N'] = 'XIV'
    chapters.loc[letters == 'O'] = 'XV'
    chapters.loc[letters == 'P'] = 'XVI'
    chapters.loc[letters == 'Q'] = 'XVII'
    chapters.loc[letters == 'R'] = 'XVIII'
    chapters.loc[letters.isin(['S', 'T'])] = 'XIX'
    chapters.loc[letters.isin(['V', 'X', 'Y'])] = 'XX'
    chapters.loc[letters == 'Z'] = 'XXI'
    
    return chapters

class BPJSDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    End-to-End preprocessor for BPJS claims data.
    Fits and manages imputers and encoders.
    """
    def __init__(self):
        # Encoders
        self.imputer = None
        self.encoder = None
        
        # Saved state for secondary merge-imputations
        self.diagfkrtl_letter_imputer = None
        self.diagfkrtl_num_imputer = None
        self.diagfkrtl_sekunder_imputer = None
        
        # Categorical configurations
        self.CMG_categories = ['-', 'G', 'H', 'U', 'J', 'I', 'K', 'B', 'M', 'L', 'E', 'N', 'V',
                              'W', 'O', 'P', 'D', 'C', 'A', 'F', 'T', 'S', 'Z', 'Q', 'QP',
                              'SA', 'ST', 'SF', 'YY', 'DD', 'II', 'IJ', 'RR', 'CD', 'X']
        self.tipekasus_categories = [str(x) for x in range(0, 10)]
        self.tipekasus_categories.insert(0, '-')
        self.spesifikkasus_categories = [str(x) for x in range(1, 100)]
        self.spesifikkasus_categories.insert(0, '-')
        self.severity_categories = ['-', '0', 'I', 'II', 'III']
        self.icd10_categories = ['I', 'II', 'III', 'IV', 'IX', 'V', 'VI', 'VII', 'VIII',
                                'X', 'XI', 'XII', 'XIII', 'XIV', 'XIX', 'XV', 'XVI',
                                'XVII', 'XVIII', 'XX', 'XXI', 'XXII']
        
        # Features definition
        self.known_cat_col = ['cbg_CMG', 'cbg_tipekasus', 'cbg_spesifikkasus',
                             'kdsa_CMG', 'kdsa_tipekasus', 'kdsa_spesifikkasus',
                             'kdsp_spesifikkasus', 'kdsr_spesifikkasus',
                             'kdsi_spesifikkasus', 'kdsd_spesifikkasus',
                             'diagfktp_icd10', 'diagfkrtl_icd10']
        
        self.unknown_cat_col = ['typefaskes']
        
        self.ordinal_col = ['jenkel', 'cbg_severity', 'kdsa_severity',
                            'kdsp_severity', 'kdsr_severity', 'kdsd_severity']
        
        self.remainder_col = ['jenispulang', 'pisat', 'biaya', 'usia', 'jenispel', 
                              'kelasrawat', 'lama_rawat', 'diagfkrtl_sekunder_counts', 'proc_count']

    def clean_main_table(self, df: pd.DataFrame, is_fit: bool = False) -> pd.DataFrame:
        """Cleans strings, drops politujuan, handles main table imputations."""
        df = df.copy()
        
        # Split target label if present, to prevent it from entering ColumnTransformer
        label_col = None
        if 'label' in df.columns:
            label_col = df['label']
            df.drop(columns=['label'], inplace=True)
            
        # 1. Clean format (uppercase, strip)
        str_cols = ['typefaskes', 'jenkel', 'politujuan', 'diagfktp', 'cbg', 'kdsa', 'kdsp', 'kdsr', 'kdsi', 'kdsd']
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
                
        # Drop politujuan
        if 'politujuan' in df.columns:
            df.drop(columns=['politujuan'], inplace=True)
            
        # 2. Main imputations
        impute_cols_cat = ['jenispulang', 'jenkel', 'pisat', 'diagfktp', 'kdsa', 'kdsp', 'kdsr', 'kdsi', 'kdsd']
        
        # Convert to numpy array as done in the notebook for fit
        if is_fit:
            self.imputer = ColumnTransformer([
                ('most_freq', SimpleImputer(strategy='most_frequent'), impute_cols_cat),
                ('mean', SimpleImputer(strategy='median'), ['biaya'])
            ], remainder='passthrough')
            
            self.imputer.fit(df)
            
        # Transform and mapping back to DataFrame (matching ColumnTransformer order)
        arr = self.imputer.transform(df)
        
        # Recreate columns after ColumnTransformer which puts imputed cols first
        columns_after_imputer = impute_cols_cat + ['biaya']
        remainder_cols = [c for c in df.columns if c not in impute_cols_cat + ['biaya']]
        df_imputed = pd.DataFrame(arr, columns=columns_after_imputer + remainder_cols)
        
        # Cast numerical fields
        df_imputed['biaya'] = pd.to_numeric(df_imputed['biaya'])
        df_imputed['usia'] = pd.to_numeric(df_imputed['usia'])
        df_imputed['kelasrawat'] = pd.to_numeric(df_imputed['kelasrawat'])
        
        # Re-add label if it was present
        if label_col is not None:
            df_imputed['label'] = label_col.values
            
        return df_imputed

    def clean_diagnosis_table(self, df_diagnosa: pd.DataFrame) -> pd.DataFrame:
        """Processes primary and secondary diagnosis data and combines them."""
        df_diagnosa = df_diagnosa.drop_duplicates(ignore_index=True)
        
        # Level 1 - Primary diagnosis
        df_primer = df_diagnosa[df_diagnosa["levelid"] == 1].copy()
        df_primer.drop(columns=["levelid"], inplace=True, errors="ignore")
        # Keep only last duplicate per ID
        df_primer = df_primer[~df_primer["id"].duplicated(keep='last')].reset_index(drop=True)
        # Rename diagnosis column
        col_name = "kddiag" if "kddiag" in df_primer.columns else "diag"
        df_primer.rename(columns={col_name: "diag"}, inplace=True)
        
        # Clean diagnosis codes
        df_primer["diag"] = clean_diagnosis_code(df_primer["diag"])
        
        # Level 2 - Secondary diagnosis
        df_sekunder = df_diagnosa[df_diagnosa["levelid"] == 2].copy()
        df_sekunder.drop(columns=["levelid"], inplace=True, errors="ignore")
        
        # Count secondary diagnoses
        col_name_sek = "kddiag" if "kddiag" in df_sekunder.columns else "diag"
        diag_counts = df_sekunder["id"].value_counts()
        df_sekunder_agg = pd.DataFrame({
            "id": diag_counts.index,
            "diagfkrtl_sekunder_counts": diag_counts.values
        })
        
        # Merge primary and secondary
        df_diagnosa_prepared = df_primer.merge(df_sekunder_agg, how='left', on='id')
        df_diagnosa_prepared["diagfkrtl_sekunder_counts"] = df_diagnosa_prepared["diagfkrtl_sekunder_counts"].fillna(0).astype(np.int64)
        
        return df_diagnosa_prepared

    def process_data(self, df_main: pd.DataFrame, df_diagnosa: pd.DataFrame, df_proc: pd.DataFrame, is_fit: bool = False) -> pd.DataFrame:
        """
        Cleans, joins and extracts ICD-10 chapters on the claim, diagnosis, and procedure datasets.
        """
        logger.info("Step 1/4: Cleaning main table...")
        df = self.clean_main_table(df_main, is_fit=is_fit)
        
        # Clean diagfktp column first
        df["diagfktp"] = clean_diagnosis_code(df["diagfktp"])
        
        # Apply stays and CBG/KDS code splitting
        logger.info("Step 2/4: Slicing dates and CBG/KDS codes...")
        df = apply_feature_engineering(df)
        
        # Process and merge diagnosis table
        logger.info("Step 3/4: Processing and merging diagnosis table...")
        df_diagnosa_prepared = self.clean_diagnosis_table(df_diagnosa)
        
        # Split FKRTL primary diagnosis into letter and number before merge
        df_diagnosa_prepared['diagfkrtl_letter'] = df_diagnosa_prepared['diag'].str.slice(0, 1)
        df_diagnosa_prepared['diagfkrtl_num'] = pd.to_numeric(df_diagnosa_prepared['diag'].str.slice(1), errors='coerce')
        df_diagnosa_prepared.drop(columns=['diag'], inplace=True)
        
        df = df.merge(df_diagnosa_prepared, how='left', on='id')
        
        # Post-merge imputation
        if is_fit:
            self.diagfkrtl_letter_imputer = SimpleImputer(strategy='most_frequent')
            self.diagfkrtl_num_imputer = SimpleImputer(strategy='median')
            self.diagfkrtl_sekunder_imputer = SimpleImputer(strategy='median')
            
            # Fit
            self.diagfkrtl_letter_imputer.fit(df[['diagfkrtl_letter']])
            self.diagfkrtl_num_imputer.fit(df[['diagfkrtl_num']])
            self.diagfkrtl_sekunder_imputer.fit(df[['diagfkrtl_sekunder_counts']])
            
        df['diagfkrtl_letter'] = self.diagfkrtl_letter_imputer.transform(df[['diagfkrtl_letter']]).ravel()
        df['diagfkrtl_num'] = pd.to_numeric(self.diagfkrtl_num_imputer.transform(df[['diagfkrtl_num']]).ravel())
        df['diagfkrtl_sekunder_counts'] = pd.to_numeric(self.diagfkrtl_sekunder_imputer.transform(df[['diagfkrtl_sekunder_counts']]).ravel())
        
        # Split diagfktp letter/num
        df['diagfktp_letter'] = df['diagfktp'].str.slice(0, 1)
        df['diagfktp_num'] = pd.to_numeric(df['diagfktp'].str.slice(1), errors='coerce').fillna(0).astype(np.int64)
        df.drop(columns=['diagfktp'], inplace=True)
        
        # Convert numeric components
        df['diagfkrtl_num'] = df['diagfkrtl_num'].fillna(0).astype(np.int64)
        
        # Create ICD-10 chapters
        df['diagfktp_icd10'] = map_icd10_chapter(df['diagfktp_letter'], df['diagfktp_num'])
        df['diagfkrtl_icd10'] = map_icd10_chapter(df['diagfkrtl_letter'], df['diagfkrtl_num'])
        
        # Drop temporary letter/num fields
        df.drop(columns=['diagfktp_letter', 'diagfktp_num', 'diagfkrtl_letter', 'diagfkrtl_num'], inplace=True)
        
        # Process and merge procedure table
        logger.info("Step 4/4: Processing and merging procedure table...")
        df_proc = df_proc.drop_duplicates()
        proc_counts = df_proc["id"].value_counts()
        df_proc_prepared = pd.DataFrame({
            "id": proc_counts.index,
            "proc_count": proc_counts.values
        })
        
        df = df.merge(df_proc_prepared, how='left', on='id')
        df["proc_count"] = df["proc_count"].fillna(0).astype(np.int64)
        
        # Force numeric types for numerical columns
        df['lama_rawat'] = pd.to_numeric(df['lama_rawat']).astype(np.int64)
        df['diagfkrtl_sekunder_counts'] = pd.to_numeric(df['diagfkrtl_sekunder_counts']).astype(np.int64)
        
        # Clean target label type if present
        if 'label' in df.columns:
            df['label'] = pd.to_numeric(df['label']).fillna(0).astype(int)
            
        return df

    def fit(self, df_main: pd.DataFrame, df_diagnosa: pd.DataFrame, df_proc: pd.DataFrame, y=None):
        """Fits the preprocess pipeline components on the raw training datasets."""
        # 1. Apply structural preprocessing
        df_structured = self.process_data(df_main, df_diagnosa, df_proc, is_fit=True)
        
        # 2. Extract features matrix X
        X = df_structured.drop(columns=['id', 'id_peserta', 'label'], errors='ignore')
        
        # 3. Setup encoders
        # OneHot for known categories
        onehot_known_enc = OneHotEncoder(
            categories=[
                self.CMG_categories, self.tipekasus_categories, self.spesifikkasus_categories,  # cbg
                self.CMG_categories, self.tipekasus_categories, self.spesifikkasus_categories,  # kdsa
                self.spesifikkasus_categories, self.spesifikkasus_categories,  # kdsp, kdsr
                self.spesifikkasus_categories, self.spesifikkasus_categories,  # kdsi, kdsd
                self.icd10_categories, self.icd10_categories  # diagfktp, diagfkrtl
            ],
            handle_unknown='ignore',
            sparse_output=False
        )
        
        # OneHot for unknown categories (typefaskes)
        onehot_unk_enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        # Ordinal Encoder
        ordinal_enc = OrdinalEncoder(
            categories=[
                ['L', 'P'],  # jenkel
                self.severity_categories,  # cbg_severity
                self.severity_categories,  # kdsa_severity
                self.severity_categories,  # kdsp_severity
                self.severity_categories,  # kdsr_severity
                self.severity_categories   # kdsd_severity
            ],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        
        # Compose ColumnTransformer
        self.encoder = ColumnTransformer([
            ('onehot1', onehot_known_enc, self.known_cat_col),
            ('onehot2', onehot_unk_enc, self.unknown_cat_col),
            ('ordinal_enc', ordinal_enc, self.ordinal_col)
        ], remainder='passthrough')
        
        logger.info("Fitting OneHot and Ordinal Encoders...")
        self.encoder.fit(X)
        return self

    def transform(self, df_main: pd.DataFrame, df_diagnosa: pd.DataFrame, df_proc: pd.DataFrame) -> np.ndarray:
        """Transforms raw inputs into preprocessed encoded numpy arrays."""
        df_structured = self.process_data(df_main, df_diagnosa, df_proc, is_fit=False)
        X = df_structured.drop(columns=['id', 'id_peserta', 'label'], errors='ignore')
        return self.encoder.transform(X)

    def get_feature_names(self) -> list:
        """Gets feature names from the encoder pipeline."""
        if self.encoder is None:
            raise ValueError("Preprocessor has not been fitted yet.")
        return list(self.encoder.get_feature_names_out())
