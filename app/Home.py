import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import os
import json
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Healthcare Insurance Fraud Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS injection
st.markdown("""
<style>
    /* Main layout adjustments */
    .main {
        background: linear-gradient(135deg, #0e1117 0%, #161a24 100%);
        color: #e0e6ed;
    }
    
    /* Premium Title Banner */
    .title-banner {
        padding: 2.5rem;
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .title-banner h1 {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        font-size: 3rem;
    }
    
    .title-banner p {
        font-size: 1.2rem;
        color: #94a3b8;
    }

    /* Glassmorphic Cards */
    .card {
        background: rgba(30, 41, 59, 0.35);
        backdrop-filter: blur(8px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.15);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    .card-title {
        font-weight: 700;
        font-size: 1.3rem;
        color: #38bdf8;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .card-body {
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Stats Panel */
    .stat-container {
        display: flex;
        justify-content: space-around;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .stat-box {
        flex: 1;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    
    .stat-num {
        font-size: 2.2rem;
        font-weight: 800;
        color: #818cf8;
        margin-bottom: 0.2rem;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# Helper to check if model metadata exists
def get_metadata():
    meta_path = "models/model_metadata.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return None

meta = get_metadata()

# Page content
st.markdown("""
<div class="title-banner">
    <h1>Hospital Insurance Fraud Detection</h1>
    <p>Production-Grade End-to-End Machine Learning System for BPJS Claims Audit</p>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([2, 1.2])

with col_left:
    st.markdown("""
    <div class="card">
        <div class="card-title">🏥 Project Context & Purpose</div>
        <div class="card-body">
            Kecurangan (Fraud) klaim asuransi kesehatan merupakan tantangan serius yang merugikan dana publik hingga triliunan rupiah per tahun. 
            Proyek ini merancang sistem <b>End-to-End Machine Learning</b> berbasis data klaim asuransi kesehatan nasional (BPJS Kesehatan) 
            untuk membantu auditor mendeteksi klaim mencurigakan sebelum persetujuan pembayaran.
            <br><br>
            Sistem ini menggunakan algoritma <b>XGBoost Classifier</b> yang dilatih di atas dataset berukuran besar dengan penanganan ketidakseimbangan kelas (Class Imbalance) menggunakan kombinasi <b>SMOTE</b> dan <b>Random Under-Sampling (RUS)</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <div class="card-title">⚙️ Core Architecture Features</div>
        <div class="card-body">
            <ul>
                <li><b>Data & Preprocessing Pipeline:</b> Otomatisasi pembersihan diagnosis, format teks penamaan, dan penanganan missing values secara robust.</li>
                <li><b>Domain Feature Engineering:</b> Ekstraksi grup INA-CBG (CMG, Tipe Kasus, Tingkat Keparahan) berdasarkan regulasi <i>Permenkes No. 27 Tahun 2014</i>, ekstraksi Chapter ICD-10, durasi rawat inap, dan agregasi data sekunder.</li>
                <li><b>Precision Optimization:</b> Tuning threshold klasifikasi khusus (Tuned Threshold = 0.7868) untuk mencapai <b>target presisi 90%</b> demi meminimalkan False Positives (klaim aman yang terblokir) yang mengganggu pelayanan rumah sakit.</li>
                <li><b>Production Readiness:</b> Kode modular terorganisir di bawah struktur <code>src/</code>, siap di-deploy ke cloud production (Streamlit Cloud).</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    # Stats box
    st.markdown("### 📊 System Metadata")
    
    if meta:
        sample_frac = meta.get("sample_fraction", 0.05)
        num_features = meta.get("num_features", 342)
        train_date = meta.get("training_date", "N/A")
        threshold = meta.get("classification_threshold", 0.7868109)
        test_metrics = meta.get("test_metrics_target_threshold", {})
        precision = test_metrics.get("precision", 0.900)
        recall = test_metrics.get("recall", 0.400)
        f1 = test_metrics.get("f1_score", 0.550)
        
        st.markdown(f"""
        <div class="stat-container">
            <div class="stat-box">
                <div class="stat-num">{precision*100:.1f}%</div>
                <div class="stat-label">Model Precision</div>
            </div>
            <div class="stat-box">
                <div class="stat-num">{recall*100:.1f}%</div>
                <div class="stat-label">Model Recall</div>
            </div>
        </div>
        
        <div class="card" style="padding: 1.2rem;">
            <div class="card-title" style="font-size: 1.1rem; color: #818cf8;">🏷️ Training Metadata</div>
            <div class="card-body" style="font-size: 0.85rem;">
                <b>Model Type:</b> XGBoost Classifier<br>
                <b>Decision Threshold:</b> {threshold:.4f}<br>
                <b>Features Count:</b> {num_features} encoded<br>
                <b>Data Fraction:</b> {sample_frac*100:.1f}% of total claims<br>
                <b>Last Trained:</b> {train_date}<br>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Default placeholder metrics
        st.markdown("""
        <div class="stat-container">
            <div class="stat-box">
                <div class="stat-num">90.0%</div>
                <div class="stat-label">Target Precision</div>
            </div>
            <div class="stat-box">
                <div class="stat-num">40.1%</div>
                <div class="stat-label">Target Recall</div>
            </div>
        </div>
        
        <div class="card" style="padding: 1.2rem;">
            <div class="card-title" style="font-size: 1.1rem; color: #f59e0b;">⚠️ Model Training Pending</div>
            <div class="card-body" style="font-size: 0.85rem;">
                Model training pipeline belum dijalankan atau file model belum disimpan di folder <code>models/</code>. 
                Sistem saat ini menampilkan metrik default berdasarkan eksperimen notebook.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
    <div class="card" style="padding: 1.2rem;">
        <div class="card-title" style="font-size: 1.1rem;">🧭 Quick Navigation</div>
        <div class="card-body" style="font-size: 0.9rem;">
            Gunakan panel navigasi sidebar untuk menjelajahi:<br>
            <ul>
                <li><b>Dataset Overview:</b> Analisis statistik data klaim awal.</li>
                <li><b>Model Performance:</b> Grafik evaluasi (ROC/PR) dan analisis threshold.</li>
                <li><b>Fraud Prediction:</b> Prediksi klaim asuransi kesehatan (manual/CSV upload).</li>
                <li><b>About Project:</b> Latar belakang teknis dan teori bisnis asuransi.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
st.sidebar.success("Navigasi Berhasil Dimuat.")
