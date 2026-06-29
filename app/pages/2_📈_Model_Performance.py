import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

# Set config
st.set_page_config(page_title="Model Performance - Fraud Detection", page_icon="📈", layout="wide")

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    .header-gradient {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0.3rem;
        font-family: 'Inter', sans-serif;
    }
    .subtitle { color: #94a3b8; font-size: 1.05rem; margin-bottom: 1.5rem; line-height: 1.6; }
    .kpi-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(30, 41, 59, 0.4) 100%);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    .kpi-value { font-size: 1.9rem; font-weight: 800; margin: 0.3rem 0; font-family: 'Inter', sans-serif; }
    .kpi-label { font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600; }
    .kpi-sub { font-size: 0.75rem; color: #64748b; margin-top: 0.2rem; }
    .insight-box {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.08) 0%, rgba(129, 140, 248, 0.08) 100%);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
        line-height: 1.7;
    }
    .section-header {
        font-size: 1.3rem; font-weight: 700; color: #e2e8f0; margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem; border-bottom: 2px solid rgba(56, 189, 248, 0.2);
    }
    .context-card {
        background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255,255,255,0.05);
        border-radius: 14px; padding: 1.3rem 1.5rem; margin-bottom: 1rem;
    }
    .warn-box {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.08) 0%, rgba(251, 146, 60, 0.08) 100%);
        border: 1px solid rgba(244, 63, 94, 0.15);
        border-radius: 12px; padding: 1.2rem 1.5rem; margin: 0.8rem 0; line-height: 1.7;
    }
    .ethics-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(56, 189, 248, 0.08) 100%);
        border: 1px solid rgba(16, 185, 129, 0.15);
        border-radius: 12px; padding: 1.2rem 1.5rem; margin: 0.8rem 0; line-height: 1.7;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='header-gradient'>📈 Model Validation Report</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Laporan evaluasi model deteksi fraud yang transparan dan akuntabel, yang mencakup metrik performa, kurva evaluasi, analisis fitur, serta pertimbangan etika dan limitasi model.</p>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
meta_path = "models/model_metadata.json"
eval_path = "models/val_eval.joblib"
model_path = "models/final_model.joblib"

meta = None
eval_data = None
model = None

if os.path.exists(meta_path):
    with open(meta_path, "r") as f:
        meta = json.load(f)
if os.path.exists(eval_path):
    eval_data = joblib.load(eval_path)
if os.path.exists(model_path):
    model = joblib.load(model_path)

if meta is None:
    st.warning("⚠️ File metadata model belum tersedia. Harap jalankan proses training terlebih dahulu (`PYTHONPATH=. python3 src/models/train.py`).")
    st.stop()

metrics = meta.get("val_metrics_target_threshold", {})
metrics_default = meta.get("val_metrics_default_0.5", {})
test_metrics = meta.get("test_metrics_target_threshold", {})
threshold = meta.get("classification_threshold", 0.7868)

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏷️ Model Summary",
    "📊 Performance Metrics", 
    "📈 Learning & Evaluation",
    "🔑 Feature Importance",
    "⚖️ Transparency & Ethics"
])

# ══════════════════════════════════════════════
# TAB 1: MODEL SUMMARY
# ══════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>🤖 Model Card</div>", unsafe_allow_html=True)
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown(f"""<div class='context-card'>
        <h4 style='margin:0 0 0.8rem 0; color:#38bdf8;'>Algoritma yang Digunakan</h4>
        <p style='line-height:1.8; margin:0;'>
        <b>XGBoost Classifier</b> (Extreme Gradient Boosting)<br><br>
        XGBoost adalah algoritma <i>ensemble learning</i> berbasis pohon keputusan yang membangun 
        ratusan pohon kecil secara berurutan, di mana setiap pohon baru berfokus memperbaiki 
        kesalahan prediksi pohon sebelumnya. Algoritma ini dikenal sebagai salah satu model 
        terbaik untuk data tabular/terstruktur.
        </p>
        </div>""", unsafe_allow_html=True)
        
    with col_info2:
        st.markdown(f"""<div class='context-card'>
        <h4 style='margin:0 0 0.8rem 0; color:#818cf8;'>Mengapa XGBoost Dipilih?</h4>
        <p style='line-height:1.8; margin:0;'>
        <b>1.</b> Performa superior pada data tabular & kategorikal campuran<br>
        <b>2.</b> Tahan terhadap fitur yang tidak diskalakan (scale-invariant)<br>
        <b>3.</b> Menyediakan feature importance bawaan untuk interpretabilitas<br>
        <b>4.</b> Mendukung early stopping untuk mencegah overfitting<br>
        <b>5.</b> Sangat cepat untuk training & inference (2 menit vs 5 jam NN)
        </p>
        </div>""", unsafe_allow_html=True)

    # Strengths & Limitations
    col_sl1, col_sl2 = st.columns(2)
    with col_sl1:
        st.markdown("""<div class='insight-box'>
        <h4 style='margin:0 0 0.5rem 0; color:#10b981;'>✅ Kekuatan Model</h4>
        • Presisi tinggi (~90%), sehingga meminimalkan salah tuduh pada klaim aman<br>
        • Specificity sangat tinggi (>99.9%), yang berarti klaim aman hampir selalu diprediksi benar<br>
        • Training cepat (~2 menit), yang memungkinkan iterasi dan retraining berkala<br>
        • Interpretable, karena feature importance dan penjelasan berbasis aturan (rule-based) telah disediakan<br>
        • Robust terhadap outlier dan missing values setelah preprocessing
        </div>""", unsafe_allow_html=True)
    with col_sl2:
        st.markdown("""<div class='warn-box'>
        <h4 style='margin:0 0 0.5rem 0; color:#f43f5e;'>⚠️ Keterbatasan Model</h4>
        • Recall moderat (~40%), yang berarti sekitar 60% klaim fraud masih lolos dari deteksi<br>
        • Dilatih pada data historis 2022, sehingga belum tentu akurat untuk modus fraud baru<br>
        • Tidak menangkap urutan temporal (time-series) antar klaim pasien<br>
        • Bergantung pada kualitas kode INA-CBG dan diagnosis. Jika data input salah, prediksi tidak akurat<br>
        • Perlu retraining berkala saat pola klaim berubah (concept drift)
        </div>""", unsafe_allow_html=True)

    # Hyperparameters & Pipeline
    st.markdown("<div class='section-header'>⚙️ Konfigurasi Training Pipeline</div>", unsafe_allow_html=True)
    
    col_hp1, col_hp2 = st.columns(2)
    with col_hp1:
        xgb_params = meta.get("xgb_params", {})
        hp_data = [
            {"Parameter": "n_estimators", "Nilai": str(xgb_params.get("n_estimators", "1000")), "Penjelasan": "Jumlah maksimum pohon keputusan"},
            {"Parameter": "max_depth", "Nilai": str(xgb_params.get("max_depth", "6")), "Penjelasan": "Kedalaman maksimum setiap pohon"},
            {"Parameter": "learning_rate", "Nilai": str(xgb_params.get("learning_rate", "0.05")), "Penjelasan": "Laju pembelajaran (step size)"},
            {"Parameter": "subsample", "Nilai": str(xgb_params.get("subsample", "0.8")), "Penjelasan": "Fraksi data untuk setiap pohon"},
            {"Parameter": "scale_pos_weight", "Nilai": str(xgb_params.get("scale_pos_weight", "1.13")), "Penjelasan": "Bobot kelas positif (fraud)"},
            {"Parameter": "early_stopping_rounds", "Nilai": "100", "Penjelasan": "Berhenti jika tidak ada perbaikan"},
            {"Parameter": "eval_metric", "Nilai": "auc, logloss", "Penjelasan": "Metrik evaluasi pada validation set"},
        ]
        st.dataframe(pd.DataFrame(hp_data), use_container_width=True, hide_index=True)
    
    with col_hp2:
        best_iter = meta.get("best_iteration", "N/A")
        training_time = meta.get("training_time_seconds", None)
        st.markdown(f"""<div class='context-card'>
        <h4 style='margin:0 0 0.8rem 0; color:#c084fc;'>📋 Detail Training</h4>
        <p style='line-height:2; margin:0;'>
        <b>Data Split:</b> Train 60% / Validation 20% / Test 20%<br>
        <b>Stratifikasi:</b> Ya (mempertahankan rasio fraud di setiap subset)<br>
        <b>Balancing:</b> SMOTE (oversampling 19.7%) + RUS (undersampling 88.5%)<br>
        <b>Random Seed:</b> 42 (untuk reproduksibilitas penuh)<br>
        <b>Best Iteration:</b> {best_iter}<br>
        <b>Jumlah Fitur (setelah encoding):</b> {meta.get('num_features', 'N/A')}<br>
        <b>Waktu Training:</b> {f'{float(training_time):.1f} detik' if training_time else 'N/A'}<br>
        <b>Classification Threshold:</b> {threshold:.7f}
        </p>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2: PERFORMANCE METRICS
# ══════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>📊 Metrik Evaluasi (Validation Set dengan Tuned Threshold)</div>", unsafe_allow_html=True)
    
    # Metric Cards
    precision = metrics.get('precision', 0)
    recall = metrics.get('recall', 0)
    f1 = metrics.get('f1_score', 0)
    roc_auc = metrics.get('roc_auc', 0)
    pr_auc = metrics.get('pr_auc', 0)
    specificity = metrics.get('specificity', 0)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Precision (Fraud)</div>
            <div class='kpi-value' style='color: #10b981;'>{precision*100:.1f}%</div>
            <div class='kpi-sub'>Dari klaim yang ditandai fraud, berapa yang benar fraud</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Recall / Sensitivity (Fraud)</div>
            <div class='kpi-value' style='color: #f59e0b;'>{recall*100:.1f}%</div>
            <div class='kpi-sub'>Dari seluruh fraud asli, berapa yang berhasil terdeteksi</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>F1-Score</div>
            <div class='kpi-value' style='color: #38bdf8;'>{f1*100:.1f}%</div>
            <div class='kpi-sub'>Rata-rata harmonis Precision & Recall</div>
        </div>""", unsafe_allow_html=True)
    
    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>ROC-AUC</div>
            <div class='kpi-value' style='color: #818cf8;'>{roc_auc:.4f}</div>
            <div class='kpi-sub'>Kemampuan membedakan fraud vs bukan fraud</div>
        </div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>PR-AUC (Avg. Precision)</div>
            <div class='kpi-value' style='color: #c084fc;'>{pr_auc:.4f}</div>
            <div class='kpi-sub'>Metrik utama untuk dataset imbalanced</div>
        </div>""", unsafe_allow_html=True)
    with col6:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Specificity</div>
            <div class='kpi-value' style='color: #22d3ee;'>{specificity*100:.2f}%</div>
            <div class='kpi-sub'>Klaim aman yang diprediksi benar aman</div>
        </div>""", unsafe_allow_html=True)
    
    # Metric Explanations
    st.markdown("<div class='section-header'>📖 Penjelasan Setiap Metrik</div>", unsafe_allow_html=True)
    
    st.markdown("""<div class='insight-box'>
    <b>Mengapa Accuracy Saja Tidak Cukup?</b><br><br>
    Dengan rasio fraud hanya ~1.4%, sebuah model yang <i>selalu</i> memprediksi "Bukan Fraud" akan memiliki 
    akurasi <b>98.6%</b>, yang terlihat sangat bagus, namun <b>tidak mendeteksi satu pun klaim fraud</b>. 
    Oleh karena itu, dalam deteksi fraud, kita fokus pada metrik yang lebih relevan:<br><br>
    • <b>Precision</b>, untuk mengukur seberapa terpercaya alarm fraud model? (90% = dari 10 alarm, 9 benar fraud)<br>
    • <b>Recall</b>, untuk mengukur seberapa lengkap model menangkap fraud? (40% = 4 dari 10 fraud terdeteksi)<br>
    • <b>F1-Score</b>, yang merupakan keseimbangan antara Precision dan Recall<br>
    • <b>PR-AUC</b>, yaitu metrik terbaik untuk dataset imbalanced (mengevaluasi trade-off precision-recall secara menyeluruh)<br>
    • <b>ROC-AUC</b>, untuk melihat kemampuan umum model membedakan kedua kelas<br>
    • <b>Specificity</b>, untuk melihat seberapa aman klaim non-fraud dari salah tuduh
    </div>""", unsafe_allow_html=True)

    # Confusion Matrix
    st.markdown("<div class='section-header'>🔢 Confusion Matrix</div>", unsafe_allow_html=True)
    
    cm_data = metrics.get('confusion_matrix', {})
    tn = cm_data.get('tn', 0)
    fp = cm_data.get('fp', 0)
    fn = cm_data.get('fn', 0)
    tp = cm_data.get('tp', 0)
    total = tn + fp + fn + tp
    
    col_cm1, col_cm2 = st.columns([1, 1.2])
    with col_cm1:
        import seaborn as sns
        sns.set_theme(style='whitegrid')
        plt.style.use('default')
        cm = np.array([[tn, fp], [fn, tp]])
        
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('white')
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', linewidths=0.5,
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'],
            ax=ax, annot_kws={'size': 14}
        )
        
        ax.set_title(
            f'Confusion Matrix (threshold={threshold:.4f})\n'
            f'TP={tp}  FP={fp}  FN={fn}  TN={tn}',
            fontweight='bold', fontsize=12, color='black'
        )
        ax.set_ylabel('Aktual', fontsize=11, color='black')
        ax.set_xlabel('Prediksi', fontsize=11, color='black')
        ax.tick_params(colors='black')
        
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    with col_cm2:
        st.markdown(f"""<div class='context-card' style='min-height: 330px;'>
        <h4 style='margin:0 0 0.8rem 0; color:#38bdf8;'>📋 Interpretasi Bisnis</h4>
        <p style='line-height:1.75; margin:0;'>
        <b style='color:#10b981;'>True Negative ({tn:,}):</b> Klaim normal yang diprediksi benar normal ✅<br>
        <b style='color:#f43f5e;'>False Positive ({fp:,}):</b> Klaim normal yang salah ditandai sebagai fraud ⚠️<br>
        → <i>Dampak: operasional rumah sakit sedikit terganggu karena perlunya audit berkas tambahan</i><br><br>
        <b style='color:#f59e0b;'>False Negative ({fn:,}):</b> Klaim fraud yang tidak terdeteksi oleh model ⚠️<br>
        → <i>Dampak: kebocoran anggaran bagi dana jaminan kesehatan nasional</i><br><br>
        <b style='color:#38bdf8;'>True Positive ({tp:,}):</b> Klaim fraud yang berhasil ditangkap oleh sistem ✅<br><br>
        <b>Penyetelan Ambang Batas:</b> Kami menaikkan ambang batas keputusan ke <code>{threshold:.4f}</code> untuk meminimalkan False Positive demi menjaga hubungan kemitraan dengan rumah sakit.
        </p>
        </div>""", unsafe_allow_html=True)

    # Business Impact Simulation
    st.markdown("<div class='section-header'>💼 Simulasi Dampak Bisnis (Estimasi Finansial)</div>", unsafe_allow_html=True)
    
    avg_fraud_biaya = 4000000.0  # Rp 4 juta per klaim fraud
    BIAYA_REVIEW_MANUAL = 500000.0  # Rp 500 ribu per review
    RECOVERY_RATE = 0.70  # 70% fraud berhasil dipulihkan
    
    nilai_fraud_dicegah = tp * avg_fraud_biaya * RECOVERY_RATE
    nilai_fraud_lolos = fn * avg_fraud_biaya
    biaya_false_alarm = fp * BIAYA_REVIEW_MANUAL
    net_benefit = nilai_fraud_dicegah - biaya_false_alarm
    
    klaim_untuk_audit = tp + fp
    efisiensi_audit = (1 - (klaim_untuk_audit / total)) * 100
    
    col_bi1, col_bi2 = st.columns(2)
    with col_bi1:
        st.markdown(f"""<div class='context-card' style='border-left: 5px solid #10b981; min-height: 230px;'>
            <h4 style='margin:0 0 0.8rem 0; color:#10b981;'>💰 Analisis Penghematan Finansial</h4>
            <p style='line-height:1.8; margin:0;'>
            • <b>Estimasi Biaya Rata-rata Klaim Fraud:</b> Rp {avg_fraud_biaya:,.0f}<br>
            • <b>Nilai Fraud Berhasil Dicegah (70% Recovery):</b> Rp {nilai_fraud_dicegah:,.0f}<br>
            • <b>Kerugian Kebocoran Dana (Fraud Lolos):</b> Rp {nilai_fraud_lolos:,.0f}<br>
            • <b>Pengeluaran Proses Review False Alarm:</b> Rp {biaya_false_alarm:,.0f}<br>
            • <b>ESTIMASI NET BENEFIT:</b> <b style='color:#10b981; font-size:1.15rem;'>Rp {net_benefit:,.0f}</b>
            </p>
        </div>""", unsafe_allow_html=True)
    with col_bi2:
        st.markdown(f"""<div class='context-card' style='border-left: 5px solid #38bdf8; min-height: 230px;'>
            <h4 style='margin:0 0 0.8rem 0; color:#38bdf8;'>⚙️ Efisiensi Operasional</h4>
            <p style='line-height:1.8; margin:0;'>
            • <b>Total Klaim Validasi:</b> {total:,} klaim<br>
            • <b>Klaim Dikirim ke Auditor Manual:</b> {klaim_untuk_audit:,} klaim ({klaim_untuk_audit/total*100:.2f}%)<br>
            • <b>Beban Audit Manual Berhasil Dipangkas:</b> <b style='color:#38bdf8; font-size:1.15rem;'>{efisiensi_audit:.1f}%</b><br>
            • <b>Sensitivitas Deteksi Kasus Fraud Aktual:</b> {tp/(tp+fn)*100:.1f}% kasus fraud berhasil ditangkap
            </p>
        </div>""", unsafe_allow_html=True)

    # Test Set Performance
    if test_metrics:
        st.markdown("<div class='section-header'>🧪 Performa pada Data Uji (Test Set)</div>", unsafe_allow_html=True)
        
        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1:
            st.markdown(f"""<div class='kpi-card'>
                <div class='kpi-label'>Test Precision</div>
                <div class='kpi-value' style='color:#10b981; font-size:1.5rem;'>{test_metrics.get('precision',0)*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with tc2:
            st.markdown(f"""<div class='kpi-card'>
                <div class='kpi-label'>Test Recall</div>
                <div class='kpi-value' style='color:#f59e0b; font-size:1.5rem;'>{test_metrics.get('recall',0)*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with tc3:
            st.markdown(f"""<div class='kpi-card'>
                <div class='kpi-label'>Test F1-Score</div>
                <div class='kpi-value' style='color:#38bdf8; font-size:1.5rem;'>{test_metrics.get('f1_score',0)*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with tc4:
            st.markdown(f"""<div class='kpi-card'>
                <div class='kpi-label'>Test ROC-AUC</div>
                <div class='kpi-value' style='color:#818cf8; font-size:1.5rem;'>{test_metrics.get('roc_auc',0):.4f}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 3: LEARNING & EVALUATION
# ══════════════════════════════════════════════
with tab3:
    import seaborn as sns
    sns.set_theme(style='whitegrid')
    plt.style.use('default')
    
    # ── 1. Learning Curves ──
    st.markdown("<div class='section-header'>📈 Kurva Pembelajaran Pelatihan (Learning Curves)</div>", unsafe_allow_html=True)
    st.markdown("Kurva ini menunjukkan perkembangan performa model XGBoost selama iterasi pelatihan menggunakan early stopping.")
    
    best_iter = meta.get("best_iteration", 0)
    
    # Check if we have pre-calculated train/val curves in eval_data
    if eval_data is not None and "train_loss" in eval_data:
        train_loss = eval_data["train_loss"]
        val_loss = eval_data["val_loss"]
        train_auc = eval_data["train_auc"]
        val_auc = eval_data["val_auc"]
        train_prec = eval_data["train_prec"]
        train_rec = eval_data["train_rec"]
        val_prec = eval_data["val_prec"]
        val_rec = eval_data["val_rec"]
        curve_iters = eval_data["curve_iterations"]
        
        fig_lc, axes_lc = plt.subplots(2, 2, figsize=(14, 10))
        fig_lc.patch.set_facecolor('white')
        
        # 1. Binary Cross-Entropy Loss
        axes_lc[0, 0].plot(train_loss, color='#2980b9', lw=2, label='Train')
        axes_lc[0, 0].plot(val_loss, color='#e74c3c', lw=2, linestyle='--', label='Validation')
        axes_lc[0, 0].axvline(best_iter, color='gray', linestyle=':', lw=1.2, label=f'Best epoch ({best_iter})')
        axes_lc[0, 0].set_title('Binary Cross-Entropy Loss', fontweight='bold')
        axes_lc[0, 0].set_xlabel('Epoch')
        axes_lc[0, 0].set_ylabel('Loss')
        axes_lc[0, 0].legend()
        axes_lc[0, 0].grid(True, alpha=0.3)
        axes_lc[0, 0].tick_params(colors='black')
        
        # 2. ROC-AUC Score
        axes_lc[0, 1].plot(train_auc, color='#2980b9', lw=2, label='Train')
        axes_lc[0, 1].plot(val_auc, color='#e74c3c', lw=2, linestyle='--', label='Validation')
        axes_lc[0, 1].axvline(best_iter, color='gray', linestyle=':', lw=1.2, label=f'Best epoch ({best_iter})')
        axes_lc[0, 1].set_title('ROC-AUC Score', fontweight='bold')
        axes_lc[0, 1].set_xlabel('Epoch')
        axes_lc[0, 1].set_ylabel('AUC')
        axes_lc[0, 1].legend()
        axes_lc[0, 1].grid(True, alpha=0.3)
        axes_lc[0, 1].tick_params(colors='black')
        
        # 3. Precision
        axes_lc[1, 0].plot(curve_iters, train_prec, color='#2980b9', lw=2, label='Train')
        axes_lc[1, 0].plot(curve_iters, val_prec, color='#e74c3c', lw=2, linestyle='--', label='Validation')
        axes_lc[1, 0].axvline(best_iter, color='gray', linestyle=':', lw=1.2, label=f'Best epoch ({best_iter})')
        axes_lc[1, 0].set_title('Precision', fontweight='bold')
        axes_lc[1, 0].set_xlabel('Epoch')
        axes_lc[1, 0].set_ylabel('Precision')
        axes_lc[1, 0].legend()
        axes_lc[1, 0].grid(True, alpha=0.3)
        axes_lc[1, 0].tick_params(colors='black')
        
        # 4. Recall
        axes_lc[1, 1].plot(curve_iters, train_rec, color='#2980b9', lw=2, label='Train')
        axes_lc[1, 1].plot(curve_iters, val_rec, color='#e74c3c', lw=2, linestyle='--', label='Validation')
        axes_lc[1, 1].axvline(best_iter, color='gray', linestyle=':', lw=1.2, label=f'Best epoch ({best_iter})')
        axes_lc[1, 1].set_title('Recall', fontweight='bold')
        axes_lc[1, 1].set_xlabel('Epoch')
        axes_lc[1, 1].set_ylabel('Recall')
        axes_lc[1, 1].legend()
        axes_lc[1, 1].grid(True, alpha=0.3)
        axes_lc[1, 1].tick_params(colors='black')
        
        plt.suptitle('Learning Curves — BPJS Fraud Detection XGBoost', fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        st.pyplot(fig_lc)
        plt.close(fig_lc)
        
        st.markdown(f"""<div class='insight-box' style='background-color: #f8fafc; border-color: #cbd5e1; color: #1e293b;'>
        <b>💡 Interpretasi Learning Curve:</b> Model berhenti berlatih secara otomatis pada iterasi ke-<b>{best_iter}</b> 
        karena skor logloss pada data validasi telah stabil dan tidak lagi mengalami peningkatan secara signifikan. 
        Mekanisme early stopping ini mencegah model dari kondisi <i>overfitting</i> pada data latih.
        </div>""", unsafe_allow_html=True)
        
    elif model is not None and hasattr(model, 'evals_result'):
        evals = model.evals_result()
        if 'validation_0' in evals:
            val_loss = evals['validation_0'].get('logloss', [])
            val_auc = evals['validation_0'].get('auc', [])
            
            fig_lc, axes_lc = plt.subplots(1, 2, figsize=(14, 5.5))
            fig_lc.patch.set_facecolor('white')
            
            # Logloss
            axes_lc[0].plot(val_loss, color='#2980b9', linewidth=2, label='Validation Loss')
            axes_lc[0].axvline(best_iter, color='gray', linestyle=':', lw=1.2,
                              label=f'Best iteration ({best_iter})')
            axes_lc[0].set_xlabel('Epoch/Iterasi')
            axes_lc[0].set_ylabel('Loss')
            axes_lc[0].set_title('Binary Cross-Entropy Loss', fontweight='bold')
            axes_lc[0].legend()
            axes_lc[0].grid(True, alpha=0.3)
            
            # AUC
            axes_lc[1].plot(val_auc, color='#2980b9', linewidth=2, label='Validation AUC')
            axes_lc[1].axvline(best_iter, color='gray', linestyle=':', lw=1.2,
                              label=f'Best iteration ({best_iter})')
            axes_lc[1].set_xlabel('Epoch/Iterasi')
            axes_lc[1].set_ylabel('AUC')
            axes_lc[1].set_title('ROC-AUC Score', fontweight='bold')
            axes_lc[1].legend()
            axes_lc[1].grid(True, alpha=0.3)
            
            plt.suptitle('Learning Curves — BPJS Fraud Detection XGBoost', fontsize=14, fontweight='bold', y=1.01)
            plt.tight_layout()
            st.pyplot(fig_lc)
            plt.close(fig_lc)
            
            st.markdown(f"""<div class='insight-box' style='background-color: #f8fafc; border-color: #cbd5e1; color: #1e293b;'>
            <b>💡 Interpretasi Learning Curve:</b> Model berhenti berlatih secara otomatis pada iterasi ke-<b>{best_iter}</b> 
            karena skor logloss pada data validasi telah stabil dan tidak lagi mengalami peningkatan secara signifikan. 
            Mekanisme early stopping ini sangat penting untuk menghindari *overfitting* pada data latih.
            </div>""", unsafe_allow_html=True)
    else:
        st.info("Kurva pembelajaran tidak tersedia.")

    # ── 2. Comprehensive 4-Panel Evaluation Plot ──
    st.markdown("<div class='section-header'>🔬 Evaluasi Komprehensif (4 Panel)</div>", unsafe_allow_html=True)
    st.markdown("Visualisasi komprehensif performa model XGBoost pada data validasi menggunakan 4 panel evaluasi utama.")
    
    if eval_data is not None:
        y_val = eval_data["y_val"]
        y_val_proba = eval_data["y_val_proba"]
        y_pred = (y_val_proba >= threshold).astype(int)
        
        cm = confusion_matrix(y_val, y_pred)
        tn_c, fp_c, fn_c, tp_c = cm.ravel()
        
        fpr_c, tpr_c, _ = roc_curve(y_val, y_val_proba)
        prec_c, rec_c, _ = precision_recall_curve(y_val, y_val_proba)
        baseline_c = y_val.mean()
        
        fig_4p, axes_4p = plt.subplots(2, 2, figsize=(14, 11))
        fig_4p.patch.set_facecolor('white')
        
        # ── Panel 1: Confusion Matrix ──
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', linewidths=0.5,
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'],
            ax=axes_4p[0, 0], annot_kws={'size': 14}
        )
        axes_4p[0, 0].set_title(
            f'Confusion Matrix (threshold={threshold:.4f})\n'
            f'TP={tp_c}  FP={fp_c}  FN={fn_c}  TN={tn_c}',
            fontweight='bold', fontsize=12
        )
        axes_4p[0, 0].set_ylabel('Aktual', fontsize=11, color='black')
        axes_4p[0, 0].set_xlabel('Prediksi', fontsize=11, color='black')
        axes_4p[0, 0].tick_params(colors='black')
        
        # ── Panel 2: ROC Curve ──
        axes_4p[0, 1].plot(fpr_c, tpr_c, color='#e74c3c', lw=2.5,
                        label=f'ROC-AUC = {roc_auc:.4f}')
        axes_4p[0, 1].plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
        axes_4p[0, 1].fill_between(fpr_c, tpr_c, alpha=0.08, color='#e74c3c')
        axes_4p[0, 1].set_title('ROC Curve', fontweight='bold', fontsize=12)
        axes_4p[0, 1].set_xlabel('False Positive Rate', fontsize=11, color='black')
        axes_4p[0, 1].set_ylabel('True Positive Rate (Recall)', fontsize=11, color='black')
        axes_4p[0, 1].legend(loc='lower right')
        axes_4p[0, 1].grid(True, alpha=0.3)
        axes_4p[0, 1].tick_params(colors='black')
        
        # ── Panel 3: Precision-Recall Curve ──
        axes_4p[1, 0].plot(rec_c, prec_c, color='#2980b9', lw=2.5,
                        label=f'PR-AUC = {pr_auc:.4f}')
        axes_4p[1, 0].axhline(baseline_c, color='gray', linestyle='--', lw=1.2,
                           label=f'Baseline ({baseline_c:.3f})')
        axes_4p[1, 0].fill_between(rec_c, prec_c, alpha=0.08, color='#2980b9')
        axes_4p[1, 0].set_title('Precision-Recall Curve', fontweight='bold', fontsize=12)
        axes_4p[1, 0].set_xlabel('Recall', fontsize=11, color='black')
        axes_4p[1, 0].set_ylabel('Precision', fontsize=11, color='black')
        axes_4p[1, 0].legend(loc='upper right')
        axes_4p[1, 0].grid(True, alpha=0.3)
        axes_4p[1, 0].tick_params(colors='black')
        
        # ── Panel 4: Distribusi Skor Fraud ──
        axes_4p[1, 1].hist(y_val_proba[y_val == 0], bins=60, alpha=0.65,
                        color='#27ae60', label='Normal', density=True)
        axes_4p[1, 1].hist(y_val_proba[y_val == 1], bins=60, alpha=0.65,
                        color='#e74c3c', label='Fraud',  density=True)
        axes_4p[1, 1].axvline(threshold, color='black', lw=2, linestyle='--',
                           label=f'Threshold = {threshold:.4f}')
        axes_4p[1, 1].set_title('Distribusi Skor Probabilitas Fraud', fontweight='bold', fontsize=12)
        axes_4p[1, 1].set_xlabel('P(Fraud)', fontsize=11, color='black')
        axes_4p[1, 1].set_ylabel('Densitas', fontsize=11, color='black')
        axes_4p[1, 1].legend()
        axes_4p[1, 1].grid(True, alpha=0.3)
        axes_4p[1, 1].tick_params(colors='black')
        
        plt.suptitle('Evaluasi Model XGBoost — BPJS Fraud Detection', fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        st.pyplot(fig_4p)
        plt.close(fig_4p)
    else:
        st.info("Visualisasi 4 panel tidak tersedia.")

# ══════════════════════════════════════════════
# TAB 4: FEATURE IMPORTANCE
# ══════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>🔑 Fitur Paling Berpengaruh</div>", unsafe_allow_html=True)
    st.markdown("Visualisasi fitur yang paling banyak digunakan oleh model XGBoost dalam membuat keputusan prediksi. Semakin tinggi skor importance, semakin berpengaruh fitur tersebut.")
    
    if model is not None:
        try:
            importance = model.feature_importances_
            feature_names = [f'feature_{i}' for i in range(len(importance))]
            
            # Load preprocessor to get actual feature names
            prep_path = "models/preprocessor_pipeline.joblib"
            if os.path.exists(prep_path):
                try:
                    preprocessor = joblib.load(prep_path)
                    feature_names = preprocessor.get_feature_names()
                except Exception as ex_prep:
                    st.warning(f"Gagal memuat preprocessor untuk mapping nama fitur: {ex_prep}")
            
            # Helper to map technical codes to clean human-readable text
            def clean_feat_name(name):
                name = name.replace('onehot1__', '').replace('onehot2__', '').replace('remainder__', '').replace('ordinal_enc__', '')
                mapping = {
                    'biaya': 'Biaya Klaim (Rupiah)',
                    'lama_rawat': 'Durasi Rawat Inap (Hari)',
                    'diagfkrtl_sekunder_counts': 'Jumlah Diagnosis Sekunder',
                    'proc_count': 'Jumlah Prosedur (Tindakan)',
                    'usia': 'Usia Pasien',
                    'kelasrawat': 'Kelas Kamar Rawat',
                    'jenkel': 'Jenis Kelamin Pasien',
                    'pisat': 'Segmen Kepesertaan',
                    'jenispel': 'Jenis Pelayanan',
                    'jenispulang': 'Status Keluar Pasien',
                    'typefaskes': 'Tipe/Kelas Faskes'
                }
                for key, val in mapping.items():
                    if name == key:
                        return val
                if 'cbg_CMG_' in name:
                    return f'Group CBG (CMG): {name.split("cbg_CMG_")[-1]}'
                if 'cbg_tipekasus_' in name:
                    return f'Tipe Kasus CBG: {name.split("cbg_tipekasus_")[-1]}'
                if 'cbg_spesifikkasus_' in name:
                    return f'Spesifikasi Kasus CBG: {name.split("cbg_spesifikkasus_")[-1]}'
                if 'cbg_severity_' in name:
                    return f'Tingkat Keparahan CBG: {name.split("cbg_severity_")[-1]}'
                if 'diagfktp_icd10_' in name:
                    return f'Diagnosis FKTP (Bab ICD-10): {name.split("diagfktp_icd10_")[-1]}'
                if 'diagfkrtl_icd10_' in name:
                    return f'Diagnosis FKRTL (Bab ICD-10): {name.split("diagfkrtl_icd10_")[-1]}'
                if 'kdsa_CMG_' in name:
                    return f'KDS Prosedur Akut (CMG): {name.split("kdsa_CMG_")[-1]}'
                if 'kdsa_tipekasus_' in name:
                    return f'KDS Tipe Kasus Akut: {name.split("kdsa_tipekasus_")[-1]}'
                if 'typefaskes_' in name:
                    return f'Tipe Faskes: {name.split("typefaskes_")[-1]}'
                return name
            
            feat_imp = pd.DataFrame({
                'Feature': feature_names, 
                'Importance': importance
            })
            feat_imp['CleanName'] = feat_imp['Feature'].apply(clean_feat_name)
            feat_imp = feat_imp.sort_values('Importance', ascending=False).head(20)
            
            # Setup plot
            import seaborn as sns
            sns.set_theme(style='whitegrid')
            plt.style.use('default')
            
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.patch.set_facecolor('white')
            
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(feat_imp)))
            ax.barh(feat_imp['CleanName'][::-1], feat_imp['Importance'][::-1], color=colors, height=0.6, edgecolor='none')
            ax.set_xlabel('Feature Importance Score', color='black', fontsize=11)
            ax.set_title('Top 20 Most Important Features', color='black', fontsize=13, fontweight='bold', pad=15)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(colors='black')
            
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            # Feature explanations
            feature_explanations = {
                'biaya': 'Total biaya klaim yang ditagihkan, di mana biaya abnormal tinggi/rendah dapat mengindikasikan upcoding atau phantom billing',
                'usia': 'Usia pasien, karena pola fraud sering terkonsentrasi pada kelompok usia tertentu',
                'kelasrawat': 'Kelas ruang rawat inap (1=VIP, 2=Standar, 3=Ekonomi)',
                'lama_rawat': 'Durasi rawat inap (hari), sebab durasi sangat panjang atau sangat pendek bisa menjadi indikator fraud',
                'diagfkrtl_sekunder_counts': 'Jumlah diagnosis sekunder, di mana banyak diagnosis bisa mengindikasikan upcoding',
                'proc_count': 'Jumlah prosedur/tindakan medis yang dilaporkan',
                'jenispel': 'Jenis pelayanan perawatan yang dijalani pasien (Rawat Inap atau Rawat Jalan)'
            }
            
            st.markdown("<div class='section-header'>📖 Penjelasan Fitur Utama</div>", unsafe_allow_html=True)
            explanations_html = ""
            for _, row in feat_imp.head(10).iterrows():
                feat = row['Feature']
                clean_n = row['CleanName']
                feat_key = feat.replace('onehot1__', '').replace('onehot2__', '').replace('remainder__', '').replace('ordinal_enc__', '').lower().strip()
                explanation = feature_explanations.get(feat_key, f'Kategori medis spesifik untuk {clean_n} (menunjukkan pengaruh signifikan diagnosis/prosedur ini terhadap klasifikasi fraud)')
                explanations_html += f"• <b>{clean_n}</b>: {explanation}<br>"
            
            st.markdown(f"""<div class='insight-box' style='background-color: #f8fafc; border-color: #cbd5e1; color: #1e293b;'>{explanations_html}</div>""", unsafe_allow_html=True)
            
        except Exception as e:
            st.warning(f"Tidak dapat mengekstrak feature importance: {e}")
    else:
        st.info("Model binary belum dimuat. Harap pastikan file `models/final_model.joblib` tersedia.")

# ══════════════════════════════════════════════
# TAB 5: TRANSPARENCY & ETHICS
# ══════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-header'>🔓 Transparansi Model</div>", unsafe_allow_html=True)
    st.markdown("Informasi lengkap tentang bagaimana model ini dibangun, untuk memastikan akuntabilitas dan reproduksibilitas.")
    
    transparency_data = [
        {"Aspek": "Sumber Data", "Detail": "BPJS Kesehatan Healthkathon 2022, yaitu data klaim rumah sakit nasional"},
        {"Aspek": "Periode Data", "Detail": "Tahun 2022 (data historis)"},
        {"Aspek": "Ukuran Populasi Data", "Detail": "11.4 juta klaim (dataset penuh BPJS Healthkathon)"},
        {"Aspek": "Pembagian Data", "Detail": "60% Training / 20% Validation / 20% Testing (stratified)"},
        {"Aspek": "Penyeimbangan Kelas", "Detail": "SMOTE (oversampling 19.7%) + Random Under-Sampling (88.5%)"},
        {"Aspek": "Preprocessing", "Detail": "Imputasi modus/median, OneHotEncoder, OrdinalEncoder, ICD-10 Chapter Mapping"},
        {"Aspek": "Feature Engineering", "Detail": "CBG/KDS code splitting, durasi rawat inap, diagnosis sekunder count"},
        {"Aspek": "Algoritma", "Detail": "XGBoost Classifier dengan early stopping"},
        {"Aspek": "Threshold Tuning", "Detail": f"Threshold digeser dari 0.50 ke {threshold:.4f} untuk Precision ~90%"},
        {"Aspek": "Random Seed", "Detail": "42 (untuk reproduksibilitas penuh)"},
        {"Aspek": "Library Versi", "Detail": "scikit-learn, xgboost, imbalanced-learn, pandas"},
    ]
    st.dataframe(pd.DataFrame(transparency_data), use_container_width=True, hide_index=True)
    
    # Limitations
    st.markdown("<div class='section-header'>⚠️ Limitasi & Situasi yang Tidak Ditangani</div>", unsafe_allow_html=True)
    st.markdown("""<div class='warn-box'>
    <b>Model ini TIDAK boleh dipercaya sepenuhnya dalam situasi berikut:</b><br><br>
    <b>1. Modus Fraud Baru:</b> Model dilatih pada pola historis 2022. Jika muncul modus fraud baru 
    yang belum pernah ada di data training (misalnya skema kolusi baru), model tidak akan mendeteksinya.<br><br>
    <b>2. Klaim dari Wilayah/Faskes Baru:</b> Jika kode wilayah (DATI2) atau tipe fasilitas kesehatan 
    yang dimasukkan tidak pernah muncul di data training, model akan memberikan prediksi yang kurang akurat.<br><br>
    <b>3. Data Input Tidak Lengkap:</b> Jika banyak field dikosongkan atau diisi dengan nilai default, 
    kualitas prediksi akan menurun signifikan.<br><br>
    <b>4. Keputusan Hukum/Administratif:</b> Output model ini <b>bukan bukti hukum</b>. 
    Model hanya memberikan indikasi probabilistik yang harus diverifikasi oleh auditor manusia.<br><br>
    <b>5. Concept Drift:</b> Seiring waktu, distribusi data klaim akan berubah. Model perlu dilatih ulang 
    secara berkala (direkomendasikan: setiap kuartal) untuk menjaga akurasi.
    </div>""", unsafe_allow_html=True)
    
    # Ethics
    st.markdown("<div class='section-header'>🤝 Pertimbangan Etika & AI yang Bertanggung Jawab</div>", unsafe_allow_html=True)
    st.markdown("""<div class='ethics-box'>
    <b>1. Human-in-the-Loop:</b> Model ini dirancang sebagai <b>alat bantu keputusan</b>, bukan pengganti 
    auditor manusia. Setiap klaim yang ditandai fraud harus melalui proses verifikasi manual oleh tim audit 
    sebelum tindakan administratif diambil.<br><br>
    <b>2. Keadilan (Fairness):</b> Model tidak menggunakan fitur yang secara langsung diskriminatif 
    (ras, agama, suku). Namun, fitur seperti lokasi geografis (DATI2) dan tipe faskes berpotensi 
    menjadi proxy untuk variabel sensitif. Pemantauan bias per kelompok demografis direkomendasikan 
    dalam deployment produksi.<br><br>
    <b>3. Transparansi:</b> Seluruh proses training, parameter model, dan metrik evaluasi didokumentasikan 
    secara terbuka pada halaman ini untuk memungkinkan audit dan peer review.<br><br>
    <b>4. Dampak Salah Prediksi:</b> False Positive (klaim aman ditandai fraud) dapat mengganggu 
    cash flow rumah sakit dan menurunkan kepercayaan mitra. Oleh karena itu, kami memprioritaskan 
    <b>Precision tinggi (90%)</b> untuk meminimalkan dampak ini.<br><br>
    <b>5. Hak untuk Menjelaskan:</b> Setiap prediksi fraud dilengkapi dengan penjelasan faktor risiko 
    yang mendorong keputusan model, memastikan keputusan dapat diaudit dan dijelaskan kepada pihak terkait.
    </div>""", unsafe_allow_html=True)
