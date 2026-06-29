import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.helper import load_config, get_project_root

# Set config
st.set_page_config(page_title="Dataset Overview - Fraud Detection", page_icon="📊", layout="wide")

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
    .subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }
    .kpi-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(30, 41, 59, 0.4) 100%);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .kpi-card:hover { transform: translateY(-2px); }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0.3rem 0;
        font-family: 'Inter', sans-serif;
    }
    .kpi-label {
        font-size: 0.78rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.2rem;
    }
    .insight-box {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.08) 0%, rgba(129, 140, 248, 0.08) 100%);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
        line-height: 1.7;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid rgba(56, 189, 248, 0.2);
    }
    .quality-good { color: #10b981; }
    .quality-warn { color: #f59e0b; }
    .quality-bad { color: #f43f5e; }
    .feature-dict-table {
        font-size: 0.85rem;
    }
    .context-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='header-gradient'>📊 Dataset Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Analisis komprehensif dataset klaim jaminan kesehatan nasional (JKN/BPJS Kesehatan) untuk memahami karakteristik data, kualitas data, dan pola distribusi sebelum melakukan pemodelan deteksi fraud.</p>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────
@st.cache_data(show_spinner="Memuat dataset klaim kesehatan...")
def load_sample_data():
    config = load_config()
    root = get_project_root()
    main_path = os.path.join(root, config["data"]["raw_main_path"])
    diag_path = os.path.join(root, config["data"]["raw_diagnosa_path"])
    proc_path = os.path.join(root, config["data"]["raw_procedure_path"])
    
    if not (os.path.exists(main_path) and os.path.exists(diag_path) and os.path.exists(proc_path)):
        return None, None, None, 0, 0, 0
    
    try:
        df_main_full = pd.read_parquet(main_path)
        df_diag_full = pd.read_parquet(diag_path)
        df_proc_full = pd.read_parquet(proc_path)
        
        total_main = len(df_main_full)
        total_diag = len(df_diag_full)
        total_proc = len(df_proc_full)
        
        df_main = df_main_full.sample(n=min(15000, total_main), random_state=42).reset_index(drop=True)
        sampled_ids = set(df_main["id"].unique())
        df_diag = df_diag_full[df_diag_full["id"].isin(sampled_ids)].reset_index(drop=True)
        df_proc = df_proc_full[df_proc_full["id"].isin(sampled_ids)].reset_index(drop=True)
        
        return df_main, df_diag, df_proc, total_main, total_diag, total_proc
    except Exception as e:
        st.error(f"Gagal memuat dataset: {e}")
        return None, None, None, 0, 0, 0

df_main, df_diag, df_proc, total_main, total_diag, total_proc = load_sample_data()

if df_main is None:
    st.warning("⚠️ Dataset Parquet mentah tidak ditemukan di `data/raw/`. Pastikan file telah ditempatkan di folder yang benar.")
    st.stop()

assert df_main is not None
assert df_diag is not None
assert df_proc is not None

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Executive Summary", 
    "🔬 Data Quality Report", 
    "📈 Distribution & Analysis",
    "📖 Data Dictionary & Preview"
])

# ══════════════════════════════════════════════
# TAB 1: EXECUTIVE SUMMARY
# ══════════════════════════════════════════════
with tab1:
    # KPI Cards Row 1
    fraud_ratio = df_main['label'].mean()
    non_fraud_ratio = 1 - fraud_ratio
    n_features = df_main.shape[1]
    missing_pct = (df_main.isnull().sum().sum() / (df_main.shape[0] * df_main.shape[1])) * 100
    completeness = 100 - missing_pct
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Total Populasi Klaim</div>
            <div class='kpi-value' style='color: #38bdf8;'>{total_main:,}</div>
            <div class='kpi-sub'>Sampel ditampilkan: {df_main.shape[0]:,}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Rasio Fraud</div>
            <div class='kpi-value' style='color: #f43f5e;'>{fraud_ratio*100:.2f}%</div>
            <div class='kpi-sub'>{int(df_main['label'].sum()):,} dari {df_main.shape[0]:,} sampel</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Jumlah Fitur</div>
            <div class='kpi-value' style='color: #818cf8;'>{n_features}</div>
            <div class='kpi-sub'>Kolom dalam tabel utama</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Median Biaya Klaim</div>
            <div class='kpi-value' style='color: #10b981;'>Rp {df_main["biaya"].median():,.0f}</div>
            <div class='kpi-sub'>Per transaksi klaim</div>
        </div>""", unsafe_allow_html=True)
    with col5:
        q_class = 'quality-good' if completeness > 95 else ('quality-warn' if completeness > 85 else 'quality-bad')
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Kelengkapan Data</div>
            <div class='kpi-value {q_class}'>{completeness:.1f}%</div>
            <div class='kpi-sub'>Data tidak kosong</div>
        </div>""", unsafe_allow_html=True)
    
    # Business Context
    st.markdown("<div class='section-header'>🏥 Konteks Bisnis</div>", unsafe_allow_html=True)
    st.markdown("""<div class='context-card'>
    <p style='line-height:1.8; margin:0;'>
    Dataset ini berasal dari <b>BPJS Kesehatan</b> (Badan Penyelenggara Jaminan Sosial Kesehatan), 
    program jaminan kesehatan nasional Indonesia yang mencakup lebih dari <b>200 juta peserta</b>. 
    Setiap baris data merepresentasikan satu transaksi klaim rumah sakit yang ditagihkan ke BPJS 
    menggunakan sistem pembayaran <b>INA-CBGs (Indonesia Case Based Groups)</b> sesuai 
    <i>Permenkes No. 27 Tahun 2014</i>.</p>
    <br>
    <p style='line-height:1.8; margin:0;'>
    <b>Fraud dalam konteks ini</b> merujuk pada klaim yang diduga mengandung kecurangan seperti:
    <b>upcoding</b> (memperbesar kode diagnosis/prosedur untuk tarif lebih tinggi), 
    <b>phantom billing</b> (menagih layanan yang tidak diberikan), atau 
    <b>unbundling</b> (memecah satu prosedur menjadi beberapa klaim terpisah).
    Kolom target (<code>label</code>) bernilai <b>1</b> jika klaim teridentifikasi fraud, dan <b>0</b> jika klaim aman.</p>
    </div>""", unsafe_allow_html=True)

    # Data Composition
    st.markdown("<div class='section-header'>📦 Komposisi Dataset</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Tabel Klaim Utama</div>
            <div class='kpi-value' style='color:#38bdf8; font-size:1.5rem;'>{total_main:,} baris</div>
            <div class='kpi-sub'>{n_features} kolom, yang berisi data pasien, biaya, rumah sakit, serta kode CBG</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Tabel Diagnosa</div>
            <div class='kpi-value' style='color:#818cf8; font-size:1.5rem;'>{total_diag:,} baris</div>
            <div class='kpi-sub'>Kode ICD-10 diagnosis utama & sekunder per klaim</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Tabel Prosedur</div>
            <div class='kpi-value' style='color:#c084fc; font-size:1.5rem;'>{total_proc:,} baris</div>
            <div class='kpi-sub'>Kode ICD-9-CM tindakan medis per klaim</div>
        </div>""", unsafe_allow_html=True)

    # Class Imbalance Insight
    st.markdown("<div class='section-header'>⚖️ Ketidakseimbangan Kelas (Class Imbalance)</div>", unsafe_allow_html=True)
    
    col_imb1, col_imb2 = st.columns([1.2, 1])
    with col_imb1:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(8, 3.5))
        counts = df_main['label'].value_counts().sort_index()
        bars = ax.barh(
            ['Bukan Fraud (0)', 'Fraud (1)'], 
            [counts.get(0, 0), counts.get(1, 0)],
            color=['#38bdf8', '#f43f5e'], 
            height=0.5,
            edgecolor='none'
        )
        for bar, val in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
            pct = val / len(df_main) * 100
            ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2, 
                    f'  {val:,}  ({pct:.1f}%)', va='center', color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel('Jumlah Klaim', color='#94a3b8', fontsize=10)
        ax.set_title('Distribusi Label Target', color='white', fontsize=13, fontweight='bold', pad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0, counts.get(0, 1) * 1.25)
        fig.patch.set_alpha(0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    with col_imb2:
        imbalance_ratio = counts.get(0, 1) / max(counts.get(1, 1), 1)
        st.markdown(f"""<div class='insight-box'>
        <b>⚠️ Mengapa ini penting?</b><br><br>
        Dataset ini memiliki rasio ketidakseimbangan <b>~{imbalance_ratio:.0f}:1</b> 
        (Non-Fraud vs Fraud). Artinya, dari setiap <b>~{imbalance_ratio:.0f} klaim</b>, 
        hanya <b>1 klaim</b> yang teridentifikasi sebagai fraud.<br><br>
        Jika model ML dilatih tanpa penanganan khusus, model akan belajar untuk <i>selalu</i> 
        memprediksi "Bukan Fraud" dan mencapai akurasi {non_fraud_ratio*100:.1f}%, tetapi <b>tidak 
        mendeteksi fraud sama sekali</b>.<br><br>
        Oleh karena itu, pipeline kami menerapkan <b>SMOTE + Random Under-Sampling</b> untuk 
        menyeimbangkan distribusi kelas sebelum pelatihan model.
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2: DATA QUALITY REPORT
# ══════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>🔍 Laporan Kualitas Data</div>", unsafe_allow_html=True)
    st.markdown("Analisis kualitas data dilakukan pada sampel acak untuk mengidentifikasi nilai yang hilang, duplikat, dan anomali yang dapat memengaruhi performa model.")
    
    # Missing Values
    st.markdown("<div class='section-header'>📊 Nilai yang Hilang (Missing Values)</div>", unsafe_allow_html=True)
    
    missing = df_main.isnull().sum()
    missing_pct_col = (missing / len(df_main) * 100)
    missing_df = pd.DataFrame({
        'Kolom': missing.index, 
        'Jumlah Missing': missing.values,
        'Persentase': missing_pct_col.values
    }).sort_values('Jumlah Missing', ascending=False)
    missing_df = missing_df[missing_df['Jumlah Missing'] > 0]
    
    if len(missing_df) > 0:
        col_miss1, col_miss2 = st.columns([1.5, 1])
        with col_miss1:
            fig, ax = plt.subplots(figsize=(9, max(3, len(missing_df) * 0.4)))
            colors = ['#f43f5e' if p > 10 else '#f59e0b' if p > 1 else '#10b981' for p in missing_df['Persentase']]
            ax.barh(missing_df['Kolom'], missing_df['Persentase'], color=colors, height=0.6)
            ax.set_xlabel('Persentase Missing (%)', color='#94a3b8')
            ax.set_title('Proporsi Missing Value per Kolom', color='white', fontweight='bold')
            ax.invert_yaxis()
            for i, (v, pct) in enumerate(zip(missing_df['Jumlah Missing'], missing_df['Persentase'])):
                ax.text(pct + 0.3, i, f'{pct:.1f}% ({v:,})', va='center', color='white', fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.patch.set_alpha(0)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with col_miss2:
            total_missing = int(missing.sum())
            total_cells = df_main.shape[0] * df_main.shape[1]
            st.markdown(f"""<div class='insight-box'>
            <b>Ringkasan Missing Values:</b><br><br>
            • Total sel kosong: <b>{total_missing:,}</b> dari {total_cells:,} sel<br>
            • Tingkat kelengkapan: <b>{completeness:.1f}%</b><br>
            • Kolom dengan missing terbanyak: <b>{missing_df.iloc[0]['Kolom']}</b> ({missing_df.iloc[0]['Persentase']:.1f}%)<br><br>
            <b>Penanganan:</b> Kolom kategorikal diisi dengan modus (nilai paling sering muncul), 
            sedangkan kolom numerik diisi dengan median. Imputasi hanya dilakukan pada data latih 
            untuk mencegah kebocoran data (<i>data leakage</i>).
            </div>""", unsafe_allow_html=True)
    else:
        st.success("✅ Tidak ditemukan nilai yang hilang pada sampel data. Dataset ini sangat bersih!")
    
    # Duplicates
    st.markdown("<div class='section-header'>🔄 Deteksi Duplikat</div>", unsafe_allow_html=True)
    n_dupes = df_main.duplicated().sum()
    n_dupes_id = df_main.duplicated(subset=['id']).sum()
    
    c1, c2 = st.columns(2)
    with c1:
        dupe_color = 'quality-good' if n_dupes == 0 else 'quality-warn'
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Baris Duplikat Penuh</div>
            <div class='kpi-value {dupe_color}' style='font-size:1.5rem;'>{n_dupes:,}</div>
            <div class='kpi-sub'>dari {df_main.shape[0]:,} baris sampel</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        dupe_id_color = 'quality-good' if n_dupes_id == 0 else 'quality-warn'
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>ID Klaim Duplikat</div>
            <div class='kpi-value {dupe_id_color}' style='font-size:1.5rem;'>{n_dupes_id:,}</div>
            <div class='kpi-sub'>Berdasarkan kolom 'id'</div>
        </div>""", unsafe_allow_html=True)
    
    # Outlier Summary
    st.markdown("<div class='section-header'>📏 Deteksi Outlier (Metode IQR)</div>", unsafe_allow_html=True)
    st.markdown("Outlier diidentifikasi menggunakan metode *Interquartile Range (IQR)*. Nilai yang berada di luar 1.5×IQR dari Q1 atau Q3 dianggap outlier.")
    
    num_cols_for_outlier = ['biaya', 'usia', 'kelasrawat']
    num_cols_available = [c for c in num_cols_for_outlier if c in df_main.columns and pd.api.types.is_numeric_dtype(df_main[c])]
    
    if num_cols_available:
        outlier_data = []
        for col in num_cols_available:
            Q1 = df_main[col].quantile(0.25)
            Q3 = df_main[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            n_outliers = int(((df_main[col] < lower) | (df_main[col] > upper)).sum())
            outlier_data.append({
                'Kolom': col, 'Q1': f'{Q1:,.1f}', 'Q3': f'{Q3:,.1f}', 'IQR': f'{IQR:,.1f}',
                'Batas Bawah': f'{lower:,.1f}', 'Batas Atas': f'{upper:,.1f}',
                'Jumlah Outlier': n_outliers, 'Persentase': f'{n_outliers/len(df_main)*100:.2f}%'
            })
        st.dataframe(pd.DataFrame(outlier_data), use_container_width=True, hide_index=True)
    
    # Data Types Summary
    st.markdown("<div class='section-header'>📋 Ringkasan Tipe Data</div>", unsafe_allow_html=True)
    dtype_counts = df_main.dtypes.value_counts()
    dtype_df = pd.DataFrame({'Tipe Data': dtype_counts.index.astype(str), 'Jumlah Kolom': dtype_counts.values})
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
# TAB 3: DISTRIBUTION & ANALYSIS
# ══════════════════════════════════════════════
with tab3:
    plt.style.use('dark_background')
    
    st.markdown("<div class='section-header'>💰 Distribusi Biaya Klaim</div>", unsafe_allow_html=True)
    
    col_cost1, col_cost2 = st.columns(2)
    
    with col_cost1:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        df_filtered = df_main[df_main['biaya'] < df_main['biaya'].quantile(0.95)]
        sns.kdeplot(data=df_filtered, x='biaya', hue='label', fill=True, common_norm=False, 
                    palette={0: '#38bdf8', 1: '#f43f5e'}, alpha=0.4, linewidth=2, ax=ax)
        ax.set_title('Density Plot: Biaya Klaim (< Persentil 95)', color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel('Biaya Klaim (Rupiah)', color='#94a3b8')
        ax.set_ylabel('Density', color='#94a3b8')
        ax.legend(['Bukan Fraud (0)', 'Fraud (1)'], facecolor='#1e293b', edgecolor='none')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.patch.set_alpha(0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    with col_cost2:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        data_box = [df_filtered[df_filtered['label']==0]['biaya'], df_filtered[df_filtered['label']==1]['biaya']]
        bp = ax.boxplot(data_box, labels=['Bukan Fraud (0)', 'Fraud (1)'], patch_artist=True, widths=0.5,
                       medianprops=dict(color='white', linewidth=2))
        bp['boxes'][0].set_facecolor('#38bdf8')
        bp['boxes'][0].set_alpha(0.5)
        bp['boxes'][1].set_facecolor('#f43f5e')
        bp['boxes'][1].set_alpha(0.5)
        ax.set_title('Boxplot: Perbandingan Biaya per Kelas', color='white', fontsize=12, fontweight='bold')
        ax.set_ylabel('Biaya Klaim (Rupiah)', color='#94a3b8')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.patch.set_alpha(0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    st.markdown("""<div class='insight-box'>
    <b>💡 Interpretasi:</b> Klaim yang teridentifikasi fraud memiliki distribusi biaya yang sedikit <b>bergeser ke kanan</b> 
    (lebih mahal) dibandingkan klaim aman. Boxplot menunjukkan bahwa median biaya klaim fraud 
    cenderung lebih tinggi, meskipun terdapat tumpang tindih yang signifikan antara kedua kelas. 
    Ini menunjukkan bahwa biaya saja tidak cukup untuk mendeteksi fraud. Oleh karena itu, diperlukan kombinasi banyak fitur secara bersamaan.
    </div>""", unsafe_allow_html=True)
    
    # Age & Gender
    st.markdown("<div class='section-header'>👥 Demografi Pasien</div>", unsafe_allow_html=True)
    col_demo1, col_demo2 = st.columns(2)
    
    with col_demo1:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.histplot(data=df_main, x='usia', hue='label', bins=30, kde=True, multiple='stack',
                     palette={0: '#38bdf8', 1: '#f43f5e'}, alpha=0.6, ax=ax)
        ax.set_title('Distribusi Usia Pasien', color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel('Usia (Tahun)', color='#94a3b8')
        ax.set_ylabel('Jumlah Klaim', color='#94a3b8')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.patch.set_alpha(0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_demo2:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        gender_fraud = df_main.groupby(['jenkel', 'label']).size().unstack(fill_value=0)
        if 0 in gender_fraud.columns and 1 in gender_fraud.columns:
            gender_fraud.plot(kind='bar', stacked=True, color=['#38bdf8', '#f43f5e'], ax=ax, width=0.5)
        ax.set_title('Distribusi Jenis Kelamin & Fraud', color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel('Jenis Kelamin', color='#94a3b8')
        ax.set_ylabel('Jumlah Klaim', color='#94a3b8')
        ax.legend(['Bukan Fraud', 'Fraud'], facecolor='#1e293b', edgecolor='none')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.patch.set_alpha(0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    # Hospital Type & Ward Class Analysis
    st.markdown("<div class='section-header'>🏥 Analisis per Tipe Fasilitas Kesehatan</div>", unsafe_allow_html=True)
    col_hosp1, col_hosp2 = st.columns(2)
    
    with col_hosp1:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if 'typefaskes' in df_main.columns:
            faskes_fraud = df_main.groupby('typefaskes')['label'].mean().sort_values(ascending=False) * 100
            colors_faskes = ['#f43f5e' if v > fraud_ratio*100*1.5 else '#f59e0b' if v > fraud_ratio*100 else '#38bdf8' for v in faskes_fraud.values]
            ax.barh(faskes_fraud.index, faskes_fraud.values, color=colors_faskes, height=0.5)
            ax.set_xlabel('Persentase Fraud (%)', color='#94a3b8')
            ax.set_title('Tingkat Fraud per Tipe Faskes', color='white', fontsize=12, fontweight='bold')
            ax.axvline(x=fraud_ratio*100, color='#64748b', linestyle='--', label=f'Rata-rata ({fraud_ratio*100:.1f}%)')
            ax.legend(facecolor='#1e293b', edgecolor='none')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        fig.patch.set_alpha(0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    with col_hosp2:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if 'kelasrawat' in df_main.columns:
            kelas_fraud = df_main.groupby('kelasrawat')['label'].mean().sort_values(ascending=False) * 100
            colors_kelas = ['#f43f5e' if v > fraud_ratio*100*1.5 else '#f59e0b' if v > fraud_ratio*100 else '#38bdf8' for v in kelas_fraud.values]
            ax.bar(kelas_fraud.index.astype(str), kelas_fraud.values, color=colors_kelas, width=0.4)
            ax.set_xlabel('Kelas Rawat', color='#94a3b8')
            ax.set_ylabel('Persentase Fraud (%)', color='#94a3b8')
            ax.set_title('Tingkat Fraud per Kelas Rawat', color='white', fontsize=12, fontweight='bold')
            ax.axhline(y=fraud_ratio*100, color='#64748b', linestyle='--', label=f'Rata-rata ({fraud_ratio*100:.1f}%)')
            ax.legend(facecolor='#1e293b', edgecolor='none')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        fig.patch.set_alpha(0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    # Correlation Heatmap
    st.markdown("<div class='section-header'>🔗 Korelasi antar Fitur Numerik</div>", unsafe_allow_html=True)
    num_cols = df_main.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) > 2:
        corr_matrix = df_main[num_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 7))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, 
                    fmt='.2f', linewidths=0.5, ax=ax, annot_kws={"size": 8},
                    cbar_kws={"shrink": 0.8})
        ax.set_title('Matriks Korelasi Pearson', color='white', fontsize=13, fontweight='bold')
        fig.patch.set_alpha(0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Strongest correlations with label
        if 'label' in corr_matrix.columns:
            label_corr = corr_matrix['label'].drop('label').abs().sort_values(ascending=False)
            top_corrs = label_corr.head(5)
            st.markdown(f"""<div class='insight-box'>
            <b>📊 Korelasi Terkuat dengan Label Fraud:</b><br><br>
            {'<br>'.join([f"• <b>{col}</b>: {corr_matrix['label'][col]:+.3f}" for col in top_corrs.index])}<br><br>
            <i>Catatan: Korelasi Pearson mengukur hubungan linear. Hubungan non-linear yang lebih kompleks 
            akan ditangkap oleh model XGBoost melalui interaksi fitur di dalam pohon keputusan.</i>
            </div>""", unsafe_allow_html=True)
    
    # ── CLINICAL PATTERNS ANALYSIS ──
    st.markdown("<div class='section-header'>🩺 Pola Medis & Clinical Intelligence</div>", unsafe_allow_html=True)
    st.markdown("Peta sebaran kode diagnosis ICD-10 dan kode prosedur ICD-9-CM yang paling sering dijumpai pada berkas klaim terindikasi fraud.")
    
    df_labels = df_main[['id', 'label']]
    df_diag_labelled = df_diag.merge(df_labels, on='id') if df_diag is not None else None
    df_proc_labelled = df_proc.merge(df_labels, on='id') if df_proc is not None else None
    
    col_med1, col_med2 = st.columns(2)
    
    with col_med1:
        if df_diag_labelled is not None and not df_diag_labelled.empty:
            df_diag_fraud = df_diag_labelled[df_diag_labelled['label'] == 1]
            diag_col = 'kddiag' if 'kddiag' in df_diag_fraud.columns else 'diag'
            top_diags = df_diag_fraud[diag_col].value_counts().head(10)
            
            if not top_diags.empty:
                fig, ax = plt.subplots(figsize=(8, 4.5))
                # Soft reddish palette for warning indications
                sns.barplot(x=top_diags.values, y=top_diags.index.astype(str), palette='flare', ax=ax)
                ax.set_title('Top 10 Diagnosis ICD-10 Terkait Fraud', color='white', fontsize=12, fontweight='bold')
                ax.set_xlabel('Jumlah Kemunculan', color='#94a3b8')
                ax.set_ylabel('Kode ICD-10', color='#94a3b8')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                fig.patch.set_alpha(0)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("ℹ️ Tidak ada data diagnosis fraud yang cukup dalam sampel.")
        else:
            st.info("ℹ️ Data tabel diagnosis tidak tersedia.")
            
    with col_med2:
        if df_proc_labelled is not None and not df_proc_labelled.empty:
            df_proc_fraud = df_proc_labelled[df_proc_labelled['label'] == 1]
            top_procs = df_proc_fraud['proc'].value_counts().head(10)
            
            if not top_procs.empty:
                fig, ax = plt.subplots(figsize=(8, 4.5))
                # Soft purple/pink palette
                sns.barplot(x=top_procs.values, y=top_procs.index.astype(str), palette='crest', ax=ax)
                ax.set_title('Top 10 Prosedur ICD-9-CM Terkait Fraud', color='white', fontsize=12, fontweight='bold')
                ax.set_xlabel('Jumlah Kemunculan', color='#94a3b8')
                ax.set_ylabel('Kode ICD-9-CM', color='#94a3b8')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                fig.patch.set_alpha(0)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("ℹ️ Tidak ada data prosedur fraud yang cukup dalam sampel.")
        else:
            st.info("ℹ️ Data tabel prosedur tidak tersedia.")
            
    st.markdown("""<div class='insight-box'>
    <b>💡 Interpretasi Klinis:</b> Analisis ini memetakan diagnosis dan tindakan medis yang paling rentan disalahgunakan. 
    Klaim fraud sering kali berpusat pada kode-kode penunjang umum seperti diagnosis administratif (misal: kode Z) 
    atau prosedur pemantauan yang mudah digelembunkan secara kuantitas tanpa konfirmasi fisik yang ketat. 
    Hal ini membantu auditor menentukan prioritas fokus audit berbasis jenis penyakit dan tindakan medis.
    </div>""", unsafe_allow_html=True)

    # Key Insights
    st.markdown("<div class='section-header'>🎯 Temuan Utama (Key Insights)</div>", unsafe_allow_html=True)
    
    # Auto-generate insights from data
    median_fraud_cost = df_main[df_main['label']==1]['biaya'].median()
    median_safe_cost = df_main[df_main['label']==0]['biaya'].median()
    cost_diff_pct = ((median_fraud_cost - median_safe_cost) / median_safe_cost) * 100
    
    most_common_faskes = df_main['typefaskes'].mode().iloc[0] if 'typefaskes' in df_main.columns else 'N/A'
    avg_age = df_main['usia'].mean()
    
    st.markdown(f"""<div class='insight-box'>
    <b>1. Ketidakseimbangan Kelas Ekstrem:</b> Hanya <b>{fraud_ratio*100:.2f}%</b> klaim yang teridentifikasi 
    sebagai fraud, sehingga diperlukan teknik penyeimbangan data (SMOTE + RUS) untuk melatih model.<br><br>
    <b>2. Biaya Klaim Fraud Lebih Tinggi:</b> Median biaya klaim fraud (<b>Rp {median_fraud_cost:,.0f}</b>) 
    adalah <b>{cost_diff_pct:+.1f}%</b> lebih {'tinggi' if cost_diff_pct > 0 else 'rendah'} 
    dibandingkan klaim aman (<b>Rp {median_safe_cost:,.0f}</b>).<br><br>
    <b>3. Tipe Faskes Terbanyak:</b> Mayoritas klaim berasal dari Faskes tipe <b>{most_common_faskes}</b>.<br><br>
    <b>4. Rata-rata Usia Pasien:</b> <b>{avg_age:.1f} tahun</b>, menunjukkan populasi pasien didominasi 
    usia produktif dan lansia awal.<br><br>
    <b>5. Korelasi Fitur:</b> Biaya klaim, usia, dan kelas rawat menunjukkan korelasi tertentu dengan label fraud, 
    namun tidak ada satu fitur pun yang cukup kuat untuk memprediksi fraud secara mandiri, 
    sehingga kita memerlukan model Machine Learning multivariabel.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 4: DATA DICTIONARY & PREVIEW
# ══════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>📖 Kamus Data (Data Dictionary)</div>", unsafe_allow_html=True)
    st.markdown("Penjelasan setiap kolom pada tabel klaim utama untuk memahami makna masing-masing fitur.")
    
    feature_dict = pd.DataFrame([
        {"Kolom": "id", "Tipe": "String", "Deskripsi": "Identitas unik setiap transaksi klaim rumah sakit"},
        {"Kolom": "id_peserta", "Tipe": "String", "Deskripsi": "Identitas unik peserta BPJS Kesehatan"},
        {"Kolom": "jenkel", "Tipe": "Kategori", "Deskripsi": "Jenis kelamin pasien. L = Laki-laki, P = Perempuan"},
        {"Kolom": "usia", "Tipe": "Numerik", "Deskripsi": "Usia pasien saat klaim diajukan (dalam tahun)"},
        {"Kolom": "pisat", "Tipe": "Kategori", "Deskripsi": "Segmen kepesertaan BPJS (relasi keluarga: 1=kepala keluarga, dst.)"},
        {"Kolom": "typefaskes", "Tipe": "Kategori", "Deskripsi": "Tipe/kelas rumah sakit. A=terbesar, B, C, D=terkecil, SC/SD=khusus"},
        {"Kolom": "dati2", "Tipe": "Kategori", "Deskripsi": "Kode Daerah Tingkat II (kabupaten/kota) tempat fasilitas kesehatan berada"},
        {"Kolom": "jenispel", "Tipe": "Kategori", "Deskripsi": "Jenis pelayanan. 1=Rawat Inap (RITL), 2=Rawat Jalan (RJTL)"},
        {"Kolom": "kelasrawat", "Tipe": "Numerik", "Deskripsi": "Kelas kamar rawat inap. 1=VIP, 2=Standar, 3=Ekonomi"},
        {"Kolom": "politujuan", "Tipe": "Kategori", "Deskripsi": "Poli/departemen tujuan pasien di rumah sakit"},
        {"Kolom": "tgldatang", "Tipe": "Tanggal", "Deskripsi": "Tanggal pasien masuk/diterima di rumah sakit"},
        {"Kolom": "tglpulang", "Tipe": "Tanggal", "Deskripsi": "Tanggal pasien keluar/pulang dari rumah sakit"},
        {"Kolom": "jenispulang", "Tipe": "Kategori", "Deskripsi": "Status keluaran: sembuh, dirujuk, meninggal, dll."},
        {"Kolom": "diagfktp", "Tipe": "String", "Deskripsi": "Diagnosis dari Faskes Tingkat Pertama (kode ICD-10)"},
        {"Kolom": "cbg", "Tipe": "String", "Deskripsi": "Kode INA-CBG (Case Based Group) untuk menentukan tarif klaim. Format: CMG-Tipe-Spesifik-Severity"},
        {"Kolom": "biaya", "Tipe": "Numerik", "Deskripsi": "Total biaya klaim yang ditagihkan ke BPJS (dalam Rupiah)"},
        {"Kolom": "kdsa/kdsp/kdsr/kdsi/kdsd", "Tipe": "String", "Deskripsi": "Kode KDS (Komponen Diagnosis Spesifik) untuk prosedur dan kondisi khusus"},
        {"Kolom": "label", "Tipe": "Binary", "Deskripsi": "Target variabel: 0 = Klaim Aman (Bukan Fraud), 1 = Klaim Fraud"},
    ])
    st.dataframe(feature_dict, use_container_width=True, hide_index=True)
    
    st.markdown("<div class='section-header'>👁️ Preview Data Mentah (10 Baris Pertama)</div>", unsafe_allow_html=True)
    st.dataframe(df_main.head(10), use_container_width=True)
    
    st.markdown("<div class='section-header'>📊 Statistik Deskriptif</div>", unsafe_allow_html=True)
    st.dataframe(df_main.describe().T.round(2), use_container_width=True)
