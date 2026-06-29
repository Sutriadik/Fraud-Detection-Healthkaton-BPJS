import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

# Set config
st.set_page_config(page_title="About Project - Fraud Detection", page_icon="ℹ️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 1.8rem;
        margin-bottom: 1.5rem;
    }
    .header-style {
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='header-style'>ℹ️ About the Project</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown("""
    <div class="glass-card">
        <h4 style="margin: 0 0 0.8rem 0; color: #38bdf8;">🧠 Technical Architecture & Pipeline</h4>
        <p>Aplikasi ini dirancang menggunakan arsitektur pipa data terstruktur dan termodulasi (Modular Pipeline) untuk memisahkan logika pemrosesan data, pemodelan, dan penyajian antarmuka:</p>
        <ol>
            <li><b>Data Loader (Ingestion):</b> Memuat data relasional klaim dari file Parquet, melakukan validasi schema, tipe data, nilai null, dan duplikasi.</li>
            <li><b>Feature Engineering:</b> Menghitung lama perawatan rumah sakit (lama_rawat) dan memecah struktur kode INA-CBG / KDS (CMG, Tipe Kasus, Kode Spesifik, Severity Level) secara optimal.</li>
            <li><b>Structural Preprocessing:</b> Melakukan cleaning teks diagnosis ICD-10, menggabungkan diagnosis sekunder dan tindakan medis (ICD-9-CM), dan mengimputasi missing values dengan parameter yang di-fit pada data latih.</li>
            <li><b>One-Hot & Ordinal Encoding:</b> Pengodean variabel kategori dengan batas kategori yang diketahui (known categories) untuk mencegah data leakage saat inferensi.</li>
            <li><b>SMOTE + RUS Balancing:</b> Mengatasi ketimpangan kelas target (fraud hanya ~1.4%) dengan kombinasi oversampling kelas minoritas (SMOTE) dan undersampling kelas mayoritas (RUS).</li>
            <li><b>XGBoost Classifier:</b> Melatih ensemble tree model dengan early stopping pada data validasi untuk mencegah overfitting.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h4 style="margin: 0 0 0.8rem 0; color: #38bdf8;">📜 Healthcare Regulations Context</h4>
        <p><b>Sistem INA-CBG (Indonesia Case Based Groups):</b></p>
        <p>Sistem INA-CBG adalah sistem pembayaran klaim rumah sakit (FKRTL) secara paket berdasarkan diagnosis penyakit dan prosedur yang diberikan kepada pasien. Struktur kode INA-CBG diatur dalam <i>Permenkes No. 27 Tahun 2014</i>, terdiri dari 4 karakter:</p>
        <ul>
            <li><b>CMG (Case-Mix Main Groups):</b> Karakter pertama berupa huruf alfabet yang menunjukkan kelompok organ/spesialisasi (misal: <code>K</code> untuk Penyakit Sistem Pencernaan).</li>
            <li><b>Tipe Kasus:</b> Karakter kedua berupa angka (1-9) menunjukkan tipe pelayanan (misal: <code>4</code> untuk Rawat Inap non-Bedah).</li>
            <li><b>Spesifik Kasus:</b> Karakter ketiga menunjukkan jenis kasus spesifik.</li>
            <li><b>Severity Level:</b> Karakter keempat berupa angka romawi (I, II, III) atau angka arab (0) menunjukkan tingkat keparahan (ringan, sedang, berat).</li>
        </ul>
        <p>Model ML memanfaatkan pemisahan fitur ini untuk mendeteksi kejanggalan kombinasi (upcoding) di mana rumah sakit menaikkan severity level atau memalsukan diagnosis utama demi mendapatkan nilai klaim paket yang lebih tinggi dari BPJS.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glass-card">
        <h4 style="margin: 0 0 0.8rem 0; color: #38bdf8;">💼 Business Fraud Typologies</h4>
        <p>Beberapa modus kecurangan klaim asuransi kesehatan yang dideteksi oleh sistem pemodelan ini:</p>
        <ul>
            <li><b>Upcoding:</b> Mengubah kode diagnosis utama atau tingkat severity menjadi lebih parah demi menaikkan tarif paket klaim.</li>
            <li><b>Phantom Billing:</b> Mengajukan biaya klaim atas tindakan medis atau obat yang sebenarnya tidak pernah diberikan kepada pasien (terdeteksi dari ketidakwajaran jumlah prosedur atau lama rawat).</li>
            <li><b>Unbundling:</b> Memecah satu paket pelayanan terpadu menjadi beberapa klaim tindakan terpisah untuk mendapatkan keuntungan lebih besar.</li>
            <li><b>Repeat Claims:</b> Mengajukan klaim ganda atas pasien dan penyakit yang sama dalam waktu berdekatan secara tidak wajar.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h4 style="margin: 0 0 0.8rem 0; color: #38bdf8;">👨‍💻 MLE Portfolio Metadata</h4>
        <p>Proyek ini dikembangkan menggunakan best-practices Software Engineering & MLOps:</p>
        <ul>
            <li><b>Modular Code:</b> Logika pemrosesan terpisah dari notebook eksperimen untuk portabilitas produksi.</li>
            <li><b>Linting & Clean Code:</b> Mengikuti panduan gaya PEP8, DRY principle, dan SOLID design.</li>
            <li><b>Reproducibility:</b> Semua seed acak dikunci pada angka <code>42</code>, dan dependencies dimuat dalam <code>requirements.txt</code>.</li>
            <li><b>Fast Verification:</b> Menyertakan test-suite berbasis <code>pytest</code> untuk memastikan pipeline inferensi bebas bug.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# System Flow Diagram
st.markdown("### 🖥️ Pipeline Data & Inferensi End-to-End")
st.markdown("""
```mermaid
graph TD
    A[Input Data Mentah / Form / CSV] --> B[Data Loader & Input Validation]
    B --> C[Feature Engineering: Stay Duration & Split CBG/KDS]
    C --> D[Data Preprocessor: Clean Diagnosa & ICD-10 Chapter Mapping]
    D --> E[ColumnTransformer Encoders: One-Hot & Ordinal]
    E --> F[Inference: Tuned XGBoost Model with Custom Threshold]
    F --> G[Hasil Prediksi: Fraud/Bukan Fraud, Probability, & Confidence Score]
```
""")
st.sidebar.success("Navigasi Berhasil Dimuat.")
