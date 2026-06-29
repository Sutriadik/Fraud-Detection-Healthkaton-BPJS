import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import os
import pandas as pd
import numpy as np
from src.prediction.predict import InferencePipeline

def get_safe_index(options_list, val, default_idx=0):
    try:
        if len(options_list) > 0:
            target_type = type(options_list[0])
            # If the option item is a string, strip and match
            if issubclass(target_type, str):
                val = str(val).strip()
                # Try finding case-insensitive or partial match
                for i, opt in enumerate(options_list):
                    if opt.strip().upper() == val.upper():
                        return i
                    if opt.split(" - ")[0].strip().upper() == val.upper():
                        return i
            else:
                val = target_type(val)
        return options_list.index(val)
    except (ValueError, TypeError, KeyError):
        return default_idx

# Set config
st.set_page_config(page_title="Fraud Prediction - Claim Verification", page_icon="🔍", layout="wide")

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
    .section-header {
        font-size: 1.3rem; font-weight: 700; color: #e2e8f0; margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem; border-bottom: 2px solid rgba(56, 189, 248, 0.2);
    }
    .context-card {
        background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255,255,255,0.05);
        border-radius: 14px; padding: 1.3rem 1.5rem; margin-bottom: 1rem;
    }
    .insight-box {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.08) 0%, rgba(129, 140, 248, 0.08) 100%);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 12px; padding: 1.2rem 1.5rem; margin: 0.8rem 0; line-height: 1.7;
    }
    .risk-critical {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.2) 0%, rgba(244, 63, 94, 0.15) 100%);
        border: 2px solid rgba(244, 63, 94, 0.5);
        padding: 1.5rem; border-radius: 16px; margin: 1rem 0;
    }
    .risk-high {
        background: linear-gradient(135deg, rgba(249, 115, 22, 0.15) 0%, rgba(244, 63, 94, 0.1) 100%);
        border: 2px solid rgba(249, 115, 22, 0.4);
        padding: 1.5rem; border-radius: 16px; margin: 1rem 0;
    }
    .risk-medium {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(251, 191, 36, 0.1) 100%);
        border: 2px solid rgba(245, 158, 11, 0.4);
        padding: 1.5rem; border-radius: 16px; margin: 1rem 0;
    }
    .risk-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(34, 211, 153, 0.1) 100%);
        border: 2px solid rgba(16, 185, 129, 0.4);
        padding: 1.5rem; border-radius: 16px; margin: 1rem 0;
    }
    .kpi-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(30, 41, 59, 0.4) 100%);
        backdrop-filter: blur(12px); border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        padding: 1.2rem 1.4rem; margin-bottom: 1rem; text-align: center;
    }
    .kpi-value { font-size: 1.9rem; font-weight: 800; margin: 0.3rem 0; font-family: 'Inter', sans-serif; }
    .kpi-label { font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600; }
    .kpi-sub { font-size: 0.75rem; color: #64748b; margin-top: 0.2rem; }
    .disclaimer-box {
        background: rgba(100, 116, 139, 0.1);
        border: 1px solid rgba(100, 116, 139, 0.2);
        border-radius: 10px; padding: 1rem 1.2rem; margin-top: 1rem;
        font-size: 0.85rem; color: #94a3b8; line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='header-gradient'>🔍 Verifikasi Klaim & Prediksi Fraud</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Masukkan data klaim kesehatan untuk memverifikasi apakah terdapat indikasi kecurangan (fraud). Sistem akan menganalisis profil risiko klaim berdasarkan model Machine Learning yang telah dilatih pada data historis BPJS Kesehatan.</p>", unsafe_allow_html=True)

# Check if model exists
model_path = "models/final_model.joblib"
if not os.path.exists(model_path):
    st.warning("⚠️ Model belum tersedia. Jalankan perintah training terlebih dahulu: `PYTHONPATH=. python3 src/models/train.py`")
    st.stop()

# Initialize pipeline
@st.cache_resource
def load_pipeline():
    return InferencePipeline()

pipeline = load_pipeline()

# ──────────────────────────────────────────────
# MODE SELECTION
# ──────────────────────────────────────────────
pred_mode = st.radio(
    "Pilih Mode Verifikasi", 
    ["📝 Prediksi Manual (Isi Form)", "📁 Prediksi Massal (Upload CSV)"], 
    horizontal=True
)

# ══════════════════════════════════════════════
# MANUAL PREDICTION
# ══════════════════════════════════════════════
if pred_mode == "📝 Prediksi Manual (Isi Form)":
    
    # ── SESSION STATE INITIALIZATION ──
    if "claim_id_val" not in st.session_state:
        st.session_state.claim_id_val = "CLM-99120"
    if "patient_id_val" not in st.session_state:
        st.session_state.patient_id_val = "PES-88127"
    if "usia_val" not in st.session_state:
        st.session_state.usia_val = 45
    if "jenkel_val" not in st.session_state:
        st.session_state.jenkel_val = "L"
    if "typefaskes_val" not in st.session_state:
        st.session_state.typefaskes_val = "C"
    if "dati2_val" not in st.session_state:
        st.session_state.dati2_val = "291"
    if "jenispel_val" not in st.session_state:
        st.session_state.jenispel_val = "1 - Rawat Inap (RITL)"
    if "kelasrawat_val" not in st.session_state:
        st.session_state.kelasrawat_val = 3
    if "pisat_val" not in st.session_state:
        st.session_state.pisat_val = "1"
    if "jenispulang_display_val" not in st.session_state:
        st.session_state.jenispulang_display_val = "1 - Sembuh atau Membaik"
    if "biaya_val" not in st.session_state:
        st.session_state.biaya_val = 3500000.0
    if "tgldatang_val" not in st.session_state:
        st.session_state.tgldatang_val = pd.to_datetime("2022-06-01")
    if "tglpulang_val" not in st.session_state:
        st.session_state.tglpulang_val = pd.to_datetime("2022-06-04")
    if "cbg_val" not in st.session_state:
        st.session_state.cbg_val = "K-4-17-I"
    if "primary_diag_val" not in st.session_state:
        st.session_state.primary_diag_val = "K29.7"
    if "secondary_diags_str_val" not in st.session_state:
        st.session_state.secondary_diags_str_val = "I10, E11.9"
    if "diagfktp_val" not in st.session_state:
        st.session_state.diagfktp_val = "K29.7"
    if "procedures_str_val" not in st.session_state:
        st.session_state.procedures_str_val = "99.04, 88.76"
    if "kdsa_val" not in st.session_state:
        st.session_state.kdsa_val = "-"
    if "kdsp_val" not in st.session_state:
        st.session_state.kdsp_val = "-"
    if "kdsr_val" not in st.session_state:
        st.session_state.kdsr_val = "-"
    if "kdsi_val" not in st.session_state:
        st.session_state.kdsi_val = "-"
    if "kdsd_val" not in st.session_state:
        st.session_state.kdsd_val = "-"
    if "trigger_auto_submit" not in st.session_state:
        st.session_state.trigger_auto_submit = False

    # Context guide
    st.markdown("""<div class='insight-box'>
    <b>📖 Panduan Pengisian:</b> Form ini dirancang untuk memverifikasi satu klaim kesehatan secara manual. 
    Setiap field dilengkapi dengan penjelasan dan contoh nilai. Jika Anda tidak memiliki data untuk field tertentu, 
    gunakan nilai default yang telah disediakan. Field opsional (seperti KDS) boleh dikosongkan dengan tanda "<b>-</b>".
    </div>""", unsafe_allow_html=True)
    
    # ── SCENARIO EXAMPLE LOADER ──
    st.markdown("### 💡 Muat Skenario Contoh")
    st.caption("Klik salah satu skenario di bawah untuk memuat data pengisian otomatis yang terverifikasi.")
    
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    with col_ex1:
        if st.button("🟢 Contoh Klaim Aman (Normal)", use_container_width=True, help="Memuat data klaim rawat jalan ringan dengan biaya rendah."):
            st.session_state.claim_id_val = "CLM-1491273"
            st.session_state.patient_id_val = "PES-1491273"
            st.session_state.usia_val = 19
            st.session_state.jenkel_val = "P"
            st.session_state.typefaskes_val = "KI"
            st.session_state.dati2_val = "111"
            st.session_state.jenispel_val = "2 - Rawat Jalan (RJTL)"
            st.session_state.kelasrawat_val = 3
            st.session_state.pisat_val = "4"
            st.session_state.jenispulang_display_val = "1 - Sembuh atau Membaik"
            st.session_state.biaya_val = 190400.0
            st.session_state.tgldatang_val = pd.to_datetime("2020-01-01")
            st.session_state.tglpulang_val = pd.to_datetime("2020-01-01")
            st.session_state.cbg_val = "Q-5-44-0"
            st.session_state.primary_diag_val = "N75.0"
            st.session_state.secondary_diags_str_val = "-"
            st.session_state.diagfktp_val = "N75.0"
            st.session_state.procedures_str_val = "-"
            st.session_state.kdsa_val = "-"
            st.session_state.kdsp_val = "-"
            st.session_state.kdsr_val = "-"
            st.session_state.kdsi_val = "-"
            st.session_state.kdsd_val = "-"
            st.session_state.trigger_auto_submit = True
            st.rerun()
            
    with col_ex2:
        if st.button("🔴 Contoh Fraud: Neonatal Jaundice (RITL)", use_container_width=True, help="Memuat kasus bayi baru lahir dengan perawatan 7 hari yang diindikasikan fraud upcoding/prolonged stay."):
            st.session_state.claim_id_val = "CLM-9502457"
            st.session_state.patient_id_val = "PES-7530739"
            st.session_state.usia_val = 0
            st.session_state.jenkel_val = "L"
            st.session_state.typefaskes_val = "B"
            st.session_state.dati2_val = "187"
            st.session_state.jenispel_val = "1 - Rawat Inap (RITL)"
            st.session_state.kelasrawat_val = 3
            st.session_state.pisat_val = "4"
            st.session_state.jenispulang_display_val = "1 - Sembuh atau Membaik"
            st.session_state.biaya_val = 0.0 # Will trigger auto-imputation
            st.session_state.tgldatang_val = pd.to_datetime("2018-09-05")
            st.session_state.tglpulang_val = pd.to_datetime("2018-09-12")
            st.session_state.cbg_val = "P-8-17-I"
            st.session_state.primary_diag_val = "P59.8"
            st.session_state.secondary_diags_str_val = "P03.4"
            st.session_state.diagfktp_val = "P03.4"
            st.session_state.procedures_str_val = "99.83"
            st.session_state.kdsa_val = "-"
            st.session_state.kdsp_val = "-"
            st.session_state.kdsr_val = "-"
            st.session_state.kdsi_val = "-"
            st.session_state.kdsd_val = "-"
            st.session_state.trigger_auto_submit = True
            st.rerun()
            
    with col_ex3:
        if st.button("🔴 Contoh Fraud: Infeksi Bakteri (RJTL)", use_container_width=True, help="Memuat kasus remaja rawat jalan (0 hari) yang diindikasikan fraud upcoding/upcharging."):
            st.session_state.claim_id_val = "CLM-7744641"
            st.session_state.patient_id_val = "PES-7301298"
            st.session_state.usia_val = 12
            st.session_state.jenkel_val = "L"
            st.session_state.typefaskes_val = "C"
            st.session_state.dati2_val = "123"
            st.session_state.jenispel_val = "2 - Rawat Jalan (RJTL)"
            st.session_state.kelasrawat_val = 3
            st.session_state.pisat_val = "4"
            st.session_state.jenispulang_display_val = "1 - Sembuh atau Membaik"
            st.session_state.biaya_val = 0.0 # Will trigger auto-imputation
            st.session_state.tgldatang_val = pd.to_datetime("2020-01-02")
            st.session_state.tglpulang_val = pd.to_datetime("2020-01-02")
            st.session_state.cbg_val = "Q-5-42-0"
            st.session_state.primary_diag_val = "A49.9"
            st.session_state.secondary_diags_str_val = "-"
            st.session_state.diagfktp_val = "R11"
            st.session_state.procedures_str_val = "-"
            st.session_state.kdsa_val = "-"
            st.session_state.kdsp_val = "-"
            st.session_state.kdsr_val = "-"
            st.session_state.kdsi_val = "-"
            st.session_state.kdsd_val = "-"
            st.session_state.trigger_auto_submit = True
            st.rerun()

    # Notice to user if auto submit is ready
    if st.session_state.trigger_auto_submit:
        st.info("ℹ️ Skenario berhasil dimuat. Klik tombol **Verifikasi Klaim** di bagian bawah form untuk menjalankan prediksi.")

    with st.form("manual_prediction_form"):
        
        # ── SECTION 1: PATIENT INFO ──
        st.markdown("<div class='section-header'>👤 Informasi Pasien</div>", unsafe_allow_html=True)
        st.caption("Data dasar tentang pasien yang mengajukan klaim kesehatan.")
        
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1:
            claim_id = st.text_input(
                "🆔 Claim ID", value=st.session_state.claim_id_val, 
                help="Nomor identifikasi unik transaksi klaim. Biasanya diberikan oleh sistem rumah sakit."
            )
        with col_p2:
            patient_id = st.text_input(
                "🆔 Patient ID", value=st.session_state.patient_id_val, 
                help="Nomor identifikasi peserta BPJS Kesehatan."
            )
        with col_p3:
            usia = st.number_input(
                "📅 Usia Pasien (Tahun)", min_value=0, max_value=120, value=st.session_state.usia_val, 
                help="Usia pasien saat klaim diajukan. Rentang normal: 0-120 tahun. Contoh: 45 untuk pasien dewasa."
            )
        with col_p4:
            jenkel_options = ["L", "P"]
            jenkel = st.selectbox(
                "⚤ Jenis Kelamin", jenkel_options, index=get_safe_index(jenkel_options, st.session_state.jenkel_val), 
                help="L = Laki-laki, P = Perempuan."
            )

        # ── SECTION 2: HOSPITAL & SERVICE ──
        st.markdown("<div class='section-header'>🏥 Informasi Rumah Sakit & Pelayanan</div>", unsafe_allow_html=True)
        st.caption("Data tentang fasilitas kesehatan yang memberikan layanan dan jenis pelayanan yang diberikan.")
        
        col_h1, col_h2, col_h3, col_h4 = st.columns(4)
        with col_h1:
            typefaskes_options = ["A", "B", "C", "D", "SC", "SD", "KI", "KP", "TP", "RB"]
            typefaskes = st.selectbox(
                "🏥 Tipe Faskes (Kelas RS)", typefaskes_options, index=get_safe_index(typefaskes_options, st.session_state.typefaskes_val), 
                help="Klasifikasi rumah sakit berdasarkan kapasitas layanan. A = RS besar/pendidikan, B = RS menengah, C = RS kecil, D = RS sangat kecil, SC/SD = RS khusus."
            )
        with col_h2:
            dati2 = st.text_input(
                "📍 Kode Wilayah (DATI2)", value=st.session_state.dati2_val, 
                help="Kode Daerah Tingkat II (Kabupaten/Kota) tempat rumah sakit berada. Lihat kode BPS untuk referensi."
            )
        with col_h3:
            jenispel_options = ["1 - Rawat Inap (RITL)", "2 - Rawat Jalan (RJTL)"]
            jenispel = st.selectbox(
                "🛏️ Jenis Pelayanan", 
                options=jenispel_options, index=get_safe_index(jenispel_options, st.session_state.jenispel_val), 
                help="RITL = Rawat Inap Tingkat Lanjut (pasien menginap). RJTL = Rawat Jalan Tingkat Lanjut (pasien pulang hari yang sama)."
            )
        with col_h4:
            kelasrawat_options = [1, 2, 3]
            kelasrawat = st.selectbox(
                "🛌 Kelas Rawat", kelasrawat_options, index=get_safe_index(kelasrawat_options, st.session_state.kelasrawat_val), 
                help="Kelas kamar rawat inap: 1 = VIP/Kelas 1 (paling mahal), 2 = Kelas 2 (standar), 3 = Kelas 3 (ekonomi/paling murah)."
            )
        
        col_h5, col_h6 = st.columns(2)
        with col_h5:
            pisat_options = ["1", "2", "3", "4", "5"]
            pisat = st.selectbox(
                "👪 Segmen Kepesertaan (PISAT)", pisat_options, index=get_safe_index(pisat_options, st.session_state.pisat_val), 
                help="Relasi peserta dalam keluarga: 1 = Kepala Keluarga, 2 = Pasangan, 3 = Anak ke-1, 4 = Anak ke-2, 5 = Anak ke-3 dst."
            )
        with col_h6:
            jenispulang_display_options = ["1 - Sembuh or Membaik", "1 - Sembuh atau Membaik", "2 - Dirujuk ke Faskes Lain", "3 - Atas Permintaan Sendiri", "4 - Meninggal Dunia", "5 - Lainnya"]
            jenispulang_display = st.selectbox(
                "🚪 Status Keluar", 
                options=jenispulang_display_options,
                index=get_safe_index(jenispulang_display_options, st.session_state.jenispulang_display_val),
                help="Status keluaran pasien dari rumah sakit setelah dirawat."
            )

        # ── SECTION 3: FINANCIAL & DURATION ──
        st.markdown("<div class='section-header'>💰 Informasi Finansial & Durasi Rawat</div>", unsafe_allow_html=True)
        st.caption("Data biaya klaim dan tanggal rawat inap. Biaya abnormal tinggi atau durasi rawat tidak wajar dapat menjadi indikator fraud.")
        
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            biaya = st.number_input(
                "💵 Biaya Klaim (Rupiah)", min_value=0.0, value=st.session_state.biaya_val, step=100000.0, 
                help="Total biaya yang ditagihkan rumah sakit ke BPJS untuk klaim ini. Isi 0 untuk membiarkan model mengimputasi biaya secara otomatis (imputasi median)."
            )
        with col_f2:
            tgldatang = st.date_input(
                "📅 Tanggal Masuk RS", value=st.session_state.tgldatang_val, 
                help="Tanggal pertama kali pasien diterima/dirawat di rumah sakit."
            )
        with col_f3:
            tglpulang = st.date_input(
                "📅 Tanggal Keluar RS", value=st.session_state.tglpulang_val, 
                help="Tanggal pasien dipulangkan dari rumah sakit. Selisih dengan tanggal masuk = durasi rawat."
            )
        
        stay_days = (tglpulang - tgldatang).days
        if stay_days < 0:
            st.error("⚠️ Tanggal keluar tidak boleh lebih awal dari tanggal masuk!")
        elif stay_days == 0:
            st.info(f"ℹ️ Durasi rawat: **0 hari** (rawat jalan / one-day care)")
        else:
            st.info(f"ℹ️ Durasi rawat: **{stay_days} hari**")

        # ── SECTION 4: MEDICAL CODES ──
        st.markdown("<div class='section-header'>🩺 Kode Medis (INA-CBG, Diagnosis & Prosedur)</div>", unsafe_allow_html=True)
        st.caption("Kode-kode medis yang menentukan tarif klaim. Kode ini biasanya diisi oleh petugas rumah sakit. Jika tidak diketahui, gunakan nilai default.")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            cbg = st.text_input(
                "📋 Kode INA-CBG", value=st.session_state.cbg_val, 
                help="Kode INA-CBG menentukan tarif klaim. Format: CMG-TipeKasus-SpesifikKasus-Severity."
            )
            primary_diag = st.text_input(
                "🔬 Diagnosis Utama (ICD-10)", value=st.session_state.primary_diag_val, 
                help="Kode ICD-10 diagnosis utama. Contoh: A09.9 (Diare), E11.9 (Diabetes), I10 (Hipertensi), J18.9 (Pneumonia), K29.7 (Gastritis)."
            )
            secondary_diags_str = st.text_input(
                "🔬 Diagnosis Sekunder (pisahkan koma)", value=st.session_state.secondary_diags_str_val, 
                help="Kode ICD-10 diagnosis penyerta, dipisahkan koma. Boleh dikosongkan jika tidak ada."
            )
        with col_m2:
            diagfktp = st.text_input(
                "📋 Diagnosis Faskes Pertama", value=st.session_state.diagfktp_val, 
                help="Diagnosis awal dari Puskesmas/klinik yang merujuk pasien."
            )
            procedures_str = st.text_input(
                "🛠️ Kode Prosedur (ICD-9-CM, pisahkan koma)", value=st.session_state.procedures_str_val, 
                help="Kode prosedur/tindakan medis menggunakan ICD-9-CM. Boleh dikosongkan."
            )
            
            with st.expander("🔧 Kode KDS (Opsional, biasanya diisi oleh petugas RS)"):
                st.caption("KDS (Kode Diagnosis Spesifik) memberikan informasi tambahan tentang prosedur khusus. Isi dengan '-' jika tidak tersedia.")
                kdsa = st.text_input("KDSA (Prosedur Akut)", value=st.session_state.kdsa_val, help="Kode prosedur spesial akut")
                kdsp = st.text_input("KDSP (Prosedur)", value=st.session_state.kdsp_val, help="Kode prosedur spesial")
                kdsr = st.text_input("KDSR (Rawat)", value=st.session_state.kdsr_val, help="Kode rawat spesial")
                kdsi = st.text_input("KDSI (Intensif)", value=st.session_state.kdsi_val, help="Kode intensif spesial")
                kdsd = st.text_input("KDSD (Sub-akut)", value=st.session_state.kdsd_val, help="Kode sub-akut spesial")
        
        st.markdown("---")
        submit = st.form_submit_button("🔍 Verifikasi Klaim", use_container_width=True)
    
    # ── PREDICTION RESULTS ──
    if submit:
        # Reset auto submit trigger
        st.session_state.trigger_auto_submit = False
        
        jenispel_val = jenispel.split(" - ")[0]
        jenispulang_val = jenispulang_display.split(" - ")[0]
        
        claim_dict = {
            "id": claim_id, "id_peserta": patient_id,
            "jenispulang": str(float(jenispulang_val)),
            "jenkel": jenkel, "pisat": str(float(pisat)),
            "diagfktp": diagfktp, "biaya": biaya, "dati2": dati2,
            "typefaskes": typefaskes, "usia": usia,
            "tgldatang": tgldatang.strftime("%Y-%m-%d"),
            "tglpulang": tglpulang.strftime("%Y-%m-%d"),
            "jenispel": str(float(jenispel_val)),
            "cbg": cbg, "kelasrawat": kelasrawat,
            "kdsa": kdsa, "kdsp": kdsp, "kdsr": kdsr, "kdsi": kdsi, "kdsd": kdsd
        }
        
        diagnoses = [{"diag": primary_diag, "levelid": 1}]
        if secondary_diags_str.strip() and secondary_diags_str != "-":
            for code in secondary_diags_str.split(","):
                code = code.strip()
                if code:
                    diagnoses.append({"diag": code, "levelid": 2})
        
        procedures = []
        if procedures_str.strip() and procedures_str != "-":
            for code in procedures_str.split(","):
                code = code.strip()
                if code:
                    procedures.append(code)
        
        with st.spinner("⏳ Menganalisis profil risiko klaim..."):
            result = pipeline.predict_single(claim_dict, diagnoses, procedures)
        
        # Parse results
        is_fraud = result["is_fraud"]
        prob = result["fraud_probability"]
        conf = result["confidence_score"]
        exp = result["explanation"]
        
        # Determine risk level
        if prob >= 0.9:
            risk_level = "🔴 CRITICAL"
            risk_class = "risk-critical"
            risk_color = "#dc2626"
            risk_desc = "Risiko kecurangan sangat tinggi. Klaim ini sangat mencurigakan dan memerlukan audit mendesak."
            action = "⛔ TAHAN pembayaran klaim. Eskalasi ke tim investigasi fraud untuk audit lapangan segera."
        elif prob >= pipeline.threshold:
            risk_level = "🟠 HIGH"
            risk_class = "risk-high"
            risk_color = "#f97316"
            risk_desc = "Risiko kecurangan tinggi. Probabilitas fraud melebihi ambang batas keputusan model."
            action = "⚠️ TAHAN pembayaran sementara. Lakukan verifikasi dokumen pendukung dan cross-check dengan data historis pasien."
        elif prob >= 0.4:
            risk_level = "🟡 MEDIUM"
            risk_class = "risk-medium"
            risk_color = "#f59e0b"
            risk_desc = "Risiko menengah. Meskipun di bawah threshold, terdapat pola yang perlu diperhatikan."
            action = "📋 PROSES klaim dengan catatan. Tandai untuk review berkala dan monitoring pola klaim pasien ini ke depan."
        else:
            risk_level = "🟢 LOW"
            risk_class = "risk-low"
            risk_color = "#10b981"
            risk_desc = "Risiko rendah. Profil klaim konsisten dengan pola klaim yang aman."
            action = "✅ PROSES klaim secara normal. Tidak diperlukan investigasi tambahan."
        
        # Display Results
        st.markdown("<div class='section-header'>📊 Hasil Analisis Risiko Klaim</div>", unsafe_allow_html=True)
        
        # Risk Banner
        st.markdown(f"""<div class='{risk_class}'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div>
                    <h2 style='margin:0; color:{risk_color}; font-size:1.8rem;'>{risk_level} RISK</h2>
                    <p style='margin:0.5rem 0 0 0; color:#e2e8f0; font-size:1rem;'>{risk_desc}</p>
                </div>
                <div style='text-align:right;'>
                    <div style='font-size:2.5rem; font-weight:800; color:{risk_color};'>{prob*100:.1f}%</div>
                    <div style='font-size:0.8rem; color:#94a3b8;'>Fraud Probability</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
        
        # KPI Cards
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.markdown(f"""<div class='kpi-card'>
                <div class='kpi-label'>Probabilitas Fraud</div>
                <div class='kpi-value' style='color:{risk_color};'>{prob*100:.2f}%</div>
                <div class='kpi-sub'>Threshold: {pipeline.threshold:.4f}</div>
            </div>""", unsafe_allow_html=True)
        with col_r2:
            st.markdown(f"""<div class='kpi-card'>
                <div class='kpi-label'>Confidence Score</div>
                <div class='kpi-value' style='color:#38bdf8;'>{conf*100:.1f}%</div>
                <div class='kpi-sub'>Tingkat keyakinan model</div>
            </div>""", unsafe_allow_html=True)
        with col_r3:
            st.markdown(f"""<div class='kpi-card'>
                <div class='kpi-label'>Keputusan Model</div>
                <div class='kpi-value' style='color:{"#f43f5e" if is_fraud else "#10b981"};'>{"FRAUD" if is_fraud else "AMAN"}</div>
                <div class='kpi-sub'>Berdasarkan threshold {pipeline.threshold:.4f}</div>
            </div>""", unsafe_allow_html=True)
        
        # Explanation & Action
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.markdown(f"""<div class='context-card'>
                <h4 style='margin:0 0 0.8rem 0; color:#38bdf8;'>📝 Penjelasan Model</h4>
                <p style='line-height:1.7; margin:0;'>{exp}</p>
            </div>""", unsafe_allow_html=True)
        with col_exp2:
            st.markdown(f"""<div class='context-card'>
                <h4 style='margin:0 0 0.8rem 0; color:#c084fc;'>📋 Rekomendasi Tindakan</h4>
                <p style='line-height:1.7; margin:0;'>{action}</p>
            </div>""", unsafe_allow_html=True)
        
        # Contributing Factors
        st.markdown(f"""<div class='context-card'>
            <h4 style='margin:0 0 0.8rem 0; color:#818cf8;'>🔍 Faktor Kontribusi Utama</h4>
            <p style='line-height:1.7; margin:0;'>
            • <b>Biaya Klaim:</b> Rp {biaya:,.0f} {'(di atas rata-rata, yang menunjukkan faktor risiko tinggi)' if biaya > 5000000 else '(dalam rentang normal)'}<br>
            • <b>Durasi Rawat:</b> {stay_days} hari {'(durasi panjang, sehingga perlu verifikasi lapangan)' if stay_days > 7 else '(normal)'}<br>
            • <b>Diagnosis Sekunder:</b> {len(diagnoses)-1} kode {'(banyak diagnosis, berpotensi terjadi manipulasi upcoding)' if len(diagnoses)-1 > 3 else '(normal)'}<br>
            • <b>Prosedur Medis:</b> {len(procedures)} tindakan {'(banyak prosedur, sehingga perlu diverifikasi rekam medisnya)' if len(procedures) > 4 else '(normal)'}<br>
            • <b>Kode INA-CBG:</b> {cbg}, dengan Tipe Faskes: {typefaskes}, Kelas Rawat: {kelasrawat}
            </p>
        </div>""", unsafe_allow_html=True)
        
        # ── FRAUD CASE STUDY ILLUSTRATION (IF INDICTED AS FRAUD) ──
        if is_fraud == 1:
            # Logic to determine the most relevant case study
            if biaya > 10000000:
                case_title = "Penggelembungan Biaya (Upcoding dan Klaim Fiktif Tindakan)"
                case_story = (
                    f"Pasien {patient_id} tercatat melakukan perawatan dengan total klaim yang sangat besar, yaitu "
                    f"Rp {biaya:,.0f}. Pada kasus nyata, modus penggelembungan biaya sering kali terjadi ketika rumah sakit "
                    f"menambahkan diagnosis sekunder palsu yang tidak pernah diderita pasien, atau memasukkan prosedur penunjang mahal "
                    f"yang sebenarnya tidak pernah diberikan. Tujuannya adalah memaksa tagihan masuk ke dalam tarif INA-CBG yang "
                    f"jauh lebih mahal (misalnya memanipulasi tingkat keparahan dari tingkat Ringan ke Berat)."
                )
                case_how_to_audit = (
                    "Bandingkan berkas rekam medis fisik (catatan harian keperawatan, resep apotik, laporan ruang operasi) "
                    "dengan rincian billing yang ditagihkan. Pastikan semua obat dan alat habis pakai yang diklaim benar-benar "
                    "tercatat telah dikonsumsi atau digunakan oleh pasien."
                )
            elif stay_days > 7:
                case_title = "Perpanjangan Hari Rawat Tanpa Indikasi Medis (Prolonged Stay)"
                case_story = (
                    f"Pasien tercatat dirawat inap selama {stay_days} hari. Modus fraud jenis ini biasanya berupa menahan pasien "
                    f"lebih lama di bangsal perawatan meskipun kondisi klinisnya sudah membaik dan sudah layak pulang sejak hari ke-3. "
                    f"Rumah sakit sengaja melakukan ini untuk menagih biaya kamar dan visit dokter harian tambahan dari BPJS."
                )
                case_how_to_audit = (
                    "Audit lembar catatan perkembangan pasien terintegrasi (CPPT) dari dokter spesialis. Periksa indikasi "
                    "medis harian dan bandingkan dengan kriteria pemulangan pasien yang standar untuk melihat apakah ada hari rawat "
                    "yang tidak efisien atau tidak perlu."
                )
            elif len(diagnoses) - 1 > 3:
                case_title = "Manipulasi Diagnosis Penyerta (Upcoding Kompleks)"
                case_story = (
                    f"Klaim ini mencantumkan sebanyak {len(diagnoses)-1} diagnosis sekunder. Ini merupakan pola klasik manipulasi upcoding, "
                    f"di mana komplikasi medis ringan atau penyakit penyerta sederhana ditulis berlebihan atau ditambahkan penyakit berat "
                    f"(seperti gagal napas atau infeksi darah berat) tanpa bukti penunjang yang sah. Hal ini dilakukan demi mendongkrak "
                    f"severity level pada sistem INA-CBG dari level 1 ke level 3, yang menaikkan tarif ganti rugi hingga ratusan persen."
                )
                case_how_to_audit = (
                    "Periksa bukti hasil laboratorium atau pemeriksaan penunjang (USG, Rontgen) untuk setiap diagnosis sekunder yang diklaim. "
                    "Diagnosis sekunder tidak boleh ditagihkan jika pasien tidak terbukti menerima terapi obat atau tindakan penanganan khusus "
                    "terhadap penyakit penyerta tersebut."
                )
            else:
                case_title = "Klaim Fiktif Pelayanan / Phantom Billing"
                case_story = (
                    f"Klaim ini mencurigakan karena ketidaksesuaian kombinasi tipe rumah sakit ({typefaskes}) dengan kode INA-CBG ({cbg}). "
                    f"Modus phantom billing sering kali melibatkan rumah sakit yang menagihkan biaya paket perawatan penyakit tertentu yang "
                    f"sebenarnya tidak pernah dikerjakan secara utuh, atau bahkan mencatut nama kartu BPJS warga untuk diklaimkan layaknya "
                    f"pasien dirawat inap."
                )
                case_how_to_audit = (
                    "Lakukan konfirmasi langsung kepada pasien (tele-collecting atau kunjungan lapangan) untuk memverifikasi apakah "
                    "pasien benar-benar pernah dirawat inap di rumah sakit tersebut pada tanggal yang tertera, dan apakah benar-benar "
                    "menerima tindakan medis sesuai yang dilaporkan."
                )
                
            st.markdown(f"""<div class='context-card' style='border-left: 5px solid #dc2626; background: rgba(220, 38, 38, 0.05);'>
                <h4 style='margin:0 0 0.5rem 0; color:#f43f5e;'>📂 Studi Kasus Modus: {case_title}</h4>
                <p style='line-height:1.7; margin:0 0 1.2rem 0; font-style: italic; color: #e2e8f0;'>
                    "{case_story}"
                </p>
                <h5 style='margin:0 0 0.3rem 0; color:#38bdf8;'>🔍 Langkah Verifikasi Lapangan untuk Auditor:</h5>
                <p style='line-height:1.6; margin:0; color: #cbd5e1;'>
                    {case_how_to_audit}
                </p>
            </div>""", unsafe_allow_html=True)
        
        # Disclaimer
        st.markdown("""<div class='disclaimer-box'>
        <b>⚖️ Disclaimer:</b> Hasil prediksi ini dihasilkan oleh model Machine Learning dan bersifat 
        <b>probabilistik</b>. Keputusan final harus dilakukan oleh auditor/verifikator manusia yang 
        berwenang. Model ini tidak menggantikan proses audit profesional dan tidak dapat dijadikan 
        bukti hukum. Selalu lakukan verifikasi dokumen pendukung sebelum mengambil tindakan administratif.
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# BATCH PREDICTION
# ══════════════════════════════════════════════
elif pred_mode == "📁 Prediksi Massal (Upload CSV)":
    st.markdown("""<div class='insight-box'>
    <b>📖 Panduan Upload CSV:</b> Unggah file CSV yang berisi data klaim kesehatan. 
    File harus memiliki kolom-kolom yang sesuai dengan format data klaim BPJS. 
    Sistem akan memproses setiap klaim secara otomatis dan menghasilkan prediksi fraud beserta skor risiko.
    </div>""", unsafe_allow_html=True)
    
    with st.expander("📋 Lihat Format Kolom CSV yang Diperlukan", expanded=False):
        required_cols = pd.DataFrame([
            {"Kolom": "id", "Wajib": "✅ Ya", "Deskripsi": "ID unik klaim", "Contoh": "CLM-001"},
            {"Kolom": "id_peserta", "Wajib": "✅ Ya", "Deskripsi": "ID peserta BPJS", "Contoh": "PES-001"},
            {"Kolom": "usia", "Wajib": "✅ Ya", "Deskripsi": "Usia pasien (tahun)", "Contoh": "45"},
            {"Kolom": "jenkel", "Wajib": "✅ Ya", "Deskripsi": "Jenis kelamin (L/P)", "Contoh": "P"},
            {"Kolom": "typefaskes", "Wajib": "✅ Ya", "Deskripsi": "Tipe rumah sakit", "Contoh": "C"},
            {"Kolom": "biaya", "Wajib": "✅ Ya", "Deskripsi": "Biaya klaim (Rp)", "Contoh": "3500000"},
            {"Kolom": "cbg", "Wajib": "✅ Ya", "Deskripsi": "Kode INA-CBG", "Contoh": "K-4-17-I"},
            {"Kolom": "tgldatang", "Wajib": "✅ Ya", "Deskripsi": "Tanggal masuk RS", "Contoh": "2022-06-01"},
            {"Kolom": "tglpulang", "Wajib": "✅ Ya", "Deskripsi": "Tanggal keluar RS", "Contoh": "2022-06-04"},
            {"Kolom": "primary_diag", "Wajib": "✅ Ya", "Deskripsi": "Diagnosis utama ICD-10", "Contoh": "K29.7"},
            {"Kolom": "secondary_diags", "Wajib": "❌ Tidak", "Deskripsi": "Diagnosis sekunder (pisah koma)", "Contoh": "I10, E11.9"},
            {"Kolom": "procedures", "Wajib": "❌ Tidak", "Deskripsi": "Kode prosedur (pisah koma)", "Contoh": "99.04, 88.76"},
        ])
        st.dataframe(required_cols, use_container_width=True, hide_index=True)
    
    uploaded_file = st.file_uploader("📎 Pilih file CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.success(f"✅ File berhasil dimuat! **{len(df_upload):,} klaim** terdeteksi.")
            st.dataframe(df_upload.head(5), use_container_width=True)
            
            if st.button("🚀 Jalankan Prediksi Massal", use_container_width=True):
                with st.spinner("⏳ Memproses pipeline prediksi..."):
                    main_cols = ['id', 'id_peserta', 'jenispulang', 'jenkel', 'pisat', 'diagfktp', 'biaya', 
                                 'dati2', 'typefaskes', 'usia', 'tgldatang', 'tglpulang', 'jenispel', 'cbg', 
                                 'kelasrawat', 'kdsa', 'kdsp', 'kdsr', 'kdsi', 'kdsd']
                    
                    df_main_parsed = df_upload.copy()
                    if 'jenispulang' not in df_main_parsed.columns:
                        df_main_parsed['jenispulang'] = '1.0'
                    if 'diagfktp' not in df_main_parsed.columns and 'primary_diag' in df_main_parsed.columns:
                        df_main_parsed['diagfktp'] = df_main_parsed['primary_diag']
                    
                    existing_main_cols = [c for c in main_cols if c in df_main_parsed.columns]
                    df_main = df_main_parsed[existing_main_cols]
                    
                    diag_rows = []
                    for _, row in df_upload.iterrows():
                        c_id = row['id']
                        prim_diag = row.get('primary_diag', row.get('diagfktp', None))
                        if pd.notna(prim_diag):
                            diag_rows.append({"id": c_id, "diag": str(prim_diag), "levelid": 1})
                        sec_diags = row.get('secondary_diags', None)
                        if pd.notna(sec_diags):
                            for code in str(sec_diags).split(','):
                                code = code.strip()
                                if code:
                                    diag_rows.append({"id": c_id, "diag": code, "levelid": 2})
                    df_diagnosa = pd.DataFrame(diag_rows) if diag_rows else pd.DataFrame(columns=["id", "diag", "levelid"])
                    
                    proc_rows = []
                    for _, row in df_upload.iterrows():
                        c_id = row['id']
                        procs = row.get('procedures', None)
                        if pd.notna(procs):
                            for code in str(procs).split(','):
                                code = code.strip()
                                if code:
                                    proc_rows.append({"id": c_id, "proc": code})
                    df_proc = pd.DataFrame(proc_rows) if proc_rows else pd.DataFrame(columns=["id", "proc"])
                    
                    results = pipeline.predict_batch(df_main, df_diagnosa, df_proc)
                    df_results = df_upload.merge(results, on='id', how='left')
                
                # Results Display
                st.markdown("<div class='section-header'>📊 Hasil Prediksi Massal</div>", unsafe_allow_html=True)
                
                fraud_count = int(df_results['is_fraud'].sum())
                total_count = len(df_results)
                fraud_pct = fraud_count / total_count * 100
                
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f"""<div class='kpi-card'>
                        <div class='kpi-label'>Total Klaim</div>
                        <div class='kpi-value' style='color:#38bdf8;'>{total_count}</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""<div class='kpi-card'>
                        <div class='kpi-label'>Terdeteksi Fraud</div>
                        <div class='kpi-value' style='color:#f43f5e;'>{fraud_count}</div>
                    </div>""", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""<div class='kpi-card'>
                        <div class='kpi-label'>Rasio Fraud</div>
                        <div class='kpi-value' style='color:#f59e0b;'>{fraud_pct:.1f}%</div>
                    </div>""", unsafe_allow_html=True)
                with c4:
                    st.markdown(f"""<div class='kpi-card'>
                        <div class='kpi-label'>Klaim Aman</div>
                        <div class='kpi-value' style='color:#10b981;'>{total_count - fraud_count}</div>
                    </div>""", unsafe_allow_html=True)
                
                # Results table
                display_cols = ['id', 'fraud_probability', 'is_fraud', 'confidence_score']
                available_display = [c for c in display_cols if c in df_results.columns]
                if 'id_peserta' in df_results.columns:
                    available_display.insert(1, 'id_peserta')
                if 'biaya' in df_results.columns:
                    available_display.insert(2, 'biaya')
                    
                st.dataframe(
                    df_results[available_display].sort_values('fraud_probability', ascending=False), 
                    use_container_width=True, hide_index=True
                )
                
                csv_data = df_results.to_csv(index=False)
                st.download_button(
                    "📥 Download Hasil Prediksi (.CSV)", csv_data, "fraud_predictions.csv", "text/csv",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"❌ Gagal memproses file: {e}. Periksa format kolom CSV Anda.")
