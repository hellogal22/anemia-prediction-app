import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.svm import SVC

# =========================================================
# CONFIG & SETTING HALAMAN
# =========================================================
st.set_page_config(
    page_title="Aplikasi Analisis & Klasifikasi Anemia", 
    layout="wide",
    page_icon="🩸"
)

st.title("🩸 Aplikasi Analisis & Klasifikasi Anemia")
st.write("Oleh: **Galuh** (UTS Data Mining)")
st.markdown("---")

# =========================================================
# LOAD DATA & ARTIFACTS (CACHE AGAR JALAN LEBIH CEPAT)
# =========================================================
@st.cache_data
def load_data():
    # Membaca dataset utama
    df = pd.read_excel("SKILICARSLAN_Anemia_DataSet.xlsx")
    df_clean = df.dropna()
    return df, df_clean

@st.cache_resource
def load_models():
    # Memuat model dan scaler yang sudah disimpan sebelumnya
    scaler = joblib.load('anemia_scaler.pkl')
    model_xgb = joblib.load('best_anemia_model.pkl')
    return scaler, model_xgb

# Menjalankan fungsi load
try:
    df, df_clean = load_data()
    scaler, best_xgb = load_models()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat file data/model: {e}")
    st.info("Pastikan file dataset (.xlsx), model (.pkl), dan scaler (.pkl) berada di folder yang sama dengan app.py")

# =========================================================
# PEMBUATAN TAB STRUKTUR
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Eksplorasi Data (EDA)", 
    "⚙️ Preprocessing", 
    "🤖 Klasifikasi Model Awal", 
    "🚀 Hyperparameter Tuning",
    "🔮 Prediksi Mandiri (Fitur Baru)"
])

# =========================================================
# TAB 1: EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================
with tab1:
    st.header("Visualisasi & Distribusi Data")
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.subheader("5 Data Pertama")
        st.dataframe(df.head())
    
    with col_info2:
        st.subheader("Informasi Ukuran Dataset")
        st.write(f"**Ukuran Awal:** {df.shape[0]} baris, {df.shape[1]} kolom")
        st.write(f"**Setelah Drop Missing Value:** {df_clean.shape[0]} baris, {df_clean.shape[1]} kolom")

    st.markdown("---")
    
    col_graph1, col_graph2 = st.columns(2)
    with col_graph1:
        st.subheader("Distribusi Target Anemia")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x='HGB_Anemia_Class', data=df_clean, ax=ax)
        plt.title('Distribusi Target Anemia')
        st.pyplot(fig)
        
    with col_graph2:
        st.subheader("Heatmap Korelasi")
        fig, ax = plt.subplots(figsize=(10, 7))
        # Hanya menghitung korelasi kolom numerik
        sns.heatmap(df_clean.select_dtypes(include=[np.number]).corr(), cmap='coolwarm', ax=ax)
        plt.title('Heatmap Korelasi')
        st.pyplot(fig)

# =========================================================
# TAB 2: PREPROCESSING
# =========================================================
with tab2:
    st.header("Proses Preprocessing Data")
    st.write("Langkah-langkah yang dilakukan pada tahap ini meliputi:")
    st.markdown("""
    1. **Pembersihan Data**: Menghapus baris yang memiliki *missing value* menggunakan fungsi `dropna()`.
    2. **Pemisahan Fitur & Target**: Mengisolasikan variabel `HGB_Anemia_Class` sebagai target ($y$) dan menghapus kolom kelas spesifik lainnya dari fitur ($X$).
    3. **Splitting Data**: Membagi dataset menjadi **80% Data Training** dan **20% Data Testing** secara seimbang (*stratified*).
    4. **Feature Scaling**: Menyamakan skala data fitur menggunakan `StandardScaler` demi optimasi performa algoritma.
    """)
    
    st.code("""
# Memisahkan Fitur dan Target
X = df.drop(columns=['HGB_Anemia_Class', 'All_Class', 'Iron_anemia_Class', 'Folate_anemia_class', 'B12_Anemia_class'])
y = df['HGB_Anemia_Class']

# Train-Test Split (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
    """, language='python')

# =========================================================
# TAB 3: KLASIFIKASI MODEL AWAL
# =========================================================
with tab3:
    st.header("Performa Model Awal (Sebelum Tuning)")
    st.write("Berikut adalah perbandingan performa dasar antara algoritma **Support Vector Machine (SVM)** dan **XGBoost Classifier**:")
    
    # Dummy data performa berdasarkan hasil script Python asli Anda
    df_performa_awal = pd.DataFrame({
        'Model': ['SVM', 'XGBoost'],
        'Accuracy': [0.95, 0.98],
        'Precision': [0.94, 0.97],
        'Recall': [0.95, 0.98],
        'F1 Score': [0.94, 0.97]
    }).set_index('Model')
    
    st.table(df_performa_awal)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    df_performa_awal.plot(kind='bar', ax=ax)
    plt.title('Perbandingan Performa Model Awal')
    plt.ylabel('Score')
    st.pyplot(fig)

# =========================================================
# TAB 4: HYPERPARAMETER TUNING
# =========================================================
with tab4:
    st.header("Optimasi via Hyperparameter Tuning")
    st.write("Proses pencarian parameter terbaik dilakukan menggunakan metode `GridSearchCV` dengan 3-Fold Cross Validation.")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.subheader("Hasil Terbaik SVM")
        st.json({"C": 10, "gamma": "scale", "kernel": "rbf"})
        st.metric(label="Akurasi Terbaik SVM Tuning", value="96.2%")
        
    with col_t2:
        st.subheader("Hasil Terbaik XGBoost")
        st.json({"learning_rate": 0.1, "max_depth": 5, "n_estimators": 100})
        st.metric(label="Akurasi Terbaik XGBoost Tuning", value="98.7%")

# =========================================================
# TAB 5: PREDIKSI MANDIRI (FITUR BARU INTERAKTIF)
# =========================================================
with tab5:
    st.header("🔮 Form Prediksi Anemia Interaktif")
    st.write("Masukkan nilai hasil tes laboratorium darah lengkap di bawah ini untuk menguji performa model secara langsung.")
    
    # Membuat form input data
    with st.form("form_prediksi"):
        col_in1, col_in2 = st.columns(2)
        
        with col_in1:
            hgb = st.number_input("HGB (Hemoglobin) - g/dL", min_value=0.0, max_value=30.0, value=12.1, step=0.1)
            rbc = st.number_input("RBC (Red Blood Cells) - 10^6/µL", min_value=0.0, max_value=15.0, value=4.3, step=0.1)
            hct = st.number_input("HCT (Hematocrit) - %", min_value=0.0, max_value=100.0, value=37.6, step=0.1)
            mcv = st.number_input("MCV (Mean Corpuscular Volume) - fL", min_value=0.0, max_value=150.0, value=87.2, step=0.1)
            mch = st.number_input("MCH (Mean Corpuscular Hemoglobin) - pg", min_value=0.0, max_value=50.0, value=29.5, step=0.1)
            mchc = st.number_input("MCHC (MHC Concentration) - g/dL", min_value=0.0, max_value=50.0, value=33.8, step=0.1)
            rdw = st.number_input("RDW (Red Cell Distribution Width) - %", min_value=0.0, max_value=50.0, value=12.8, step=0.1)
            
        with col_in2:
            gender = st.selectbox("GENDER (Jenis Kelamin)", options=[0, 1], format_func=lambda x: "Perempuan (0)" if x == 0 else "Laki-laki (1)")
            wbc = st.number_input("WBC (White Blood Cells)", value=10.6)
            ne = st.number_input("NE# (Neutrophils)", value=6.3)
            ly = st.number_input("LY# (Lymphocytes)", value=2.7)
            mo = st.number_input("MO# (Monocytes)", value=0.9)
            eo = st.number_input("EO# (Eosinophils)", value=0.5)
            ba = st.number_input("BA# (Basophils)", value=0.06)
            plt_val = st.number_input("PLT (Platelets)", value=364.0)
            mpv = st.number_input("MPV", value=9.6)
            pct = st.number_input("PCT", value=0.35)

        # Tombol submit di dalam form
        submit_button = st.form_submit_button(label="Cek Hasil Prediksi", type="primary")

    if submit_button:
        # 1. Mengumpulkan semua variabel input menjadi satu baris array.
        # Catatan: Pastikan susunan kolom ini PERSIS sama dengan susunan kolom X pada file asli saat melatih model.
        raw_inputs = [
            gender, wbc, ne, ly, mo, eo, ba, rbc, hgb, hct, mcv, mch, mchc, rdw, plt_val, mpv, pct
        ]
        
        # Mengubah ke format 2D Array untuk dimasukkan ke Scaler & Model
        input_array = np.array([raw_inputs])
        
        try:
            # 2. Lakukan transformasi scaling menggunakan object scaler yang telah di-load
            input_scaled = scaler.transform(input_array)
            
            # 3. Jalankan prediksi menggunakan model XGBoost terbaik hasil Tuning
            prediksi = best_xgb.predict(input_scaled)
            probabilitas = best_xgb.predict_proba(input_scaled)[0]
            
            st.markdown("---")
            st.subheader("📋 Kesimpulan Analisis Sistem:")
            
            # 4. Tampilkan output visual berdasarkan hasil prediksi kelas
            if prediksi[0] == 1:
                st.error("🚨 **Hasil Prediksi: TERDETEKSI ANEMIA (POSITIF)**")
                st.write(f"Sistem mendeteksi indikasi anemia dengan tingkat keyakinan model sebesar **{probabilitas[1] * 100:.2f}%**.")
            else:
                st.success("✅ **Hasil Prediksi: KONDISI NORMAL / NEGATIF ANEMIA**")
                st.write(f"Sistem menyatakan kondisi darah normal dengan tingkat keyakinan model sebesar **{probabilitas[0] * 100:.2f}%**.")
                
        except Exception as prediction_error:
            st.error(f"Gagal memproses prediksi. Pastikan jumlah fitur input ({len(raw_inputs)}) sesuai dengan dimensi model. Detail Error: {prediction_error}")

    # Catatan kaki / Disclaimer akademis
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption("⚠️ **Disclaimer:** Aplikasi ini dibangun menggunakan model kecerdasan buatan berbasis contoh dataset untuk memenuhi tugas UTS Data Mining. Hasil prediksi tidak dapat dijadikan acuan klinis mutlak pengganti diagnosa dokter medis.")
