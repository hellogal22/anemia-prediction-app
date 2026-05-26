import streamlit as pd
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score
)

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Anemia Classification App", layout="wide")

st.title("🩸 Aplikasi Analisis & Klasifikasi Anemia")
st.markdown("Oleh: **Galuh** (UTS Data Mining)")
st.write("---")

# 1. LOAD DATASET (Menggunakan caching agar tidak reload terus-menerus)
@st.cache_data
def load_data():
    # Pastikan file excel berada di folder yang sama dengan app.py
    df = pd.read_excel("SKILICARSLAN_Anemia_DataSet.xlsx")
    df_clean = df.dropna()
    return df, df_clean

try:
    df_raw, df = load_data()
except FileNotFoundError:
    st.error("❌ File 'SKILICARSLAN_Anemia_DataSet.xlsx' tidak ditemukan. Pastikan file berada di folder yang sama dengan script ini.")
    st.stop()

# Buat Menu Navigasi Menggunakan Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Eksplorasi Data (EDA)", 
    "⚙️ Preprocessing", 
    "🤖 Klasifikasi Model Awal", 
    "🚀 Hyperparameter Tuning"
])

# ==========================================
# TAB 1: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
with tab1:
    st.header("Visualisasi & Distribusi Data")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("5 Data Pertama")
        st.dataframe(df.head())
    with col2:
        st.subheader("Informasi Ukuran Dataset")
        st.write(f"**Ukuran Awal:** {df_raw.shape[0]} baris, {df_raw.shape[1]} kolom")
        st.write(f"**Setelah Drop Missing Value:** {df.shape[0]} baris, {df.shape[1]} kolom")

    st.write("---")
    
    # Distribusi Target & Heatmap side-by-side
    col3, col4 = st.columns([1, 2])
    with col3:
        st.subheader("Distribusi Target Anemia")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(x='HGB_Anemia_Class', data=df, ax=ax)
        plt.title('Distribusi Target Anemia')
        st.pyplot(fig)
        
    with col4:
        st.subheader("Heatmap Korelasi")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), cmap='coolwarm', ax=ax)
        plt.title('Heatmap Korelasi')
        st.pyplot(fig)

    st.write("---")
    st.subheader("Analisis Fitur Penting")
    fitur_penting = ['HGB', 'RBC', 'HCT', 'MCV', 'MCH', 'MCHC']
    
    # Pilih Fitur untuk ditampilkan grafiknya
    selected_feature = st.selectbox("Pilih fitur untuk melihat Distribusi dan Outlier:", fitur_penting)
    
    col5, col6 = st.columns(2)
    with col5:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[selected_feature], kde=True, ax=ax)
        plt.title(f'Distribusi {selected_feature}')
        st.pyplot(fig)
    with col6:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=df[selected_feature], ax=ax)
        plt.title(f'Boxplot {selected_feature}')
        st.pyplot(fig)

# ==========================================
# TAB 2: PREPROCESSING
# ==========================================
with tab2:
    st.header("Proses Pemisahan Data & Scaling")
    
    # Memisahkan Fitur dan Target
    X = df.drop(columns=['HGB_Anemia_Class', 'All_Class', 'Iron_anemia_Class', 'Folate_anemia_class', 'B12_Anemia_class'])
    y = df['HGB_Anemia_Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.success("✅ Data berhasil dipisahkan menjadi Data Train (80%) dan Data Test (20%) dengan Stratified Sampling.")
    
    col_pre1, col_pre2 = st.columns(2)
    with col_pre1:
        st.write("**Ukuran Matriks Fitur (X):**", X.shape)
        st.write("**Ukuran Target (y):**", y.shape)
    with col_pre2:
        st.write("**Jumlah Data Train:**", X_train.shape[0])
        st.write("**Jumlah Data Test:**", X_test.shape[0])

# ==========================================
# TAB 3: MODELLING AWAL
# ==========================================
with tab3:
    st.header("Perbandingan Model Awal (Tanpa Tuning)")
    
    # Tombol jalankan model awal
    if st.button("Jalankan Model SVM & XGBoost Awal"):
        with st.spinner("Melatih model... Mohon tunggu..."):
            # SVM
            svm_model = SVC(kernel='rbf', C=1, gamma='scale', probability=True, class_weight='balanced', random_state=42)
            svm_model.fit(X_train_scaled, y_train)
            y_pred_svm = svm_model.predict(X_test_scaled)
            y_prob_svm = svm_model.predict_proba(X_test_scaled)[:,1]
            
            # XGBoost
            xgb_model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss')
            xgb_model.fit(X_train, y_train)
            y_pred_xgb = xgb_model.predict(X_test)
            y_prob_xgb = xgb_model.predict_proba(X_test)[:,1]
            
            # Buat Dataframe Hasil
            hasil = pd.DataFrame({
                'Model': ['SVM', 'XGBoost'],
                'Accuracy': [accuracy_score(y_test, y_pred_svm), accuracy_score(y_test, y_pred_xgb)],
                'Precision': [precision_score(y_test, y_pred_svm), precision_score(y_test, y_pred_xgb)],
                'Recall': [recall_score(y_test, y_pred_svm), recall_score(y_test, y_pred_xgb)],
                'F1 Score': [f1_score(y_test, y_pred_svm), f1_score(y_test, y_pred_xgb)],
                'ROC AUC': [roc_auc_score(y_test, y_prob_svm), roc_auc_score(y_test, y_prob_xgb)]
            })
            
            st.subheader("Metrik Performa Model")
            st.dataframe(hasil.style.highlight_max(axis=0, color='lightgreen', subset=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']))
            
            # Grafik Perbandingan
            fig, ax = plt.subplots(figsize=(10, 5))
            hasil.set_index('Model').plot(kind='bar', ax=ax)
            plt.ylabel('Score')
            plt.title('Perbandingan Performa Model Awal')
            st.pyplot(fig)
            
            # Feature Importance XGBoost
            st.write("---")
            st.subheader("Top Feature Importance (XGBoost)")
            importance = pd.DataFrame({'Feature': X.columns, 'Importance': xgb_model.feature_importances_}).sort_values(by='Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(data=importance.head(10), x='Importance', y='Feature', ax=ax)
            st.pyplot(fig)

# ==========================================
# TAB 4: HYPERPARAMETER TUNING
# ==========================================
with tab4:
    st.header("Optimasi Model dengan GridSearchCV")
    st.write("Klik tombol di bawah ini untuk memulai pencarian parameter terbaik.")
    
    if st.button("Mulai Hyperparameter Tuning"):
        with st.spinner("GridSearchCV sedang berjalan (Proses ini memakan waktu beberapa saat)..."):
            
            # --- TUNING SVM ---
            param_svm = {'C': [1, 10], 'gamma': ['scale'], 'kernel': ['rbf']}
            grid_svm = GridSearchCV(estimator=SVC(probability=True), param_grid=param_svm, cv=3, scoring='accuracy', n_jobs=-1)
            grid_svm.fit(X_train_scaled, y_train)
            
            best_svm = grid_svm.best_estimator_
            y_pred_svm_t = best_svm.predict(X_test_scaled)
            
            # --- TUNING XGBOOST ---
            param_xgb = {'n_estimators': [100], 'max_depth': [3, 5], 'learning_rate': [0.1]}
            grid_xgb = GridSearchCV(estimator=XGBClassifier(random_state=42, eval_metric='logloss'), param_grid=param_xgb, cv=3, scoring='accuracy', n_jobs=-1)
            grid_xgb.fit(X_train, y_train)
            
            best_xgb = grid_xgb.best_estimator_
            y_pred_xgb_t = best_xgb.predict(X_test)
            
            # --- TAMPILKAN HASIL TUNING ---
            st.success("🎉 Proses Hyperparameter Tuning Selesai!")
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.info("**Parameter Terbaik SVM:**")
                st.write(grid_svm.best_params_)
                st.write(f"Best CV Accuracy: {grid_svm.best_score_:.4f}")
            with col_t2:
                st.info("**Parameter Terbaik XGBoost:**")
                st.write(grid_xgb.best_params_)
                st.write(f"Best CV Accuracy: {grid_xgb.best_score_:.4f}")
                
            # Tabel Perbandingan Akhir
            hasil_tuning = pd.DataFrame({
                'Model': ['SVM Tuning', 'XGBoost Tuning'],
                'Accuracy': [accuracy_score(y_test, y_pred_svm_t), accuracy_score(y_test, y_pred_xgb_t)],
                'Precision': [precision_score(y_test, y_pred_svm_t), precision_score(y_test, y_pred_xgb_t)],
                'Recall': [recall_score(y_test, y_pred_svm_t), recall_score(y_test, y_pred_xgb_t)],
                'F1 Score': [f1_score(y_test, y_pred_svm_t), f1_score(y_test, y_pred_xgb_t)]
            })
            
            st.write("---")
            st.subheader("Hasil Perbandingan Akhir Setelah Tuning")
            st.dataframe(hasil_tuning)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            hasil_tuning.set_index('Model').plot(kind='bar', ax=ax)
            plt.ylabel('Score')
            st.pyplot(fig)

# 1. Tambahkan tab baru di kode Anda
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Eksplorasi Data (EDA)", 
    "⚙️ Preprocessing", 
    "🤖 Klasifikasi Model Awal", 
    "🚀 Hyperparameter Tuning",
    "🔮 Prediksi Mandiri"  # <-- TAB BARU
])

with tab1:
    # ... Kode EDA kamu yang sekarang sudah muncul di gambar ...
    st.subheader("Visualisasi & Distribusi Data")

with tab2:
    # ... Kode Preprocessing kamu ...

with tab3:
    # ... Kode Klasifikasi Model Awal kamu ...

with tab4:
    # ... Kode Hyperparameter Tuning kamu ...

# =========================================================
# 2. ISI TAB BARU UNTUK INPUT USER
# =========================================================
with tab5:
    st.subheader("Masukkan Nilai Laboratorium untuk Prediksi Anemia")
    
    # Buat input form
    col1, col2 = st.columns(2)
    with col1:
        hgb = st.number_input("HGB (Hemoglobin)", value=12.0)
        rbc = st.number_input("RBC (Red Blood Cells)", value=4.5)
        hct = st.number_input("HCT (Hematocrit)", value=40.0)
    with col2:
        mv = st.number_input("MCV", value=85.0)
        mch = st.number_input("MCH", value=30.0)
        mchc = st.number_input("MCHC", value=33.0)
        
    if st.button("Jalankan Prediksi"):
        # Lakukan scaling dan gunakan model terbaikmu (best_xgb atau best_svm)
        # Pastikan kamu sudah meload 'scaler' dan 'best_xgb' di bagian atas app.py
        
        input_data = [[hgb, rbc, hct, mv, mch, mchc]]
        input_scaled = scaler.transform(input_data)
        
        # Contoh memakai model xgb tuning kamu
        prediksi = best_xgb.predict(input_scaled) 
        
        if prediksi[0] == 1:
            st.error("🚨 Hasil: Terdeteksi Anemia")
        else:
            st.success("✅ Hasil: Kondisi Normal / Tidak Anemia")
