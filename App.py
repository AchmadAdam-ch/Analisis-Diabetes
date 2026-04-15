import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier # Perubahan di sini
import os

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sistem Diagnosis Diabetes v4", page_icon="🏥", layout="centered")

# --- 2. FUNGSI LOADING & MODEL ---
@st.cache_resource
def siapkan_model_final():
    nama_file = 'Dataset_Diabetes_ML.xlsx'
    if not os.path.exists(nama_file):
        return None

    # Membaca data Excel
    df = pd.read_excel(nama_file)
    
    # Mapping manual (Memastikan konversi teks ke angka stabil)
    mapping_level = {'Rendah': 0, 'Sedang': 1, 'Tinggi': 2}
    mapping_biner = {'Tidak': 0, 'Ya': 1}
    
    df['Aktivitas'] = df['Aktivitas'].map(mapping_level)
    df['Stres'] = df['Stres'].map(mapping_level)
    df['Merokok'] = df['Merokok'].map(mapping_biner)
    df['Diabetes'] = df['Diabetes'].map(mapping_biner)
    
    # Isi data kosong jika ada
    df = df.fillna(df.median(numeric_only=True))

    # Fitur yang digunakan
    X = df[['Usia', 'Gula', 'Aktivitas', 'Tidur', 'Stres', 'Merokok', 'BMI']]
    y = df['Diabetes']
    
    # --- PERUBAHAN METODE: RANDOM FOREST ---
    # n_estimators=100 artinya kita membuat "hutan" dengan 100 pohon keputusan
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X, y)
    
    return model

# --- 3. TAMPILAN UTAMA ---
def main():
    st.markdown("<h2 style='text-align: center;'>Analisis Risiko Diabetes (Random Forest)</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    model = siapkan_model_final()

    if model is None:
        st.error("File 'Dataset_Diabetes_ML.xlsx' tidak ditemukan di folder!")
        return

    # Input Form
    with st.container():
        st.subheader("Data Pasien")
        col1, col2 = st.columns(2)
        
        with col1:
            usia = st.number_input("Usia", 1, 100, 25)
            gula = st.number_input("Kadar Gula Darah (mg/dL)", 50, 300, 90)
            bmi = st.number_input("BMI (Indeks Massa Tubuh)", 10.0, 60.0, 22.0)
            
        with col2:
            tidur = st.slider("Durasi Tidur (Jam)", 3, 12, 7)
            stres = st.selectbox("Tingkat Stres", ["Rendah", "Sedang", "Tinggi"])
            aktivitas = st.selectbox("Aktivitas Fisik", ["Rendah", "Sedang", "Tinggi"])
            merokok = st.radio("Riwayat Merokok", ["Tidak", "Ya"], horizontal=True)

    st.markdown("---")
    
    if st.button("JALANKAN DIAGNOSIS", use_container_width=True):
        # Konversi input user
        map_lvl = {"Rendah": 0, "Sedang": 1, "Tinggi": 2}
        map_bin = {"Tidak": 0, "Ya": 1}

        # Susun data sesuai urutan fitur model
        data_input = pd.DataFrame([[
            usia, gula, map_lvl[aktivitas], tidur, map_lvl[stres], map_bin[merokok], bmi
        ]], columns=['Usia', 'Gula', 'Aktivitas', 'Tidur', 'Stres', 'Merokok', 'BMI'])

        # Prediksi
        prediksi = model.predict(data_input)
        prob = model.predict_proba(data_input)

        # Output Hasil
        st.subheader("Hasil Analisis Sistem:")
        
        if prediksi[0] == 1:
            st.error("### KESIMPULAN: BERISIKO DIABETES")
            st.write(f"Sistem mendeteksi pola risiko sebesar **{prob[0][1]*100:.1f}%**")
            st.info("Saran: Perhatikan asupan gula dan segera lakukan cek laboratorium.")
        else:
            st.success("### KESIMPULAN: RISIKO RENDAH (AMAN)")
            st.write(f"Tingkat keyakinan sistem: **{prob[0][0]*100:.1f}%**")
            st.info("Saran: Pertahankan pola hidup sehat dan olahraga teratur.")

if __name__ == "__main__":
    main()