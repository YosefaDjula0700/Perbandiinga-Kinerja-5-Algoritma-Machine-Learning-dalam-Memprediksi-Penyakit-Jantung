import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import joblib
import os

# =========================
# Fungsi: Set Background
# =========================
def set_background(png_file_path):
    if os.path.exists(png_file_path):
        with open(png_file_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded}");
                background-size: cover;
                background-attachment: fixed;
                background-position: center;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

set_background("Biru Modern Klinik Kesehatan Zoom Virtual Background.png")

# =========================
# Fungsi: Load Model dan Metrik
# =========================
def load_model_and_metrics(model_name):
    model_file = f"{model_name}_model.pkl"
    metric_file = f"{model_name}_metrics.pkl"
    model, feature_order = joblib.load(model_file)
    metrics = joblib.load(metric_file)
    return model, feature_order, metrics

# =========================
# UI Header
# =========================
st.markdown("<h1 style='text-align: center; color: crimson;'>❤️ Deteksi Risiko Penyakit Jantung</h1>", unsafe_allow_html=True)
st.markdown("### Powered by Machine Learning")

st.sidebar.header("⚙️ Pengaturan")
model_option = st.sidebar.selectbox("Pilih Model", [
    "Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "LightGBM"
])

# =========================
# Form Input
# =========================
with st.form("heart_form"):
    st.subheader("Masukkan Data Pasien")
    nama = st.text_input("Nama Pasien")

    col1, col2 = st.columns(2)
    with col1:
        age_cat = st.selectbox("Kategori Umur", ["18-24", "25-29", "30-34", "35-39", "40-44",
                                                  "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"])
        gen_health = st.selectbox("Kesehatan Umum", ["Excellent", "Very good", "Good", "Fair", "Poor"])
        sex = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        stroke = st.selectbox("Pernah Stroke?", ["Yes", "No"])
        diabetic = st.selectbox("Diabetes?", ["Yes", "No"])
        kidney = st.selectbox("Penyakit Ginjal?", ["Yes", "No"])
        physical = st.slider("Hari Kesehatan Fisik Buruk", 0, 30)

    with col2:
        smoking = st.selectbox("Merokok?", ["Yes", "No"])
        alcohol = st.selectbox("Minum Alkohol?", ["Yes", "No"])
        diff_walk = st.selectbox("Kesulitan Berjalan?", ["Yes", "No"])
        phys_act = st.selectbox("Aktivitas Fisik?", ["Yes", "No"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1)
        sleep_time = st.number_input("Jam Tidur", min_value=1, max_value=24)
        mental = st.slider("Hari Kesehatan Mental Buruk", 0, 30)

    submitted = st.form_submit_button("Submit")

# =========================
# Fungsi: Encode Input
# =========================
def encode_input():
    map_yesno = {"Yes": 1, "No": 0}
    map_gender = {"Male": 1, "Female": 0}
    map_genhealth = {"Excellent": 5, "Very good": 4, "Good": 3, "Fair": 2, "Poor": 1}
    map_age = {"18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5,
               "45-49": 6, "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10,
               "70-74": 11, "75-79": 12, "80 or older": 13}

    return pd.DataFrame([{ 
        "AgeCategory": map_age[age_cat],
        "GenHealth": map_genhealth[gen_health],
        "Sex": map_gender[sex],
        "Stroke": map_yesno[stroke],
        "Diabetic": map_yesno[diabetic],
        "Smoking": map_yesno[smoking],
        "AlcoholDrinking": map_yesno[alcohol],
        "DiffWalking": map_yesno[diff_walk],
        "PhysicalActivity": map_yesno[phys_act],
        "KidneyDisease": map_yesno[kidney],
        "BMI": bmi,
        "SleepTime": sleep_time,
        "PhysicalHealth": physical,
        "MentalHealth": mental,
        "Asthma": 0,
        "Race": 1,
        "SkinCancer": 0
    }])

# =========================
# Prediksi & Visualisasi
# =========================
if submitted:
    if not nama:
        st.warning("⚠️ Nama pasien tidak boleh kosong.")
    else:
        X_input = encode_input()
        try:
            model, feature_order, metrics = load_model_and_metrics(model_option)
            X_input = X_input[feature_order]
            prediction = model.predict(X_input)[0]
            probability = model.predict_proba(X_input)[0][1]

            status = "Berisiko" if prediction == 1 else "Tidak Berisiko"
            st.markdown(f"<div style='background-color:#f8f9fa; padding:10px; border-radius:10px'>"
                        f"<strong>Nama Pasien:</strong> {nama}<br>"
                        f"<strong>Status:</strong> <span style='color:{'red' if prediction==1 else 'green'}'>{status}</span><br>"
                        f"<strong>Probabilitas:</strong> {probability:.2%}</div>", unsafe_allow_html=True)

            # Menampilkan metrik
            st.markdown("---")
            st.markdown(f"### ✅ Evaluasi Model {model_option}")
            for met in ["accuracy", "f1", "roc_auc", "precision", "recall"]:
                if met in metrics:
                    st.write(f"{met.upper()}: {metrics[met]:.2%}")

            # Fitur paling memengaruhi
            if hasattr(model, "feature_importances_"):
                top_feature_idx = np.argmax(model.feature_importances_)
                top_feature = feature_order[top_feature_idx]
                st.success(f"Fitur paling memengaruhi prediksi: `{top_feature}`")

            fig, ax = plt.subplots()
            ax.pie([probability, 1 - probability], labels=["Positif", "Negatif"], autopct='%1.1f%%',
                   colors=["salmon", "lightgreen"], startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

            with st.expander("📋 Data Pasien yang Dientri"):
                st.dataframe(X_input)

            if prediction == 0:
                st.balloons()
            else:
                st.error("⚠️ Risiko tinggi terdeteksi. Segera konsultasikan dengan dokter.")
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
