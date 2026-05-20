import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

from utils import fknn_predict, train_model
import menu.prepocessing as show_preprocessing_page
import menu.data as data_page
import menu.modelling as modelling

st.set_page_config(page_title="Aplikasi Klasifikasi Diabetes", layout="wide")

st.title("Aplikasi Klasifikasi Diabetes")


main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs(
    ["App", "Data Understanding", "Preprocessing", "Modelling"]
)

# SIDEBAR INPUT
st.sidebar.header("Input & Prediksi Data Baru")

with st.sidebar.form("form_prediksi"):

    jk = st.selectbox("Jenis Kelamin", ["L", "P"])
    umur = st.number_input("Umur", 1, 120)
    imt = st.number_input("IMT", format="%.2f")
    lp = st.number_input("Lingkar Perut", format="%.2f")
    sys = st.number_input("Sistolik", format="%.2f")
    dias = st.number_input("Diastolik", format="%.2f")
    hba1c = st.number_input("Hba1c", format="%.2f")
    gdp = st.number_input("GDP Puasa", format="%.2f")
    gd2pp = st.number_input("GD 2PP", format="%.2f")

    submit = st.form_submit_button("CEK DATA")


if submit:

    if not os.path.exists("model/best_fknn.pkl") or not os.path.exists("model/scaler.pkl"):
        st.sidebar.error("Model belum dilatih! Silakan Train Model dulu.")
    else:

        model = joblib.load("model/best_fknn.pkl")
        scaler = joblib.load("model/scaler.pkl")

        jk_numeric = 1 if jk == "L" else 0

        input_data = np.array([[jk_numeric, umur, imt, lp,
                                sys, dias, hba1c, gdp, gd2pp]])

        input_scaled = scaler.transform(input_data)

        pred = fknn_predict(
            model["X_train"],
            model["y_train"],
            input_scaled,
            k=model["k"],
            p=model["p"]
        )

        hasil = "DM" if pred[0] == 1 else "NON DM"

        data_input = {
            "JK": jk,
            "Umur": umur,
            "IMT": imt,
            "Lingkar Perut": lp,
            "Sistolik": sys,
            "Diastolik": dias,
            "Hba1c": hba1c,
            "GDPuasa": gdp,
            "GD2PP": gd2pp,
            "Hasil Prediksi": hasil
        }

        if "input_history" not in st.session_state:
            st.session_state["input_history"] = []

        st.session_state["input_history"].append(data_input)
        st.session_state["last_prediction"] = hasil

        st.rerun()

# APP
with main_tab1:

    st.subheader("Hasil Prediksi Terakhir")

    if "last_prediction" in st.session_state:
        hasil = st.session_state["last_prediction"]

        if hasil == "DM":
            st.error(f"⚠️ Hasil Prediksi: {hasil}")
        else:
            st.success(f"✅ Hasil Prediksi: {hasil}")
    else:
        st.info("Belum ada prediksi.")

    st.divider()

    st.subheader("Riwayat Input Data")

    if "input_history" in st.session_state:
        df_history = pd.DataFrame(st.session_state["input_history"])
        st.dataframe(df_history, use_container_width=True)
    else:
        st.info("Belum pernah input data.")


# DATA
with main_tab2:
    data_page.data_tab()


# PREPROCESSING
with main_tab3:
    show_preprocessing_page.show_preprocessing_tab()


# MODELLING
with main_tab4:
    modelling.modelling_tab()

