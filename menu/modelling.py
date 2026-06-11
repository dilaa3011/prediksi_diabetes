import streamlit as st
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

from utils import load_data, clean_data, fknn_predict


def modelling_tab():

    # DATA
    st.subheader("Data Hasil Preprocessing")

    df = load_data()
    df = clean_data(df)

    st.dataframe(df.head())

    # EVALUASI
    st.subheader("Evaluasi Model FKNN")

    if st.button("TRAIN MODEL"):

        progress = st.progress(0)

        start_total = time.perf_counter()

        # LOAD DATA
        df = load_data()
        df = clean_data(df)

        X = df[['JK','Umur','IMT','Lingkar Perut',
                'Sistolik','Diastolik',
                'Hba1c','GDPuasa','GD2PP']]

        y = df['DX']

        # NORMALISASI
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # SPLIT DATA
        X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

        progress.progress(20)

        # SMOTE
        smote = SMOTE(random_state=42)

        X_train_res, y_train_res = smote.fit_resample(
            X_train,
            y_train
        )

        progress.progress(40)

        # PARAMETER MODEL
        k =7
        p = 1   # Manhattan Distance

        # TRAINING + PREDIKSI
        start_model = time.perf_counter()

        pred = fknn_predict(
            X_train_res,
            y_train_res,
            X_test,
            k=k,
            p=p
        )

        end_model = time.perf_counter()

        model_time = end_model - start_model

        progress.progress(70)

        # EVALUASI
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred)
        recall = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)

        end_total = time.perf_counter()
        total_time = end_total - start_total

        progress.progress(100)

        st.success("Training selesai!")

        # HASIL MODEL
        st.subheader("Hasil Evaluasi Model")

        st.success(f"""
        Metode Jarak  : Manhattan
        Nilai K       : 7
        Accuracy      : {round(accuracy,4)}
        Precision     : {round(precision,4)}
        Recall        : {round(recall,4)}
        F1-Score      : {round(f1,4)}
        Waktu Model   : {round(model_time,4)} detik
        Total Waktu   : {round(total_time,4)} detik
        """)

        # CONFUSION MATRIX
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, pred)

        fig, ax = plt.subplots(figsize=(5,4))

        im = ax.imshow(cm)

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j],
                        ha="center",
                        va="center")

        st.pyplot(fig)

        # TABEL METRIK
        st.subheader("Tabel Hasil Evaluasi")

        hasil = pd.DataFrame({
            "Metode": ["FKNN"],
            "Jarak": ["Manhattan"],
            "K": [7],
            "Accuracy": [round(accuracy,4)],
            "Precision": [round(precision,4)],
            "Recall": [round(recall,4)],
            "F1-Score": [round(f1,4)],
            "Waktu (detik)": [round(model_time,4)]
        })

        st.dataframe(hasil, use_container_width=True)

        # ANALISIS
        st.subheader("Analisis Hasil")

        st.markdown(f"""
        Model Fuzzy K-Nearest Neighbor (FKNN) menggunakan metode jarak Manhattan dengan nilai K = 7 menghasilkan:
        
        - Accuracy sebesar **{round(accuracy,4)}**
        - Precision sebesar **{round(precision,4)}**
        - Recall sebesar **{round(recall,4)}**
        - F1-Score sebesar **{round(f1,4)}**
        """)