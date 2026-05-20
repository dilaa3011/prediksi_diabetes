import streamlit as st
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    roc_curve, auc
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
        st.subheader("Evaluasi Lengkap Model FKNN")

        if st.button("TRAIN MODEL"):

            progress = st.progress(0)
            start_total = time.perf_counter()

            df = load_data()
            df = clean_data(df)

            X = df[['JK','Umur','IMT','Lingkar Perut','Sistolik',
                    'Diastolik','Hba1c','GDPuasa','GD2PP']]
            y = df['DX']

            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2,
                random_state=42, stratify=y
            )

            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

            K_VALUES = [3,5,7,9,11]
            DISTANCE_PARAMS = {'Euclidean':2,'Manhattan':1,'Minkowski':3}

            results = []
            total_comb = len(K_VALUES) * len(DISTANCE_PARAMS)
            counter = 0

            best_acc = 0
            best_model = None

            for distance_name, p_val in DISTANCE_PARAMS.items():
                for k in K_VALUES:

                    start_time = time.perf_counter()

                    pred = fknn_predict(
                        X_train_res, y_train_res,
                        X_test,
                        k=k,
                        p=p_val
                    )

                    end_time = time.perf_counter()
                    comp_time = end_time - start_time

                    acc = accuracy_score(y_test, pred)

                    results.append({
                        "Jarak": distance_name,
                        "K": k,
                        "Accuracy": acc,
                        "Waktu (detik)": comp_time
                    })

                    if acc > best_acc:
                        best_acc = acc
                        best_model = (k, p_val, distance_name, pred)

                    counter += 1
                    progress.progress(counter / total_comb)

            end_total = time.perf_counter()
            total_time = end_total - start_total

            st.success("Training selesai!")

            st.write(f"⏱ Total Waktu Komputasi: {round(total_time,3)} detik")

            df_results = pd.DataFrame(results)

            best_row = df_results.loc[df_results["Accuracy"].idxmax()]

            st.subheader("Best Model")

            st.success(f"""
            Metode Jarak  : {best_row['Jarak']}
            Nilai K       : {best_row['K']}
            Akurasi       : {round(best_row['Accuracy'],4)}
            Waktu (detik) : {round(best_row['Waktu (detik)'],4)}
            """)

            # GRAFIK WAKTU KOMPUTASI
            st.subheader("Grafik Waktu Komputasi per Kombinasi")

            fig1, ax1 = plt.subplots(figsize=(10,5))

            for distance in df_results["Jarak"].unique():
                subset = df_results[df_results["Jarak"] == distance]
                ax1.plot(subset["K"], subset["Waktu (detik)"],
                        marker="o", label=distance)

            ax1.set_xlabel("Nilai K")
            ax1.set_ylabel("Waktu (detik)")
            ax1.legend()

            st.pyplot(fig1)

           
            st.subheader("Tabel Semua Kombinasi")

            df_results_sorted = df_results.sort_values(
                by="Accuracy", ascending=False
            ).reset_index(drop=True)

            st.dataframe(
                df_results_sorted[["Jarak","K","Accuracy","Waktu (detik)"]],
                use_container_width=True
            )

            # GRAFIK AKURASI
            st.subheader("Grafik Akurasi per Kombinasi")


            fig2, ax2 = plt.subplots(figsize=(10,5))

            for distance in df_results["Jarak"].unique():
                subset = df_results[df_results["Jarak"] == distance]
                ax2.plot(subset["K"], subset["Accuracy"],
                        marker="o", label=distance)

            ax2.set_xlabel("Nilai K")
            ax2.set_ylabel("Accuracy")
            ax2.legend()

            st.pyplot(fig2)

            st.markdown("""
            Dari grafik di atas, kita dapat melihat bagaimana waktu komputasi berubah dengan nilai K dan metode jarak yang berbeda. Berikut penjelasan singkat untuk setiap metode jarak:
            1. Metode Minkowski (Hijau)
                    - Memiliki waktu komputasi paling tinggi di semua nilai K.
                    - Paling lambat terjadi pada K = 3 (~0.20 detik).
                    - Seiring bertambahnya K, waktu cenderung menurun dan stabil.
                Hal ini karena Minkowski dengan parameter p > 2 memerlukan perhitungan pangkat yang lebih kompleks dibanding Euclidean dan Manhattan. Sehingga Minkowski lebih berat secara komputasi.
            2. Metode Euclidean (Biru)
                    - Tidak menunjukkan peningkatan signifikan saat K bertambah.
                    - Waktu komputasi relatif stabil di kisaran 0.03 – 0.06 detik.
                Perhitungan jarak Euclidean cukup efisien karena hanya menggunakan akar kuadrat dari jumlah kuadrat selisih. Sehingga jarak Euclidean cukup efisien dan stabil.
            3. Metode Manhattan (Oranye)
                    - Waktu awal cukup rendah (sekitar 0.03 detik). Namun meningkat cukup signifikan pada K = 11 (~0.07 detik).
                Manhattan menggunakan penjumlahan absolut yang relatif ringan, tetapi ketika K besar, proses agregasi membership lebih banyak. Sehingga Manhattan cepat pada K kecil, tapi cenderung meningkat saat K besar.                    
        """)

