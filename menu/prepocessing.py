import streamlit as st
import pandas as pd
import numpy as np

def show_preprocessing_tab():
    st.title("Preprocessing Data")

    # LOAD DATA
    file_id = '1RwqEU4qi8EceAoOlfx2RYjBK6DkL-Snf'
    data = f'https://drive.google.com/uc?export=download&id={file_id}'
    df = pd.read_csv(data)

    # Data
    st.subheader("1️⃣ Data Asli")
    df = df[['JK','Umur','IMT','Lingkar Perut','Sistolik',
             'Diastolik','Hba1c','GDPuasa','GD2PP','DX']]
    st.dataframe(df.head())

    st.subheader("2️⃣ Cek Missing Value")
    st.write("Jumlah missing value tiap kolom:")
    st.dataframe(df.isnull().sum())

    # PENANGANAN MISSING VALUE
    st.subheader("3️⃣  Penanganan Missing Value (Drop NA)")
    df = df.dropna()

    st.write("Setelah penanganan missing value:")
    st.dataframe(df.isnull().sum())
    st.write("Jumlah data setelah dropna:", len(df))

    # ENCODING
    st.subheader("4️⃣ Encoding Variabel Kategorikal")

    df['JK'] = df['JK'].map({'L': 1, 'P': 0})
    df['DX'] = df['DX'].astype(str).str.strip().str.upper().map({'DM': 1, 'NON DM': 0})

    num_cols = ['Umur','IMT','Lingkar Perut','Sistolik','Diastolik',
                'Hba1c','GDPuasa','GD2PP']

    df[num_cols] = (
        df[num_cols]
        .astype(str)
        .replace(',', '.', regex=True)
        .apply(lambda x: x.str.strip())
        .astype(float)
    )

    st.write("Data setelah encoding:")
    st.dataframe(df.head())

    # NORMALISASI
    st.subheader("5️⃣ Normalisasi MinMax")

    from sklearn.preprocessing import MinMaxScaler

    X = df[['JK','Umur','IMT','Lingkar Perut','Sistolik',
            'Diastolik','Hba1c','GDPuasa','GD2PP']]
    y = df['DX']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    st.write("Data hasil normalisasi :")
    st.dataframe(pd.DataFrame(X_scaled, columns=X.columns).head())

    # SPLIT DATA
    st.subheader("6️⃣ Split Data Train-Test")

    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    st.write("Jumlah Data Setelah Split:")
    st.write(f"Data Train: {len(X_train)}")
    st.write(f"Data Test : {len(X_test)}")

    col1, col4 = st.columns(2)

    with col1:
        fig3, ax3 = plt.subplots(figsize=(3,3))
        pd.Series(y_train).value_counts().sort_index().plot(kind='bar', ax=ax3)
        ax3.set_title("Distribusi Kelas Data Train")
        st.pyplot(fig3)

    with col4:
        fig4, ax4 = plt.subplots(figsize=(3,3))
        pd.Series(y_test).value_counts().sort_index().plot(kind='bar', ax=ax4)
        ax4.set_title("Distribusi Kelas Data Test")
        st.pyplot(fig4)

    # SMOTE
    st.subheader("7️⃣ Penanganan Imbalance dengan SMOTE (Data Train)")

    from imblearn.over_sampling import SMOTE

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    sebelum = pd.Series(y_train).value_counts().sort_index()
    sesudah = pd.Series(y_train_res).value_counts().sort_index()

    st.write("Distribusi kelas sebelum SMOTE:", sebelum.to_dict())
    st.write("Distribusi kelas setelah SMOTE:", sesudah.to_dict())

    col3, col2 = st.columns(2)

    with col3:
        fig3, ax3 = plt.subplots(figsize=(3,2.5))
        sebelum.plot(kind='bar', ax=ax3)
        ax3.set_title("Sebelum SMOTE")
        st.pyplot(fig3)

    with col2:
        fig4, ax4 = plt.subplots(figsize=(3,2.5))
        sesudah.plot(kind='bar', ax=ax4)
        ax4.set_title("Setelah SMOTE")
        st.pyplot(fig4)