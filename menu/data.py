import streamlit as st
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import zscore


def data_tab():
    st.title("Data Understanding")

    st.subheader("Data Asli")

    file_id = '1RwqEU4qi8EceAoOlfx2RYjBK6DkL-Snf'
    data = f'https://drive.google.com/uc?export=download&id={file_id}'
    df = pd.read_csv(data)
    df = df[['JK','Umur','IMT','Lingkar Perut','Sistolik',
            'Diastolik','Hba1c','GDPuasa','GD2PP','DX']]

    st.write("Jumlah Data:", len(df))
    st.dataframe(df.head())

    st.subheader("Tipe Data")
    dtype_df = pd.DataFrame({
        "Kolom": df.columns,
        "Tipe Data": df.dtypes.values
    })
    st.dataframe(dtype_df)

    st.write("Missing Value:")
    st.write(df.isnull().sum())

    df['JK'] = df['JK'].map({'L':1,'P':0})
    df['DX'] = df['DX'].astype(str).str.strip().str.upper().map({'DM':1,'NON DM':0})

    num_cols = ['Umur','IMT','Lingkar Perut','Sistolik','Diastolik',
                'Hba1c','GDPuasa','GD2PP']
    df[num_cols] = (
        df[num_cols]
        .astype(str)
        .replace(',', '.', regex=True)
        .apply(lambda x: x.str.strip())
        .astype(float)
    )

    st.title("Exploration Data Analyst")

    st.subheader("Persebaran Data Boxplot")
    # Boxplot
    fig_box, axes = plt.subplots(3, 3, figsize=(16, 14))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.boxplot(data=df, x="DX", y=col, ax=axes[i])
        axes[i].set_title(f"Boxplot {col} vs DX")
    # Remove empty subplot if num_cols < 9
    for j in range(len(num_cols), 9):
        fig_box.delaxes(axes[j])
    fig_box.tight_layout()
    st.pyplot(fig_box)



