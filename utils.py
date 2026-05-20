import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

def load_data():
    file_id = '1IsnBFlnQfDsDs0OFZDIgx3KCfSsEScrW'

    data = f'https://drive.google.com/uc?export=download&id={file_id}'
    return pd.read_csv(data)

def fknn_predict(X_train, y_train, X_test, k, m=2, p=2):
    preds = []

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    classes = np.unique(y_train)

    N = len(y_train)
    U = np.zeros((N, len(classes)))

    for i, c in enumerate(classes):
        n_c = np.sum(y_train == c)
        for j in range(N):
            if y_train[j] == c:
                U[j, i] = 0.51
            else:
                U[j, i] = 0.49 * (n_c / N)

    for x in X_test:
        jarak = np.linalg.norm(X_train - x, axis=1, ord=p)
        idx = np.argsort(jarak)[:k]
        d_k = jarak[idx]
        d_k[d_k == 0] = 1e-6

        u = {}
        for i, c in enumerate(classes):
            num = np.sum(U[idx, i] / (d_k ** (2 / (m - 1))))
            den = np.sum(1 / (d_k ** (2 / (m - 1))))
            u[c] = num / den

        preds.append(max(u, key=u.get))

    return np.array(preds)

def clean_data(df):

    df = df[['JK','Umur','IMT','Lingkar Perut','Sistolik',
             'Diastolik','Hba1c','GDPuasa','GD2PP','DX']]

    df = df.dropna()

    df['JK'] = df['JK'].map({'L':1,'P':0})
    df['DX'] = df['DX'].astype(str).str.strip().str.upper().map({'DM':1,'NON DM':0})

    num_cols = ['Umur','IMT','Lingkar Perut','Sistolik',
                'Diastolik','Hba1c','GDPuasa','GD2PP']

    df[num_cols] = (
        df[num_cols]
        .astype(str)
        .replace(',', '.', regex=True)
        .apply(lambda x: x.str.strip())
        .astype(float)
    )

    return df

@st.cache_data
def train_model():

    df = load_data()

    df = df[['JK','Umur','IMT','Lingkar Perut','Sistolik',
             'Diastolik','Hba1c','GDPuasa','GD2PP','DX']]

    df = df.dropna()

    df['JK'] = df['JK'].map({'L':1,'P':0})
    df['DX'] = df['DX'].astype(str).str.strip().str.upper().map({'DM':1,'NON DM':0})

    num_cols = ['Umur','IMT','Lingkar Perut','Sistolik',
                'Diastolik','Hba1c','GDPuasa','GD2PP']

    df[num_cols] = (
        df[num_cols]
        .astype(str)
        .replace(',', '.', regex=True)
        .apply(lambda x: x.str.strip())
        .astype(float)
    )

    X = df[['JK','Umur','IMT','Lingkar Perut','Sistolik',
            'Diastolik','Hba1c','GDPuasa','GD2PP']]
    y = df['DX']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2,
        random_state=42,
        stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    K_VALUES = [3,5,7,9,11]
    DISTANCE_PARAMS = {'Euclidean':2,'Manhattan':1,'Minkowski':3}

    best_acc = 0
    best_k = None
    best_p = None
    best_distance = None
    results = []

    for distance_name, p_val in DISTANCE_PARAMS.items():
        for k in K_VALUES:

            pred = fknn_predict(
                X_train_res, y_train_res,
                X_test,
                k=k,
                p=p_val
            )

            acc = accuracy_score(y_test, pred)

            results.append({
                "Jarak": distance_name,
                "K": k,
                "Accuracy": acc
            })

            if acc > best_acc:
                best_acc = acc
                best_k = k
                best_p = p_val
                best_distance = distance_name

    pred_final = fknn_predict(
        X_train_res, y_train_res,
        X_test,
        k=best_k,
        p=best_p
    )

    cm = confusion_matrix(y_test, pred_final)

    model_data = {
        "X_train": X_train_res,
        "y_train": y_train_res,
        "k": best_k,
        "p": best_p,
        "distance_name": best_distance
    }

    os.makedirs("model", exist_ok=True)
    joblib.dump(model_data, "model/best_fknn.pkl")
    joblib.dump(scaler, "model/scaler.pkl")

    return best_acc, best_k, best_distance, cm, results