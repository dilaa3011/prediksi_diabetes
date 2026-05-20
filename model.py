import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from utils import fknn_predict

# =========================
# LOAD DATA
# =========================
file_id = '1RwqEU4qi8EceAoOlfx2RYjBK6DkL-Snf'
data = f'https://drive.google.com/uc?export=download&id={file_id}'
df = pd.read_csv(data)

# =========================
# PREPROCESSING
# =========================
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

# =========================
# SCALING
# =========================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# SMOTE
# =========================
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# =========================
# GRID SEARCH FKNN
# =========================
K_VALUES = [3,5,7,9,11]
DISTANCE_PARAMS = {'euclidean':2,'manhattan':1,'minkowski':3}

best_acc = 0
best_k = None
best_p = None

for p_name, p_val in DISTANCE_PARAMS.items():
    for k in K_VALUES:

        pred = fknn_predict(
            X_train_res, y_train_res,
            X_test,
            k=k,
            p=p_val
        )

        acc = accuracy_score(y_test, pred)

        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_p = p_val

print(f"Model terbaik: K={best_k}, p={best_p}, Akurasi={best_acc}")

# =========================
# SAVE MODEL
# =========================
model_data = {
    "X_train": X_train_res,
    "y_train": y_train_res,
    "k": best_k,
    "p": best_p
}

os.makedirs("model", exist_ok=True)

joblib.dump(model_data, "model/best_fknn.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("✅ Model berhasil disimpan!")
