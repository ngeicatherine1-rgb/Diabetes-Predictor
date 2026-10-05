import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
ROOT_STR = str(ROOT)

import os
os.chdir(ROOT_STR)

model_path = ROOT / 'models' / 'random_forest.pkl'
scaler_path = ROOT / 'models' / 'scaler.pkl'

data_dir = ROOT / 'data' / 'raw'
data_dir.mkdir(parents=True, exist_ok=True)
(ROOT / 'models').mkdir(parents=True, exist_ok=True)

if not model_path.exists() or not scaler_path.exists():
    np.random.seed(42)
    n = 768
    df = pd.DataFrame({
        'Pregnancies': np.random.randint(0, 17, n),
        'Glucose': np.random.randint(44, 200, n),
        'BloodPressure': np.random.randint(24, 122, n),
        'SkinThickness': np.random.randint(0, 99, n),
        'Insulin': np.random.randint(0, 846, n),
        'BMI': np.random.uniform(18.2, 67.1, n),
        'DiabetesPedigreeFunction': np.random.uniform(0.078, 2.42, n),
        'Age': np.random.randint(21, 81, n),
    })
    df['Outcome'] = ((df['Glucose'] > 125) & (df['BMI'] > 30)).astype(int)
    noise_idx = np.random.choice(len(df), 150, replace=False)
    df.loc[noise_idx, 'Outcome'] = 1 - df.loc[noise_idx, 'Outcome']
    df.to_csv(data_dir / 'diabetes.csv', index=False)

    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    scaler.fit(X_train)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(scaler.transform(X_train), y_train)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

import app
client = app.app.test_client()
resp = client.get('/health')
print('status', resp.status_code)
print(json.dumps(resp.get_json()))
assert resp.status_code == 200
assert resp.get_json()['status'] == 'healthy'
