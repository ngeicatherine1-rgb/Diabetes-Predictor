import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

Path('data/raw').mkdir(parents=True, exist_ok=True)
Path('models').mkdir(parents=True, exist_ok=True)

if not (Path('models/random_forest.pkl').exists() and Path('models/scaler.pkl').exists()):
    np.random.seed(42)
    n_samples = 768
    df = pd.DataFrame(
        {
            'Pregnancies': np.random.randint(0, 17, n_samples),
            'Glucose': np.random.randint(44, 200, n_samples),
            'BloodPressure': np.random.randint(24, 122, n_samples),
            'SkinThickness': np.random.randint(0, 99, n_samples),
            'Insulin': np.random.randint(0, 846, n_samples),
            'BMI': np.random.uniform(18.2, 67.1, n_samples),
            'DiabetesPedigreeFunction': np.random.uniform(0.078, 2.42, n_samples),
            'Age': np.random.randint(21, 81, n_samples),
        }
    )
    df['Outcome'] = ((df['Glucose'] > 125) & (df['BMI'] > 30)).astype(int)
    noise_idx = np.random.choice(len(df), 150, replace=False)
    df.loc[noise_idx, 'Outcome'] = 1 - df.loc[noise_idx, 'Outcome']
    df.to_csv('data/raw/diabetes.csv', index=False)

    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    scaler.fit(X_train)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(scaler.transform(X_train), y_train)

    joblib.dump(model, 'models/random_forest.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

import app.web_app as app_module

client = app_module.app.test_client()
response = client.get('/health')
print('status', response.status_code)
print(json.dumps(response.get_json()))
assert response.status_code == 200
assert response.get_json()['status'] == 'healthy'
