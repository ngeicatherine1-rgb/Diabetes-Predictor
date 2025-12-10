# Data Directory

## Folder Structure

### raw/
Contains the original, unmodified datasets. 

**Dataset:** Pima Indians Diabetes Dataset
- **Source:** UCI Machine Learning Repository
- **Samples:** 768 records
- **Features:** 8 medical measurements
- **Target:** Binary classification (diabetes: yes/no)

Download from: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

### processed/
Contains cleaned and preprocessed datasets ready for modeling.

**Preprocessing steps:**
1. Handle missing values (zero imputation or deletion)
2. Feature scaling/normalization (StandardScaler)
3. Feature engineering (if applicable)
4. Train/test split (typically 80/20)

## Feature Descriptions

| Feature | Description | Unit |
|---------|-------------|------|
| Pregnancies | Number of times pregnant | Count |
| Glucose | Plasma glucose concentration | mg/dL |
| BloodPressure | Diastolic blood pressure | mmHg |
| SkinThickness | Triceps skin fold thickness | mm |
| Insulin | 2-hour serum insulin level | mu U/ml |
| BMI | Body mass index | kg/m² |
| DiabetesPedigreeFunction | Genetic predisposition score | - |
| Age | Age of the patient | years |
| Outcome | Diabetes diagnosis | 0=No, 1=Yes |

## Data Loading

Use `src/data_loader.py` to load and preprocess data:

```python
from src.data_loader import load_and_preprocess_data

X_train, X_test, y_train, y_test = load_and_preprocess_data('data/raw/diabetes.csv')
```

## Notes

- Handle missing/zero values appropriately (medical context matters)
- Check for class imbalance in target variable
- Ensure data privacy and HIPAA compliance if using real patient data
