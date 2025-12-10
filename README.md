# Diabetes Predictor

A machine learning project to predict the likelihood of diabetes in patients using classification models.

## Project Overview

This project implements a diabetes prediction system using the Pima Indians Diabetes Dataset. It includes:
- Exploratory Data Analysis (EDA)
- Feature engineering and preprocessing
- Model training and comparison (Logistic Regression, Random Forest)
- Evaluation metrics and visualization
- CLI and web application for predictions

## Project Structure

```
diabetes-predictor/
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned and preprocessed data
│   └── README.md               # Dataset documentation
├── notebooks/
│   ├── exploratory.ipynb       # EDA notebook
│   ├── preprocessing.ipynb     # Feature engineering
│   └── modeling.ipynb          # Model training and experiments
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── train_model.py          # Training pipeline
│   ├── evaluate_model.py       # Evaluation metrics
│   └── predict.py              # Prediction functions
├── models/
│   ├── logistic_reg.pkl        # Trained logistic regression model
│   ├── random_forest.pkl       # Trained random forest model
│   └── scaler.pkl              # Fitted scaler
├── app/
│   ├── cli.py                  # Command-line interface
│   ├── web_app.py              # Flask/Streamlit web app
│   └── templates/              # HTML templates
├── configs/
│   ├── model_config.yaml       # Model hyperparameters
│   ├── data_config.yaml        # Data paths and settings
│   └── app_config.yaml         # App configuration
├── logs/                       # Training and application logs
├── experiment_tracking/        # MLflow/W&B experiment logs
├── tests/
│   ├── test_data_loader.py
│   ├── test_train_model.py
│   └── test_predict.py
├── .gitignore                  # Git ignore file
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── pyproject.toml              # Project metadata
└── LICENSE                     # Project license
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd diabetes-predictor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Models
```bash
python src/train_model.py --model logistic_regression
python src/train_model.py --model random_forest
```

### Making Predictions (CLI)
```bash
python app/cli.py --input "6,148,72,35,0,33.6,0.627,50"
```

### Running Web App
```bash
python app/web_app.py
```
Then visit `http://localhost:5000` in your browser.

## Dataset

The Pima Indians Diabetes Dataset contains medical measurements and diabetes outcomes for 768 female patients. Features include:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

[Dataset Source](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

## Model Performance

Results from trained models:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 78.5% | 0.78 | 0.65 | 0.71 |
| Random Forest | 81.2% | 0.80 | 0.72 | 0.76 |

## Testing

Run tests using pytest:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Configuration

Edit configuration files in the `configs/` directory:
- `model_config.yaml` - Model hyperparameters
- `data_config.yaml` - Data paths and preprocessing settings
- `app_config.yaml` - Application settings

## Logging

Logs are saved to the `logs/` directory with timestamps. Configure logging in your application.

## Experiment Tracking (Optional)

To use MLflow or Weights & Biases:
1. Uncomment relevant packages in `requirements.txt`
2. Initialize tracking in your training script
3. View results in the `experiment_tracking/` directory

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Follow PEP 8 style guide
4. Write tests for new features
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author
Catherine Ngei

## Acknowledgments

- Pima Indians Diabetes Dataset from UCI Machine Learning Repository
- scikit-learn documentation and community
# Diabetes-Predictor
# Diabetes-Predictor
# Diabetes-Predictor
