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

## System Requirements

### Operating System
- **Windows**: 10 or later (64-bit recommended)
- **macOS**: 10.14 or later (Intel or Apple Silicon)
- **Linux**: Ubuntu 18.04+, Debian 10+, CentOS 7+, or equivalent

### Hardware Requirements
- **Minimum RAM**: 4 GB (8 GB recommended)
- **Disk Space**: 2 GB for project and dependencies
- **Processor**: Dual-core processor or better

### Software Prerequisites
| Requirement | Version | Purpose |
|---|---|---|
| **Python** | 3.8 or higher | Runtime environment |
| **pip** | 20.0+ | Package manager |
| **Git** | 2.0+ | Version control |
| **Virtual Environment Tool** | Built-in (venv) | Isolated Python environment |

### Optional Tools
| Tool | Purpose |
|---|---|
| **Jupyter Notebook/Lab** | Interactive notebook development (included in requirements.txt) |
| **Visual Studio Code** | Recommended code editor |
| **MLflow** | Experiment tracking (optional) |
| **Weights & Biases** | Advanced experiment tracking (optional) |

---

## Installation & Setup Instructions

### Step 1: Verify Python Installation

**Windows (PowerShell):**
```powershell
python --version
```

**macOS/Linux (Terminal):**
```bash
python3 --version
```

Expected output: `Python 3.8.0` or higher

If Python is not installed:
- **Windows**: Download from [python.org](https://www.python.org/downloads/) and ensure "Add Python to PATH" is checked
- **macOS**: `brew install python` or download from [python.org](https://www.python.org/downloads/)
- **Linux (Ubuntu/Debian)**: `sudo apt install -y python3 python3-pip`
- **Linux (CentOS/RHEL)**: `sudo yum install -y python3 python3-pip`

### Step 2: Install Git

**Windows:**
Download and install from [git-scm.com](https://git-scm.com/download/win)

**macOS:**
```bash
brew install git
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt install -y git
```

**Verify Installation:**
```bash
git --version
```

### Step 3: Clone the Repository

```bash
git clone https://github.com/ngeicatherine1-rgb/Diabetes-Predictor.git
cd Diabetes-Predictor
```

### Step 4: Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> **Troubleshooting**: If you get an execution policy error:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> .\.venv\Scripts\Activate.ps1
> ```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS/Linux (Bash/Zsh):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

✅ **Success**: Your shell prompt should show `(.venv)` at the beginning

### Step 5: Upgrade pip and Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**What gets installed:**
- scikit-learn (machine learning models)
- pandas (data manipulation)
- numpy (numerical computing)
- matplotlib & seaborn (visualization)
- plotly (interactive charts)
- flask (web server)
- jupyter (interactive notebooks)
- pytest (testing framework)
- black, flake8, pylint (code quality tools)

### Step 6: Download the Dataset

**Option A: Manual Download**
1. Download from [UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
2. Save the CSV file to `data/raw/` folder
3. Rename it to `diabetes.csv` (or update `configs/data_config.yaml`)

**Option B: Programmatic Download (Recommended)**

Create `download_dataset.py` in project root:
```python
import pandas as pd
from pathlib import Path

# Create directory if it doesn't exist
Path('data/raw').mkdir(parents=True, exist_ok=True)

# Load dataset (example using public CSV)
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=column_names)
df.to_csv('data/raw/diabetes.csv', index=False)
print("Dataset downloaded successfully!")
```

Then run:
```bash
python download_dataset.py
```

### Step 7: Verify Installation

Run tests to confirm everything is set up correctly:

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_data_loader.py::test_handle_missing_values_mean PASSED
tests/test_data_loader.py::test_handle_missing_values_drop PASSED
tests/test_train_model.py::test_train_random_forest PASSED
tests/test_predict.py::test_make_prediction PASSED
```

---

## Quick Start Guide

### Option 1: Jupyter Notebook Workflow (Recommended for Learning)

```bash
jupyter notebook
```

Then open these notebooks in order:
1. `notebooks/exploratory.ipynb` - Data exploration and analysis
2. `notebooks/preprocessing.ipynb` - Feature engineering and scaling
3. `notebooks/modeling.ipynb` - Model training and evaluation

### Option 2: CLI Prediction (After Training)

```bash
python app/cli.py --input "6,148,72,35,0,33.6,0.627,50"
```

**Input format**: `Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age`

### Option 3: Web Application

```bash
python app/web_app.py
```

Then visit `http://localhost:5000` in your browser

### Step 8: Train Models (if using scripts)

```bash
# Train all models with default config
python src/train_model.py

# Train specific model
python src/train_model.py --model random_forest
python src/train_model.py --model logistic_regression
```

Models and scaler will be saved to `models/` directory

## Usage

### Making Predictions (CLI)

**Basic usage:**
```bash
python app/cli.py --input "6,148,72,35,0,33.6,0.627,50"
```

**With custom model:**
```bash
python app/cli.py --input "6,148,72,35,0,33.6,0.627,50" --model models/logistic_reg.pkl --scaler models/scaler.pkl
```

**Example output:**
```
==================================================
DIABETES PREDICTION RESULTS
==================================================
Input: Pregnancies=6, Glucose=148, BloodPressure=72, ...
--------------------------------------------------
Prediction: POSITIVE (Diabetes)
Confidence: 82.45%
Diabetes Probability: 82.45%
No Diabetes Probability: 17.55%
==================================================
```

### Running Web App (Flask)

```bash
python app/web_app.py
```

Features:
- 🎨 Beautiful UI for entering patient metrics
- 📊 Real-time prediction results
- 📈 Probability visualization with progress bars
- 📱 Mobile-friendly responsive design

Access at: `http://localhost:5000`

### Training Models

**Using scripts:**
```bash
# Train Random Forest model
python src/train_model.py

# Custom training
python src/train_model.py --model random_forest
```

**Using notebooks (interactive):**
```bash
jupyter notebook notebooks/modeling.ipynb
```

Models are saved to `models/` directory:
- `random_forest.pkl` - Random Forest classifier
- `logistic_reg.pkl` - Logistic Regression classifier
- `scaler.pkl` - StandardScaler for feature normalization

### Running Tests

**All tests:**
```bash
pytest tests/ -v
```

**Specific test file:**
```bash
pytest tests/test_data_loader.py -v
```

**With coverage report:**
```bash
pytest tests/ --cov=src --cov-report=html
```

Open `htmlcov/index.html` to view coverage report

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

## Environment Management

### Deactivating Virtual Environment
```bash
deactivate
```

### Freezing Dependencies (for reproducibility)
```bash
pip freeze > requirements-lock.txt
```

### Upgrading Packages Safely
```bash
pip install --upgrade -r requirements.txt
```

### Removing Virtual Environment
**macOS/Linux:**
```bash
rm -rf .venv
```

**Windows (PowerShell):**
```powershell
Remove-Item -Recurse -Force .venv
```

### Cleaning Project Cache Files
```bash
# Remove Python cache
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# Remove pytest cache
rm -rf .pytest_cache

# Remove coverage reports
rm -rf htmlcov .coverage
```

---

## Testing

Run tests using pytest:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Troubleshooting

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| `python: command not found` | Python not installed or not in PATH | Install Python from [python.org](https://www.python.org/downloads/) and ensure PATH is set |
| `Activate.ps1 cannot be loaded` | PowerShell execution policy | Run: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `ModuleNotFoundError: No module named 'sklearn'` | Dependencies not installed | Run: `pip install -r requirements.txt` |
| `Port 5000 already in use` | Another app using the port | Change port in `app/web_app.py`: `app.run(port=5001)` |
| `No such file or directory: 'data/raw/diabetes.csv'` | Dataset not downloaded | Download dataset using Step 6 above |
| `models/random_forest.pkl not found` | Models not trained yet | Run: `python src/train_model.py` |
| `Git push fails` | Untracked files or branch mismatch | Run: `git add .` then `git commit -m "message"` then `git push` |

### Virtual Environment Not Activating

**Windows (PowerShell):**
```powershell
# Check execution policy
Get-ExecutionPolicy -Scope CurrentUser

# If Restricted, run:
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Check if venv is activated:**
```bash
# Should show path to .venv/python
which python  # macOS/Linux
where python  # Windows
```

---

## Configuration

Edit configuration files in the `configs/` directory:
- `model_config.yaml` - Model hyperparameters (adjust for better accuracy)
- `data_config.yaml` - Data paths and preprocessing settings
- `app_config.yaml` - Application settings (host, port, debug mode)

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
- Inspired by ML best practices and production-ready project standards
