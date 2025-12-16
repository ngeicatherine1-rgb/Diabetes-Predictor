# Toolkit Documentation

This document provides comprehensive information about the tools, libraries, and utilities used in the Diabetes Predictor project.

## Table of Contents

1. [Core Libraries](#core-libraries)
2. [Development Tools](#development-tools)
3. [Testing Framework](#testing-framework)
4. [Deployment Tools](#deployment-tools)
5. [Code Quality Tools](#code-quality-tools)
6. [Visualization Libraries](#visualization-libraries)
7. [Configuration Management](#configuration-management)

## Core Libraries

### Machine Learning
- **scikit-learn** (v1.3.2): Primary ML library for models, preprocessing, and evaluation
  - **Models**:
    - `LogisticRegression`: Linear classifier with L-BFGS solver, max_iter=1000, balanced class weights
    - `RandomForestClassifier`: Ensemble of 100 trees, max_depth=10, balanced class weights
  - **Preprocessing**:
    - `StandardScaler`: Feature normalization (mean=0, std=1) - saved to `models/scaler.pkl`
    - `train_test_split`: 80/20 split with stratification for balanced classes
  - **Metrics**: accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
  - **Feature Set**: 8 medical features (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)

### Data Processing
- **pandas**: Data manipulation and analysis
  - DataFrames for structured data handling
  - Data cleaning and preprocessing operations
  
- **numpy**: Numerical computing
  - Array operations and mathematical functions
  - Integration with scikit-learn

## Development Tools

### Web Framework
- **Flask** (v3.0.0): Lightweight web framework for the prediction API
  - **Routes**:
    - `/` - Home page with prediction form (`app/templates/index.html`)
    - `/predict` - POST endpoint for diabetes predictions (JSON input/output)
    - `/health` - Health check endpoint for monitoring
  - Template rendering with Jinja2
  - Configuration via `configs/app_config.yaml`
  - Model loading on startup for fast predictions
- **gunicorn** (v21.2.0): Production WSGI server for Flask deployment
- **python-dotenv** (v1.0.0): Environment variable management

### Interactive Development
- **Jupyter Notebook**: Interactive development environment
  - Exploratory data analysis
  - Model experimentation
  - Visualization

## Testing Framework

### pytest
- Unit testing framework
- Test discovery and execution
- Fixtures for test setup
- Coverage reporting

**Usage:**
```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

## Deployment Tools

### Docker
- **Dockerfile**: Containerization for consistent deployment
- **docker-compose.yml**: Multi-container orchestration

### Heroku
- **Procfile**: Process definition for Heroku deployment
- **runtime.txt**: Python version specification

## Code Quality Tools

### Formatting
- **black**: Code formatter (PEP 8 compliant)
- **autopep8**: Additional formatting support

### Linting
- **flake8**: Style guide enforcement
- **pylint**: Static code analysis

**Usage:**
```bash
black src/
flake8 src/
pylint src/
```

## Visualization Libraries

### matplotlib
- Static plotting and visualization
- Model performance charts
- Data distribution plots

### seaborn
- Statistical data visualization
- Enhanced matplotlib functionality
- Correlation heatmaps

### plotly
- Interactive visualizations
- Web-based charts
- Dashboard creation

## Configuration Management

### YAML Configuration
- **PyYAML**: YAML file parsing (via `yaml.safe_load()`)
- Configuration files in `configs/`:
  - `model_config.yaml`: Model hyperparameters
    - Logistic Regression: max_iter, solver, class_weight
    - Random Forest: n_estimators, max_depth, min_samples_split, min_samples_leaf
  - `data_config.yaml`: Data paths and preprocessing settings
  - `app_config.yaml`: Flask application settings (host, port, debug, model paths)

## Project Structure Tools

### Path Management
- **pathlib**: Modern path handling (Python 3.4+)
- Cross-platform path operations
- Used in `src/train_model.py` for creating model directories

### Logging
- **logging**: Built-in Python logging module
- Structured log files in `logs/` directory
- Configured in all modules (`src/`, `app/`) for debugging and monitoring
- Log levels: INFO for operations, ERROR for exceptions

## Model Persistence

### joblib
- Model serialization and deserialization (preferred over pickle for scikit-learn models)
- Saving trained models to `models/` directory:
  - `models/random_forest.pkl` - Random Forest classifier
  - `models/logistic_reg.pkl` - Logistic Regression classifier
  - `models/scaler.pkl` - StandardScaler for feature normalization
- Loading models for predictions via `src/train_model.py` and `src/predict.py`
- More efficient than pickle for numpy arrays used in scikit-learn

## Command-Line Interface

### argparse
- CLI argument parsing in `app/cli.py`
- User-friendly command-line interface for predictions
- Input validation for 8 feature values
- **Usage**: `python app/cli.py --input "6,148,72,35,0,33.6,0.627,50"`
- Supports custom model and scaler paths via `--model` and `--scaler` flags

## Version Control

### Git
- Version control and collaboration
- Branch management
- Commit history tracking

## Package Management

### pip
- Python package installer
- Dependency management via `requirements.txt`
- Virtual environment support

### setuptools
- Package distribution
- `setup.py` for package installation
- `pyproject.toml` for modern Python packaging

## Environment Management

### venv
- Virtual environment creation
- Dependency isolation
- Cross-platform support

## Utilities

### Scripts
- `scripts/check_env.ps1`: Environment verification (PowerShell) - checks Python version, dependencies, and project structure
- `deploy.sh`: Deployment automation script for production deployment

### Data Processing Pipeline
- **Data Loading**: `src/data_loader.py`
  - `load_data()`: Load CSV from `data/raw/diabetes.csv`
  - `handle_missing_values()`: Fill missing values using mean/median/drop methods
  - `preprocess_data()`: Complete preprocessing pipeline
  - `load_and_preprocess_data()`: Combined load and preprocess function
- **Dataset**: Pima Indians Diabetes Dataset (768 samples, 8 features, binary classification)

## Recommended IDE/Editor

- **Visual Studio Code**: Recommended code editor
  - Python extension
  - Git integration
  - Debugging support
  - Jupyter notebook support

## Diabetes Predictor Specific Details

### Dataset Features
The model uses 8 medical features to predict diabetes:
1. **Pregnancies**: Number of times pregnant
2. **Glucose**: Plasma glucose concentration (mg/dL)
3. **BloodPressure**: Diastolic blood pressure (mm Hg)
4. **SkinThickness**: Triceps skin fold thickness (mm)
5. **Insulin**: 2-Hour serum insulin (mu U/ml)
6. **BMI**: Body mass index (kg/m²)
7. **DiabetesPedigreeFunction**: Diabetes pedigree function
8. **Age**: Age in years

### Model Performance
- **Random Forest**: ~81.8% accuracy (primary model)
- **Logistic Regression**: ~79.2% accuracy (baseline model)
- Models saved in `models/` directory after training
- Feature importance: Glucose (28.5%), BMI (19.2%), Age (15.7%) are top predictors

### Prediction Workflow
1. Load trained model and scaler from `models/` directory
2. Accept 8 feature values (list, dict, or array)
3. Scale features using saved StandardScaler
4. Make prediction (0 = No Diabetes, 1 = Diabetes)
5. Return probability scores and confidence percentage

## Additional Resources

### Documentation
- Project README: `README.md` - Complete setup and usage guide
- Deployment Guide: `DEPLOYMENT_GUIDE.md` - Production deployment instructions
- Quick Start: `DEPLOY_QUICK_START.md` - Fast deployment guide
- Code Summary: `CODE_SUMMARY.md` - Project structure overview
- Working Examples: `WORKING_CODE_EXAMPLES.md` - Code samples and outputs

### External Resources
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [pandas Documentation](https://pandas.pydata.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [pytest Documentation](https://docs.pytest.org/)

## Tool Versions

Refer to `requirements.txt` for specific version numbers of all dependencies.

## Installation

All tools can be installed via:
```bash
pip install -r requirements.txt
```

## Troubleshooting

### Common Issues
1. **Version Conflicts**: Use virtual environment to isolate dependencies
2. **Missing Dependencies**: Run `pip install -r requirements.txt`
3. **Import Errors**: Verify virtual environment is activated
4. **Tool Not Found**: Check PATH and installation

## Contributing

When adding new tools:
1. Update this documentation
2. Add to `requirements.txt` if applicable
3. Update installation instructions
4. Document usage examples

