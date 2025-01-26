# CATH Hierrarchy Data
Data is sourced from: https://github.com/wouterboomsma/cath_datasets?tab=readme-ov-file

### cath_3class_ca data:
- [CATH database context and additional information on training data](https://github.com/jairus-m/cath_classification_cnn/blob/main/src/README.md)


# Running Experiments

### 1. Activate the venv (w/ uv)
- Run `uv venv` to initialize the venv
- Activate the venv by with the following CLI commands:
    - MacOS: Run `source .venv/bin/activate`
    - Windows Run `.venv\Scripts\activate.ps1` 
- Run `uv sync` to download the dependencies

### 2. Run the mlflow server
- Run `mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host 0.0.0.0 --port 5001`
- Open your localhost at the specified port to let mlflow track the runs

### 3. Run the Python module
- Run `python -m src.mlflow_experiments`
- The three models will be trained, evaluated, and logged
    1. NN - 1 Hidden Layer, 4 Neurons (<30s on my M1 Macbook Pro)
    2. NN - 2 Hidden Layers, 64 Neurons Each (<30s on my M1 Macbook Pro)
    3. CNN (25+ minutes on my M1 Macbook Pro)


# Modules
- protein_class_prediction_data.py
  - Info on protein data set
- mlflow_experiments.py
  - Experiment tracking module
