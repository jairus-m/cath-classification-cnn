# CATH Hierrarchy Data

Data is sourced from: https://github.com/wouterboomsma/cath_datasets?tab=readme-ov-file

### cath_3class_ca data:

The cath_3class.npz dataset is the simplest set. It considers the "class" level of 
the CATH hierarchy, which in the CATH database consists of "Mainly Alpha", "Mainly Beta", "Alpha Beta" and 
"Few secondary structures". Since the latter category is small, and structurally heterogeneous, we omit
it from our set. The three remaining categories are each reduced to have the same number of members 
(see filtering below), thus creating a balanced set. The three classes differ mainly in the relative quantities of 
alpha helices and beta strands (protein secondary structure). The main task in this set is thus to detect 
protein secondary structure in any orientation, and quantify the total amount of the different 
secondary structure elements in the entire image. The dataset contains only Carbon-alpha positions for 
each protein (i.e. only a single atom for each amino acid).

# Running Experiments

### 1. Activate the venv
- Run `uv venv` to initialize the venv
- Activate the venv by with the following CLI commands:
    - MacOS: Run `source .venv/bin/activate`
    - Windows Run `.venv\Scripts\activate.ps1` 
- Run `uv sync` to download the dependencies

### 2. Run the mlflow server
- Run `mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host 0.0.0.0 --port 5001`
- Open your localhost at the specified port to let mlflow track the runs

### 3. Run the Python module
- Run `python src/mlflow_experiments.py`
- The three models will be trained, evaluated, and logged
    1. NN - 1 Hidden Layer, 4 Neurons (<30s on my M1 Macbook Pro)
    2. NN - 2 Hidden Layers, 64 Neurons Each (<30s on my M1 Macbook Pro)
    3. CNN (25+ minutes on my M1 Macbook Pro)


# Modules
- protein_class_prediction_data.py
  - Info on protein data set
- mlflow_experiments.py
  - Experiment tracking module
