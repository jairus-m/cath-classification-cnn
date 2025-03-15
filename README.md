# CATH Hierrarchy Data
Data is sourced from: https://github.com/wouterboomsma/cath_datasets?tab=readme-ov-file

### cath_3class_ca data:
- [CATH database context and additional information on training data](https://github.com/jairus-m/cath_classification_cnn/blob/main/src/README.md)


# Running Experiments

NOTE: A Dockerfile is included to run the entire project within a container. Refer to the Dockerfile for build instructions.

### 1. Activate the venv (w/ uv)
- Run `uv venv` to initialize the venv
- Activate the venv by with the following CLI commands:
    - MacOS: Run `source .venv/bin/activate`
    - Windows Run `.venv\Scripts\activate.ps1` 
- Run `uv sync` to download the dependencies

### 2. Run the mlflow server
- Run `mlflow ui`
- Open your localhost at the specified port to let mlflow track the runs

### 3. Run the Python module
- Run `python -m src.main`
- Four models will be trained, evaluated, and logged (training time is based on an M1 Macbook Pro)
    1. NN - 1 Hidden Layer, 4 Neurons (~6s)
    2. NN - 2 Hidden Layers, 64 Neurons Each (~10s)
    3. CNN (<21min on my M1 Macbook Pro)
    4. Simplifed CNN w/ Early Stopping (<10min)

# Modules
- `src.models`
  - Different experimental deep learning models using the keras/mlflow API
- `src.utils` 
  - Interface for downloading/interacting with CATH protein dataset
  - Utilities for data pre- and post-processing  
- `src.main`
  - Main entry point for experiment tracking
 

# Project Dir Structure
```bash
cath-classification-cnn/
├── data/
│   ├── project_docs/
│   └── cath_3class_ca.npz           # raw CATH data
├── images/                          
│   ├── mlflow_results/
│   └── plotted_protein_results/
├── src/                             # main Python module                           
│   ├── models/
│   │   └── experimental_models.py   # Keras NNs/CNNs w/ mlflow API
│   ├── utils/                       
│   │   ├── cath_data.py             # util to interact with CATH data
│   │   └── process_data.py          # util to pre-process data/plot training curves
│   ├── main.py                      # entry point for running experiments
│   └── README.md                    # training data docs
├── Dockerfile                       # create image/run container for dev env 
├── pyproject.toml                   # project deps
└── uv.lock                          # uv lock file for venv management
```
