"""
Jairus Martinez
Date: 1-25-25

This is a mlflow experiment that tracks the results of 3 different models:
1. NN - 1 Hidden Layer, 4 Neurons
2. NN - 2 Hidden Layers, 64 Neurons Each
3. CNN
4. Simplified CNN with Early Stopping 

I have an experimental Jupyter Notebook with more detail for the CNN implementation, the theory behind its approach,
and the inuition behind the different architectual choices. That code has extensive comments and markdown.

For this module, the goal was to spin up an mlflow experiment for tracking all 3 models, abstract the data pre-processing
for the two model types (NN/CNN), and then train, evaluate, and log the results in a reproducible way. Therefore the code here
is straight to the point and streamlines the original EDA/dev training.
"""

import logging
import mlflow
from mlflow.models.signature import infer_signature

from .utils import load_protein_data, preprocess_nn_data, preprocess_cnn_data
from .models import (
    nn_single__1,
    nn_double__2,
    cnn__3,
    cnn__4
)

logging.basicConfig(level=logging.INFO)

DATA_PATH = "../data/cath_3class_ca.npz"
IMAGE_PATH = "images/"

PROTEIN_DATA = load_protein_data(DATA_PATH)

if __name__ == "__main__":
    # nn-specific data pre-processed data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_nn_data(
        PROTEIN_DATA
    )
    signature = infer_signature(X_train, y_train)

    ### NNs

    # Experiment 1: Basic Neural Network with one Layer
    nn_single__1(X_train, X_val, X_test, y_train, y_val, y_test, signature)

    # Experiment 2: Basic Neural Network, 2 Hidden layers
    nn_double__2(X_train, X_val, X_test, y_train, y_val, y_test, signature)

    ### CNNs

    # cnn-specific data pre-processed data (overwrite nn vars)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_cnn_data(
        PROTEIN_DATA
    )

    # Experiment 3: CNN
    cnn__3(X_train, X_val, X_test, y_train, y_val, y_test, signature)

    # Experiment 4: Simplified CNN with Early Stopping
    cnn__4(X_train, X_val, X_test, y_train, y_val, y_test, signature)
