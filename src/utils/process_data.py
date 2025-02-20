""" 
This is a pre and post processing utility. It contains two functions that preps
data for a NN and CNN implementation. In addition, also contains 
a plotting function for results based off the keras History object.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import mlflow
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.src.callbacks.history import History


def plot_results(history: History, model_name: str, image_path: str):
    """
    Plots model accuracy vs epoch AND model loss vs epoch (for train and validation data).
    Args:
        history (Keras History Object): This is the Keras's History object returned when running model.fit()
        model_name (str): Name of the model that will correspond with plot PNG name
        image_path (str): Path to save plot results PNG file
    Returns:
        None
    """
    # training & validation accuracy values
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy vs Epochs")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend(loc="upper left")

    # training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss vs Epochs")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend(loc="upper left")

    plt.tight_layout()

    plt.savefig(f"{image_path}{model_name}.png")
    mlflow.log_artifact(f"{image_path}{model_name}.png")


def preprocess_nn_data(protein_data: dict[np.ndarray]) -> tuple:
    """
    Preprocess the protein data for the neural networks.
    Args:
        protein_data (dict[np.ndarray]): Protein data
    Returns:
        (tuple): X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        protein_data["positions"], protein_data["labels"], test_size=0.1
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2
    )

    # format train set
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
    y_train = to_categorical(y_train - 1)

    # format val set
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1] * X_val.shape[2]))
    y_val = to_categorical(y_val - 1)

    # format test set
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))
    y_test = to_categorical(y_test - 1)

    logging.info("NN Data Shapes:")
    logging.info(f"X_train: {X_train.shape}")
    logging.info(f"y_train: {y_train.shape}")
    logging.info(f"X_val: {X_val.shape}")
    logging.info(f"y_val: {y_val.shape}")
    logging.info(f"X_test: {X_test.shape}")
    logging.info(f"y_test: {y_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_cnn_data(protein_data: dict[np.ndarray]) -> tuple:
    """
    Preprocess the protein data for the convolutional neural network.
    The experimental notebook has more details/comments on the intuition
    and reasoning for the model architecture.
    Args:
        protein_data (dict[np.ndarray]): Protein data
    Returns:
        (tuple): X_train, X_val, X_test, y_train, y_val, y_test
    """
    # add spatial/channel dims
    tensor_positions = tf.convert_to_tensor(protein_data["positions"], dtype=tf.float32)
    tensor_positions = tf.expand_dims(tensor_positions, axis=-1)
    tensor_positions = tf.expand_dims(tensor_positions, axis=2)

    tensor_labels = tf.convert_to_tensor(protein_data["labels"], dtype=tf.int32)
    tensor_labels = tf.keras.utils.to_categorical(tensor_labels - 1, num_classes=3)

    # convert tensors temporarily to ndarrays to work with train_test_split
    np_positions = tensor_positions.numpy()
    np_labels = tensor_labels.numpy()

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        np_positions, np_labels, test_size=0.1
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2
    )

    # convert ndarrays back to tensors for CNN
    X_train = tf.convert_to_tensor(X_train)
    y_train = tf.convert_to_tensor(y_train)

    X_val = tf.convert_to_tensor(X_val)
    y_val = tf.convert_to_tensor(y_val)

    X_test = tf.convert_to_tensor(X_test)
    y_test = tf.convert_to_tensor(y_test)

    logging.info("CNN Data Shapes:")
    logging.info(f"X_train: {X_train.shape}")
    logging.info(f"y_train: {y_train.shape}")
    logging.info(f"X_val: {X_val.shape}")
    logging.info(f"y_val: {y_val.shape}")
    logging.info(f"X_test: {X_test.shape}")
    logging.info(f"y_test: {y_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test
