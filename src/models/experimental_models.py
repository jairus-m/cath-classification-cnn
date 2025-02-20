"""
This module contains each experimental deep learning model and their 
corresponding mlflow logs, wrapped in their own functions. 
"""
import mlflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

from src.utils import plot_results

IMAGE_PATH = "images/"

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("PROTEIN_NN_CNN")

def nn_single__1(X_train, X_val, X_test, y_train, y_val, y_test, signature):
    """
    Experiment 1: Basic Neural Network with one Layer
    """
    with mlflow.start_run(run_name="Basic NN - One Layer"):
        model = keras.Sequential(
            [layers.Dense(4, activation="relu"), layers.Dense(3, activation="softmax")]
        )
        model.compile(
            optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        mlflow.log_param("hidden_layers", 1)
        mlflow.log_param("neurons", 4)
        mlflow.log_param("hidden_layer_activations", "relu")
        mlflow.log_param("output_layer_activation", "softmax")
        mlflow.log_param("optimizer", "rmsprop")
        mlflow.log_param("loss_function", "categorical_crossentropy")

        history = model.fit(
            X_train, y_train, epochs=1, batch_size=128, validation_data=(X_val, y_val)
        )

        plot_results(history, "nn_single__1", IMAGE_PATH)

        mlflow.log_param("epochs", 20)
        mlflow.log_param("batch_size", 128)

        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_loss", test_loss)

        mlflow.keras.log_model(model, "nn_single__1", signature=signature)

        mlflow.end_run()


def nn_double__2(X_train, X_val, X_test, y_train, y_val, y_test, signature):
    """
    Experiment 2: Basic Neural Network, 2 Hidden layers
    """
    with mlflow.start_run(run_name="Basic NN - Two Layers"):
        model = keras.Sequential(
            [
                layers.Dense(64, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(3, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        mlflow.log_param("hidden_layers", 2)
        mlflow.log_param("neurons_per_layer", 64)
        mlflow.log_param("hidden_layer_activations", "relu")
        mlflow.log_param("output_layer_activation", "softmax")
        mlflow.log_param("optimizer", "rmsprop")
        mlflow.log_param("loss_function", "categorical_crossentropy")

        history = model.fit(
            X_train, y_train, epochs=20, batch_size=128, validation_data=(X_val, y_val)
        )

        plot_results(history, "nn_double__2", IMAGE_PATH)

        mlflow.log_param("epochs", 20)
        mlflow.log_param("batch_size", 128)

        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_loss", test_loss)

        mlflow.keras.log_model(model, "nn_double__2", signature=signature)

        mlflow.end_run()


def cnn__3(X_train, X_val, X_test, y_train, y_val, y_test, signature):
    """
    Experiment 3: CNN
    """
    with mlflow.start_run(run_name="CNN"):
        model = keras.Sequential(
            [
                layers.Input(shape=(1202, 1, 3, 1)),
                layers.Conv3D(
                    filters=32, kernel_size=(3, 1, 3), activation="relu", padding="same"
                ),
                layers.MaxPooling3D(pool_size=(2, 1, 1), padding="same"),
                layers.Conv3D(
                    filters=64, kernel_size=(3, 1, 3), activation="relu", padding="same"
                ),
                layers.MaxPooling3D(pool_size=(2, 1, 1), padding="same"),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(3, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        mlflow.log_param("model_type", "CNN")
        mlflow.log_param("input_shape", (1202, 1, 3, 1))
        mlflow.log_param("conv_layers", 2)
        mlflow.log_param("dense_layers", 3)
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("loss_function", "categorical_crossentropy")

        history = model.fit(
            X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=128
        )

        plot_results(history, "cnn__3", IMAGE_PATH)

        mlflow.log_param("epochs", 20)
        mlflow.log_param("batch_size", 128)

        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_loss", test_loss)

        mlflow.keras.log_model(model, "cnn__3", signature=signature)

        mlflow.end_run()


def cnn__4(X_train, X_val, X_test, y_train, y_val, y_test, signature):
    """
    Experiment 4: Simplified CNN with Early Stopping
    """
    with mlflow.start_run(run_name="Simplified CNN w/ Early Stopping"):
        model = keras.Sequential(
            [
                layers.Conv3D(
                    32,
                    kernel_size=(3, 1, 3),
                    activation="relu",
                    padding="same",
                    input_shape=(1202, 1, 3, 1),
                ),
                layers.MaxPooling3D(pool_size=(2, 1, 1), padding="same"),
                layers.Conv3D(
                    64, kernel_size=(3, 1, 3), activation="relu", padding="same"
                ),
                layers.MaxPooling3D(pool_size=(2, 1, 1), padding="same"),
                layers.Flatten(),
                layers.Dense(3, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )

        # Updated log_params
        mlflow.log_params({
            "model_type": "CNN",
            "input_shape": (1202, 1, 3, 1),
            "conv_layers": 2,
            "dense_layers": 1,
            "optimizer": "adam",
            "loss_function": "categorical_crossentropy",
            "max_epochs": 50,
            "batch_size": 128,
            "early_stopping_patience": 2,
            "conv1_filters": 32,
            "conv2_filters": 64,
            "conv_kernel_size": (3, 1, 3),
            "pool_size": (2, 1, 1),
            "activation_function": "relu",
            "output_activation": "softmax"
        })

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=50,  # Increase max epochs, early stopping will prevent overfitting
            batch_size=128,
            callbacks=[early_stopping],
        )

        plot_results(history, "cnn__4", IMAGE_PATH)

        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_loss", test_loss)

        mlflow.keras.log_model(model, "cnn__4", signature=signature)

        mlflow.end_run()
