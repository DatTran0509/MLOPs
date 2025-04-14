from config import *
from models import build_model
from dataset_utils import load_data
from mlflow_utils import hash_dataset
import mlflow
import optuna
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.callbacks import EarlyStopping
import os
import random

mlflow.set_tracking_uri("file:///kaggle/working/mlruns")
mlflow.set_experiment(EXPERIMENT_NAME)

X_train, X_test, y_train, y_test, le = load_data(DATA_DIR, IMG_SIZE)
num_classes = len(np.unique(y_train))

def objective(trial):
    learning_rate = trial.suggest_categorical("learning_rate", [1e-2, 1e-3, 1e-4])
    dropout = 0.5
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    epochs = trial.suggest_categorical("epochs", [10, 15])

    model = build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                        num_classes=num_classes,
                        learning_rate=learning_rate,
                        dropout=dropout)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "learning_rate": learning_rate,
            "dropout": dropout,
            "batch_size": batch_size,
            "epochs": epochs,
            "dataset_hash": hash_dataset(DATA_DIR)
        })

        # Huấn luyện
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
            verbose=1
        )

        # Dự đoán và đánh giá
        y_pred = np.argmax(model.predict(X_test), axis=1)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")

        # Log metrics
        mlflow.log_metrics({
            "val_accuracy": acc,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1
        })

        # Log model
        mlflow.keras.log_model(model, "model")

        # Log plot loss/accuracy
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(history.history['loss'], label='Train')
        ax[0].plot(history.history['val_loss'], label='Val')
        ax[0].set_title(f'Trial {trial.number} Loss')
        ax[0].legend()
        ax[1].plot(history.history['accuracy'], label='Train')
        ax[1].plot(history.history['val_accuracy'], label='Val')
        ax[1].set_title(f'Trial {trial.number} Accuracy')
        ax[1].legend()
        plot_filename = f"plot_trial_{trial.number}.png"
        fig.savefig(plot_filename)
        mlflow.log_artifact(plot_filename, artifact_path="plots")
        plt.tight_layout()
        plt.show()  # ✅ In ra sau mỗi trial
        plt.close(fig)

    return f1
