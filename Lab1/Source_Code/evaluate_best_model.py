from config import *
from dataset_utils import load_data
from mlflow_utils import get_best_run
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import os
import random

X_train, X_test, y_train, y_test, le = load_data(DATA_DIR, IMG_SIZE)

# Load best model from mlruns
import mlflow.keras

best_run = get_best_run(EXPERIMENT_NAME)

model_path = f"/kaggle/working/mlruns/{best_run.info.experiment_id}/{best_run.info.run_id}/artifacts/model"
model = mlflow.keras.load_model(model_path)

# Predict
y_pred = np.argmax(model.predict(X_test), axis=1)

# Metrics
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")

# Print metrics
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
