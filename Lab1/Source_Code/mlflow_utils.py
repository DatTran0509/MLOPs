import hashlib
import os
import mlflow

def hash_dataset(path):
    hash_md5 = hashlib.md5()
    for root, dirs, files in os.walk(path):
        for f in sorted(files):
            with open(os.path.join(root, f), "rb") as file:
                for chunk in iter(lambda: file.read(4096), b""):
                    hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_best_run(experiment_name):
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(exp.experiment_id, order_by=["metrics.val_f1 DESC"])
    return runs[0]
