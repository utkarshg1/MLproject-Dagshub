import pandas as pd
from utils import load_model, get_yaml_params
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from dotenv import load_dotenv
import os
import mlflow

# Load environment variables
load_dotenv()


# Load params from yaml file
params = get_yaml_params("train")


# Evaluate model
def evaluate(data_path, model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    Y = data["Outcome"]

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    model = load_model(model_path)

    preds = model.predict(X)
    acc = accuracy_score(Y, preds)
    f1 = f1_score(Y, preds)
    f1_cv = cross_val_score(model, X, Y, cv=5, scoring="f1_macro").mean()
    results = {
        "accuracy": acc,
        "f1_macro": f1,
        "f1_cv": f1_cv
    }
    mlflow.log_metrics(results)
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Macro : {f1:.4f}")
    print(f"F1 CV : {f1_cv:.4f}")


if __name__ == "__main__":
    evaluate(params["data"], params["models"])
