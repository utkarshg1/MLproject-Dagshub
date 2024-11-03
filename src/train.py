import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utils import get_yaml_params, save_model
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split, GridSearchCV
from mlflow.models import infer_signature
from urllib.parse import urlparse
from dotenv import load_dotenv
import mlflow
import os

# Load environment variables
load_dotenv()

# Hyperparameter tuning


def hyperparameter_tuning(xtrain, ytrain, param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2,
        scoring="f1_macro"
    )
    grid_search.fit(xtrain, ytrain)
    return grid_search


# Load params from yaml file
params = get_yaml_params("train")

# train model


def train(data_path, model_path, random_state):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    Y = data["Outcome"]

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    with mlflow.start_run():
        xtrain, xtest, ytrain, ytest = train_test_split(
            X, Y, test_size=0.2, random_state=random_state)
        signature = infer_signature(xtrain, ytrain)
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [4, 5],
            "min_samples_leaf": [4, 5]
        }
        grid_search = hyperparameter_tuning(xtrain, ytrain, param_grid)

        # Get best model and params
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Best parameters : {best_params}")
        best_score = grid_search.best_score_
        print(f"f1 macro score cv 5: {best_score:.4f}")

        # Predict and evaluate model
        ypred_test = best_model.predict(xtest)
        acc = accuracy_score(ytest, ypred_test)
        print(f"Accuracy Score : {acc:.4f}")

        f1 = f1_score(ytest, ypred_test, average="macro")
        print(f"F1 Macro score : {f1:.4f}")

        cm = confusion_matrix(ytest, ypred_test)
        print(cm)

        report = classification_report(ytest, ypred_test)
        print(report)

        mlflow.log_params(best_params)
        results = {
            "accuracy": acc,
            "f1_macro": f1,
            "f1_cv": best_score
        }
        mlflow.log_metrics(results)
        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(report, "classification_report.txt")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                best_model, "model", registered_model_name="Best Model")
        else:
            mlflow.sklearn.log_model(best_model, "model", signature=signature)

        # Save model pickle
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        save_model(best_model, model_path)


if __name__ == "__main__":
    train(
        params["data"],
        params["models"],
        params["random_state"]
    )
