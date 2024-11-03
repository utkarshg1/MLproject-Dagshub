import yaml
import pickle


def get_yaml_params(param):
    with open("params.yaml", "r") as f:
        value = yaml.safe_load(f)[param]
    return value


def save_model(model, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved at : {filepath}")


def load_model(filepath):
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model
