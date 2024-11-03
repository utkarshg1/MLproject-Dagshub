import pandas as pd
from utils import get_yaml_params
import os

# Load parameters from params.yaml file
params = get_yaml_params("preprocess")

# Preprocess data


def preprocess(input_path, output_path):
    data = pd.read_csv(input_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to : {output_path}")


if __name__ == "__main__":
    preprocess(params["input"], params["output"])
