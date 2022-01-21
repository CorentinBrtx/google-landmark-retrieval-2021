import json
import os
from typing import Dict

from src.utils.file_manipulation import get_environ


def save_model_config(model_config: Dict, model_name: str):
    """
    Save model configuration to a file.
    """

    data_dir = get_environ("LANDMARK_RETRIEVAL_DATA_DIR")
    model_config_path = os.path.join(data_dir, "models", model_name, "config.json")

    os.makedirs(os.path.dirname(model_config_path), exist_ok=True)

    with open(model_config_path, "w") as f:
        f.write(json.dumps(model_config, indent=4))


def load_model_config(model_name: str) -> Dict:
    """
    Load model configuration from a file.
    """

    data_dir = get_environ("LANDMARK_RETRIEVAL_DATA_DIR")
    model_config_path = os.path.join(data_dir, "models", model_name, "config.json")

    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    return model_config
