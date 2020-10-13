import os
import json

WEIGHTS_DIR = "Prediction/traj_models"
MODELS_JSON_PATH = os.path.join(WEIGHTS_DIR, "models_params.json")


def get_weights_path(model_name, suffix='best'):
    return os.path.join(WEIGHTS_DIR, f"{model_name}_{suffix}.pth")


def get_models_dict():
    with open(MODELS_JSON_PATH, 'r') as fp:
        return json.load(fp)


def get_trials_split(model_name):
    json_dict = get_models_dict()
    train_trials = json_dict[model_name]['training_params']['train_trials']
    val_trials = json_dict[model_name]['training_params']['val_trials']
    test_trials = json_dict[model_name]['training_params']['test_trials']
    return train_trials, val_trials, test_trials
