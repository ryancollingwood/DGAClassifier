import os.path
import joblib


def load_model(model_filename):
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Model not found at {model_filename}, have you finished model training?")

    return joblib.load(model_filename)
