import sys
import os.path
import joblib


def resource_path(relative_path):
    """
    as per: https://stackoverflow.com/a/13790741/2805700

    Get absolute path to resource, works for dev and for PyInstaller
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def load_model(model_filename):

    model_resource_path = resource_path(model_filename)

    if not os.path.exists(model_resource_path):
        raise FileNotFoundError(f"Model not found at {model_filename}, have you finished model training?")

    return joblib.load(model_resource_path)
