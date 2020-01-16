import logging
import numpy as np
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import joblib
from joblib import Memory

from src.data import load_data
from src.pipeline import preprocess_model_inputs
from src.pipeline import model_grid_search_cv


def get_pipeline_memory(cache_location = "train_cache"):
    return Memory(location = cache_location, verbose = 10)


def get_base_estimator(random_state):
    return RandomForestClassifier(
        bootstrap = True,
        criterion = "gini",
        max_features = "sqrt",
        class_weight = "balanced_subsample",
        oob_score = False,
        random_state = random_state,
    )


def get_grid_search_params():
    params = {
        "max_depth": np.linspace(25, 30, 3, dtype=int),
        "n_estimators": np.linspace(190, 230, 4, dtype=int),
        "min_samples_split": np.linspace(8, 10, 2, dtype=int),
        "min_samples_leaf": [2, 4],
    }

    logging.debug(f"Grid search params: {params}")

    return params


def eval_predictions(eval_type, predictor, X, y, output_path):

    logging.info(f"Evaluating: {eval_type}")
    y_pred = predictor.predict(X)

    with open(f"{output_path}/report_{eval_type}.txt", "w") as f:
        report = classification_report(y, y_pred)
        f.write(report)
        logging.info(report)


def write_out_model_params(params, output_path):
    def convert(o):
        if isinstance(o, np.int64): return int(o)
        elif isinstance(o, np.int32): return int(o)
        return str(o)

    with open(f"{output_path}/trained_params.json", 'w') as f:
        model_params = json.dumps(params, default=convert)
        f.write(model_params)
        logging.debug(f"model params: {model_params}")


def train_model(
        path, x_columns, y_column,
        output_path = "models",
        test_size = 0.3,
        random_state = None,
        cross_validation_folds = 5,
        verbose = 2,
):
    logging.info("Begin `train_model`")
    logging.debug(f"Load Data from {path}")
    logging.debug(f"x_columns: {x_columns}")
    logging.debug(f"y_column: {y_column}")

    df = load_data(path, x_columns + [y_column])

    logging.info("Preprocessing Inputs")
    df, new_X_columns = preprocess_model_inputs(df, x_columns, y_column)

    logging.debug(f"new_X_columns: {new_X_columns}")
    X = df[new_X_columns]
    y = df[y_column]

    logging.info("Splitting data into test and train sets")
    logging.info(f"test_size: {test_size}")
    logging.info(f"random_state: {random_state}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size = test_size,
        random_state = random_state,
        stratify = y,
    )

    grid_params = get_grid_search_params()

    logging.info("Getting Grid Search CV Pipeline")
    logging.debug(f"cross_validation_folds: {cross_validation_folds}")

    pipeline = model_grid_search_cv(
        new_X_columns,
        get_base_estimator(random_state),
        grid_params,
        verbose = verbose,
        cross_validation_folds = cross_validation_folds
    )

    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        logging.exception("Exception during pipeline fitting")
        raise e

    logging.info("Finished Grid Search CV")
    logging.info(f"Best Score: {pipeline['grid_search_cv'].best_score_}")
    logging.info(f"Best Params: {pipeline['grid_search_cv'].best_params_}")

    write_out_model_params(pipeline["grid_search_cv"].best_params_, output_path)

    eval_predictions("train", pipeline, X_train, y_train, output_path)
    eval_predictions("test", pipeline, X_test, y_test, output_path)

    logging.info(f"Writing out model to {output_path}/trained.model")

    joblib.dump(pipeline, f"{output_path}/trained.model")

    logging.info("End `train_model`")

