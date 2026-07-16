from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from .prepare_model_inputs import pipeline_prepare_model_inputs


def model_grid_search_cv(
        x_columns,
        estimator,
        grid_params,
        verbose = 2,
        cross_validation_folds = 10,
):
    steps = pipeline_prepare_model_inputs(x_columns)

    steps.append(
        tuple(
            [
                "grid_search_cv",
                GridSearchCV(
                    estimator,
                    grid_params,
                    scoring="f1_micro",
                    n_jobs=-1,
                    cv=cross_validation_folds,
                    refit=True,
                    verbose=verbose,
                    error_score="raise",
                    return_train_score=False,
                )
            ]
        )
    )

    # `GridSearchCV` accepts an integer verbosity level, but since
    # scikit-learn 1.x `Pipeline.verbose` must be a bool.
    return Pipeline(steps, verbose = bool(verbose))
