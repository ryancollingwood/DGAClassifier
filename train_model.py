import os
import sys
import argparse
import logging
from src.logging import setup_logging
from src.model import train_model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True,
                    help="Path to input csv, can be a url - e.g. `data/raw/dga_domains.csv`", default = "data/raw/dga_domains.csv")
    ap.add_argument("-o", "--output", required = True,
                    help="Path to where the model will be saved - default: `models`", default = "models")
    ap.add_argument("-x", "--input_columns", required=False,
                    help="Comma separated column names for input variables e.g. `domain,host` - default: `domain`",
                    default="domain")
    ap.add_argument("-y", "--target_column", required=False,
                    help="Column name for target variable e.g. `class` - default `class`", default="class")
    ap.add_argument("-t", "--test_size", required=False,
                    help="Ratio of data to holdout for testing e.g. `0.3` for 30% - default `0.3`", default=0.3, type=float)
    ap.add_argument("-cv", "--cross_validation_folds", required=False,
                    help="Number of cross validation folds e.g. 10 for 10 Folds - default `5`", default=5, type=int)
    ap.add_argument("-r", "--random_state", required=False,
                    help="Specify the random state for reproducibility e.g. `42`", default=None, type=int)
    ap.add_argument("-v", "--verbose", required=False,
                    help="Specify verbosity of training process e.g. `0` for no training updates", default=1, type=int)

    args = vars(ap.parse_args())

    if not os.path.exists(args['output']):
        os.makedirs(args['output'])

    log_path = os.path.join(args['output'], "training.log")
    setup_logging(logging.DEBUG, file_name=log_path)

    try:
        logging.info(f"Logging training to: {log_path}")
    except FileNotFoundError:
        print(f"Unable to create log file at {log_path}, do you have the needed file system permissions?")
        sys.exit(1)

    train_model(
        args["path"],
        x_columns=args["input_columns"].split(","),
        y_column=args["target_column"],
        output_path=args["output"],
        test_size=args["test_size"],
        random_state=args["random_state"],
        cross_validation_folds=args["cross_validation_folds"],
        verbose=args["verbose"]
    )
