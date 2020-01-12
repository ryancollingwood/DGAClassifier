from src.build import train_model
import argparse

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
                    help="Ratio of data to holdout for testing e.g. `0.3` for 30% - default `0.3`", default=0.3)
    ap.add_argument("-cv", "--cross_validation_folds", required=False,
                    help="Number of cross validation folds e.g. 10 for 10 Folds - default `5`", default=5)
    ap.add_argument("-r", "--random_state", required=False,
                    help="Specify the random state for reproducibility e.g. `42`", default=None)
    ap.add_argument("-v", "--verbose", required=False,
                    help="Specify verbosity of training process e.g. `0` for no training updates", default=1)

    args = vars(ap.parse_args())

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
