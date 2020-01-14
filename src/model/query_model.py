import pandas as pd
from .load_model import load_model


class QueryModel:
    def __init__(self, model_filename):
        self.model = load_model(model_filename)

    def predict(self, value):
        given_value = str(value).split(",")
        if not isinstance(given_value, list):
            given_value = list(given_value)

        predict_df = pd.DataFrame(given_value, columns=["input"])

        results = self.model.predict(predict_df)
        predict_df["result"] = results
        print(predict_df)

        return all(predict_df["result"] == "legit")

    def get_input(self):
        given_value = input(">").strip()

        return given_value

    def interactive(self):
        last_result = None
        in_value = self.get_input()

        while in_value:
            last_result = self.predict(in_value)
            in_value = self.get_input()

        return last_result

