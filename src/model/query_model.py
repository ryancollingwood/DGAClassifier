from copy import copy
import pandas as pd
from .load_model import load_model


class QueryModel:
    def __init__(self, model_filename):
        self.model = load_model(model_filename)
        self.input_columns = list()

        # get the columns names we fed into the first step of our model
        # so that we can feed in a Dataframe with the same columns headers
        for transformer_step in self.model.steps[0][1].transformer_list:
            [self.input_columns.append(x) for x in transformer_step[1].in_columns if x not in self.input_columns]

    @staticmethod
    def str_to_list(value: str):
        if isinstance(value, list):
            return value

        result = str(value).split(",")
        if not isinstance(result, list):
            result = list(result)

        return result

    def predict(self, in_values):
        values = copy(in_values)

        if isinstance(values, str) or isinstance(values, list):
            if isinstance(values, str):
                split_values = values.split(",")
            else:
                split_values = values

            values = dict()
            if len(self.input_columns) > 1:
                for i, column in enumerate(self.input_columns):
                    values[column] = split_values[i]
            else:
                values[self.input_columns[0]] = split_values

            try:
                values = self.validate_input(values, False)
            except ValueError:
                return None
        else:
            for key in values:
                values[key] = self.str_to_list(values[key])

        predict_df = pd.DataFrame(values, columns=self.input_columns)

        results = self.model.predict(predict_df)
        predict_df["result"] = results
        print(predict_df)

        return all(predict_df["result"] == "legit")

    def get_input(self):
        given_inputs = dict()
        for column in self.input_columns:
            given_inputs[column] = input(f"{column}>").strip()

        return self.validate_input(given_inputs, True)

    def validate_input(self, given_inputs, retry_input):
        n_elements = None

        for key in given_inputs:
            given_inputs[key] = self.str_to_list(given_inputs[key])
            if n_elements is not None:
                if n_elements != len(given_inputs[key]):
                    if retry_input:
                        print("Must enter an equal number of elements.")
                        return self.get_input()
                    else:
                        raise ValueError("Must enter an equal number of elements.")
            else:
                n_elements = len(given_inputs[key])
        return given_inputs

    def interactive(self):
        last_result = None
        in_value = self.get_input()

        while in_value:
            last_result = self.predict(in_value)
            in_value = self.get_input()

        return last_result

