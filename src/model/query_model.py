from typing import List, Dict
from copy import copy
import pandas as pd
from .load_model import load_model


class QueryModel:
    def __init__(self, model_filename: str, desired_class: str):
        """
        Object that facilitates predictions of user submitted data on
        a persisted model

        :param model_filename: str - file path to the model
        :param desired_class: str - what is the class we want
            If all predicted values are of the classs `desired_class`
            then the `predict` function will return true.
        """
        self.model = load_model(model_filename)
        self.input_columns = list()

        # get the columns names we fed into the first step of our model
        # so that we can feed in a Dataframe with the same columns headers
        for transformer_step in self.model.steps[0][1].transformer_list:
            [self.input_columns.append(x) for x in transformer_step[1].in_columns if x not in self.input_columns]

        try:
            assert desired_class
        except AssertionError:
            raise ValueError("`desired_class` must be defined")

        try:
            assert(desired_class in self.model.classes_)
        except AssertionError:
            raise ValueError(f"`desired_class` must be in ({self.model.classes_}), got: {desired_class}")

        self.desired_class = str(desired_class)

    @staticmethod
    def str_to_list(value: str) -> List[str]:
        """
        Convert a comma separated string input to list,
        if already a list the input is returned
        :param value: str but allows for list
        :return: list
        """
        if isinstance(value, list):
            return value

        result = str(value).split(",")
        if not isinstance(result, list):
            result = list(result)

        return result

    def predict(self, in_values: Dict[str, List]) -> bool:
        """
        For the given in_values make a prediction using the loaded model
        The input is expected to be a Dictionary of List[str], however
        str and List are acceptable inputs.

        If not a Dict then the following logic applies:

          If in_values is a str, it is expected to be a comma separated str

          If the number of columns in the input layer of the model is > 1
          Then in_values is treated as features of a single observation

          If the number of columns in the input layer of the model == 1
          Then in_values is treated as multiple observations.

        :param in_values: Dict[str, List] also allows for str and List[str]
        :return: bool - True if all values matched `desired_class`, False otherwise
        """
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

        try:
            assert(isinstance(values, dict))
        except AssertionError:
            raise ValueError("`in_values` must either be Pandas DataFrame friendly dict, or string, or list of strings")

        predict_df = pd.DataFrame(values, columns=self.input_columns)

        results = self.model.predict(predict_df)
        predict_df["result"] = results
        print(predict_df)

        return all(predict_df["result"] == self.desired_class)

    def get_input(self) -> Dict[str, List]:
        """
        Get input from console and validate it
        Return None if no input given so we can quit

        :return: Dict[str, List]
        """
        given_inputs = dict()
        for column in self.input_columns:
            current_input = input(f"{column}>").strip()

            if not current_input:
                return None

            given_inputs[column] = current_input

        return self.validate_input(given_inputs, True)

    def validate_input(self, given_inputs: Dict[str, List], retry_input: bool) -> Dict[str, List]:
        """
        Ensure the inputs are valid. Validity means that every list in the
        dictionary is of equal length, and thus can be read in as a Pandas Dataframe

        :param given_inputs: Dict[str, List]
        :param retry_input: Error handling flow
            if true when `given_inputs` are NOT valid then attempt to
                recapture from console input

            if false when `given_inputs` are NOT valid then raise a
                ValueError
        :return: Dict[str, List]
        """
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

    def interactive(self) -> bool:
        """
        Allow the user to enter input values.

        User can either enter a single string for each input feature of the model.
        Or they can specify multiple values for the input features as a comma
        separated string.

        e.g.
        `reddit` -> Single observation
        `reddit,disney,kotaku` -> 3 Observations

        If the model has multiple input features then the user is expected to
        enter an equal number of elements for each feature.

        :return: bool
        """
        result = None

        in_value = self.get_input()

        while in_value:
            last_result = self.predict(in_value)

            if result is not None:
                result = result and last_result
            else:
                result = last_result

            in_value = self.get_input()

        return result

