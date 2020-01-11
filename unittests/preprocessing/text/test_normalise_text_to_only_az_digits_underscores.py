import pytest


def test_normalise_text_to_only_az_digits_underscores_pass():
    from src.preprocessing.text import normalise_text_to_only_regex_matches
    from src.preprocessing.text import get_regex_for_az_digits_underscores

    input_values = ["jessica-jones", "...-...", "so?", "Up", "Release Radar", "", "123 4"]
    expected_values = ["jessica-jones", "-", "so", "p", "eleaseadar", "", "1234"]

    matcher = get_regex_for_az_digits_underscores()

    for i, given_value in enumerate(input_values):
        try:
            output = normalise_text_to_only_regex_matches(given_value, matcher)
            assert(output == expected_values[i])
        except AssertionError:
            pytest.fail(f"Regex Normalisation of az_digits_underscores failed: {given_value} -> {output} != f{expected_values[i]}")


def test_normalise_text_to_only_az_digits_underscores_no_side_effects():
    from copy import copy
    from src.preprocessing.text import normalise_text_to_only_regex_matches
    from src.preprocessing.text import get_regex_for_az_digits_underscores

    input_values = ["jessica-jones", "...-...", "so?", "Up", "Release Radar", "", "123 4"]

    matcher = get_regex_for_az_digits_underscores()

    for given_value in input_values:
        try:
            given_value_copy = copy(given_value)
            normalise_text_to_only_regex_matches(given_value, matcher)
            assert(given_value == given_value_copy)
        except AssertionError:
            pytest.fail(f"Regex Normalisation of az_digits_underscores altered input data")


def test_normalise_text_to_only_az_digits_underscores_fail():
    from src.preprocessing.text import normalise_text_to_only_regex_matches
    from src.preprocessing.text import get_regex_for_az_digits_underscores

    input_values = ["jessica-jones", "...-...", "so?", "Up", "Release Radar", "", "123 4"]

    matcher = get_regex_for_az_digits_underscores()

    for i, given_value in enumerate(input_values):
        try:
            output = normalise_text_to_only_regex_matches(given_value, matcher)
            assert(output == given_value)
        except AssertionError:
            return
        except Exception:
            pytest.fail("Regex Normalisation of az_digits_underscores did not fail as expected")
