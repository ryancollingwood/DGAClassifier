import pytest


def test_normalise_text_to_only_non_whitespace_pass():
    from src.preprocessing.text import normalise_text_to_only_regex_matches
    from src.preprocessing.text import get_regex_for_non_whitespace

    input_values = ["jessica-jones", "...-...", "so\t?", "U\rp\n", "Release Radar", "", "123 4", "insta white space"]
    expected_values = ["jessica-jones", "...-...", "so?", "Up", "ReleaseRadar", "", "1234", "instawhitespace"]

    matcher = get_regex_for_non_whitespace()

    for i, given_value in enumerate(input_values):
        try:
            output = normalise_text_to_only_regex_matches(given_value, matcher)
            assert(output == expected_values[i])
        except AssertionError:
            pytest.fail(f"Regex Normalisation of non_whitespace failed: {given_value} -> {output} != f{expected_values[i]}")


def test_normalise_text_to_only_non_whitespace_no_side_effects():
    from src.preprocessing.text import normalise_text_to_only_regex_matches
    from src.preprocessing.text import get_regex_for_non_whitespace
    from copy import copy

    input_values = ["jessica-jones", "...-...", "so\t?", "U\rp\n", "Release Radar", "", "123 4", "insta white space"]

    matcher = get_regex_for_non_whitespace()

    for given_value in input_values:
        try:
            given_value_copy = copy(given_value)
            normalise_text_to_only_regex_matches(given_value, matcher)
            assert(given_value == given_value_copy)
        except AssertionError:
            pytest.fail(f"Regex Normalisation of non_whitespace altered input data")


def test_normalise_text_to_only_non_whitespace_fail():
    from src.preprocessing.text import normalise_text_to_only_regex_matches
    from src.preprocessing.text import get_regex_for_non_whitespace

    input_values = ["jessica-jones", "...-...", "so\t?", "U\rp\n", "Release Radar", "", "123 4", "insta white space"]

    matcher = get_regex_for_non_whitespace()

    for i, given_value in enumerate(input_values):
        try:
            output = normalise_text_to_only_regex_matches(given_value, matcher)
            assert(output == given_value)
        except AssertionError:
            return
        except Exception:
            pytest.fail("Regex Normalisation of non_whitespace did not fail as expected")
