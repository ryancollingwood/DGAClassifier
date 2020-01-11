import pytest


def test_can_call_normalise_text_to_ascii():
    try:
        import src.preprocessing.text
        src.preprocessing.text.normalise_text_to_ascii()
    except AttributeError:
        pytest.fail("Couldn't call normalise_text_to_ascii")
    except Exception:
        pass


def test_normalise_text_to_ascii_pass():
    from src.preprocessing.text import normalise_text_to_ascii

    given_inputs = ["née", "entrepôt", "Zoë", "über", "háček", "Lech Wałęsa", "Chișinău"]
    expected_outputs = ["nee", "entrepot", "Zoe", "uber", "hacek", "Lech Walesa", "Chisinau"]

    for i, given_input in enumerate(given_inputs):
        try:
            normalised_value = normalise_text_to_ascii(given_input)
            assert(normalised_value == expected_outputs[i])
        except AssertionError:
            pytest.fail(f"ASCII Normalisation failed: {given_input} -> {normalised_value} != f{expected_outputs[i]}")


def test_normalise_text_to_ascii_no_side_effects():
    from copy import copy
    from src.preprocessing.text import normalise_text_to_ascii

    given_inputs = ["née", "entrepôt", "Zoë", "über", "háček", "Lech Wałęsa", "Chișinău"]

    for given_input in given_inputs:
        try:
            given_input_copy = copy(given_input)
            normalise_text_to_ascii(given_input)
            assert (given_input == given_input_copy)
        except AssertionError:
            pytest.fail("normalise_text_to_ascii altered input data")


def test_normalise_text_to_ascii_fail():
    from src.preprocessing.text import normalise_text_to_ascii

    given_inputs = ["née", "entrepôt", "Zoë", "über", "háček", "Lech Wałęsa", "Chișinău"]

    for i, give_input in enumerate(given_inputs):
        try:
            normalised_value = normalise_text_to_ascii(give_input)
            assert(normalised_value == give_input)
        except AssertionError:
            return
        except Exception:
            pytest.fail("normalise_text_to_ascii did not fail as expected")
