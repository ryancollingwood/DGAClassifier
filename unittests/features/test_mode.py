import pytest
import numpy as np


def test_can_call_mode():
    import src.features

    try:
        src.features.mode()
    except AttributeError:
        pytest.fail("Couldn't call mode")
    except Exception:
        pass


def test_mode_pass():
    from src.features import mode

    input_value = np.array([1,2,3,3,4,5,6,7,8])
    expected = 3

    result = mode(input_value)
    try:
        assert(result == expected)
    except:
        pytest.fail("Mode did not return expected result")


def test_mode_pass_single_distinct_element():
    from src.features import mode

    input_value = np.array([1,1,1,1,1,1,1,1,1])
    expected = 1

    result = mode(input_value)

    try:
        assert(result == expected)
    except AssertionError:
        pytest.fail("Mode did not return expected result when only 1 distinct element")


def test_mode_no_mode_pass():
    from src.features import mode

    input_value = np.array([1,2,3,4,5,6,7,8,9])

    result = mode(input_value)

    try:
        assert(np.isnan(result))
    except AssertionError:
        pytest.fail("Mode did not return expected result no mode")
