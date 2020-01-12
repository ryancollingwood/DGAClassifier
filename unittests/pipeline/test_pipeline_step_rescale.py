import pytest


def test_can_call_pipeline_step_rescale():
    import src.pipeline.steps

    try:
        src.pipeline.steps.rescale()
    except AttributeError:
        pytest.fail("Couldn't call `rescale`")


def test_pipeline_step_rescale_expected_type():
    from src.pipeline.steps import rescale

    result = rescale()
    try:
        assert(isinstance(result, tuple))
    except AssertionError:
        pytest.fail("`rescale` did not return a tuple")

