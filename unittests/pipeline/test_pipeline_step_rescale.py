import pytest


def test_can_call_pipeline_step_rescale():
    import src.pipeline

    try:
        src.pipeline.pipeline_step_rescale()
    except AttributeError:
        pytest.fail("Couldn't call `pipeline_step_rescale`")


def test_pipeline_step_rescale_expected_type():
    from src.pipeline import pipeline_step_rescale

    result = pipeline_step_rescale()
    try:
        assert(isinstance(result, tuple))
    except AssertionError:
        pytest.fail("`pipeline_step_rescale` did not return a tuple")

