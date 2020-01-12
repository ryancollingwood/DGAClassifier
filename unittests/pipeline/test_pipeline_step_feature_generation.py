import pytest


def test_can_call_pipeline_step_feature_generation():
    import src.pipeline.steps

    try:
        src.pipeline.steps.feature_generation()
    except AttributeError:
        pytest.fail("Couldn't call `feature_generation`")
    except Exception:
        pass


def test_pipeline_step_feature_generation_expcetd_type():
    from src.pipeline.steps import feature_generation

    result = feature_generation()
    try:
        assert(isinstance(result, tuple))
    except AssertionError:
        pytest.fail("`feature_generation` did not return a tuple")

