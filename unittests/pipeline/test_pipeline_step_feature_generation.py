import pytest


def test_can_call_pipeline_step_feature_generation():
    import src.pipeline

    try:
        src.pipeline.pipeline_step_feature_generation()
    except AttributeError:
        pytest.fail("Couldn't call `pipeline_step_feature_generation`")
    except Exception:
        pass


def test_pipeline_step_feature_generation_expcetd_type():
    from src.pipeline import pipeline_step_feature_generation

    result = pipeline_step_feature_generation()
    try:
        assert(isinstance(result, tuple))
    except AssertionError:
        pytest.fail("`pipeline_step_feature_generation` did not return a tuple")

