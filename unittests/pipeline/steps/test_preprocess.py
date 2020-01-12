import pytest


def test_can_call_pipeline_step_preprocess():
    import src.pipeline.steps

    try:
        src.pipeline.steps.preprocess()
    except AttributeError:
        pytest.fail("Couldn't call `preprocess`")
    except Exception:
        pass


def test_pipeline_step_preprocess_expected_type():
    from src.pipeline.steps import preprocess

    result = preprocess()
    try:
        assert(isinstance(result, tuple))
    except AssertionError:
        pytest.fail("`preprocess` did not return a tuple")

