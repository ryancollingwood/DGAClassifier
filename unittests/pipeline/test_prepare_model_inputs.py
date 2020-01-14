import pytest
from sklearn.pipeline import Pipeline


def test_can_call_prepare_model_inputs():
    import src.pipeline

    try:
        src.pipeline.prepare_model_inputs()
    except AttributeError:
        pytest.fail("Couldn't call `prepare_model_inputs`")
    except Exception:
        pass


def test_can_call_pipeline_prepare_model_inputs():
    import src.pipeline

    try:
        src.pipeline.pipeline_prepare_model_inputs()
    except AttributeError:
        pytest.fail("Couldn't call `pipeline_prepare_model_inputs`")
    except Exception:
        pass


def test_pipeline_prepare_model_inputs_return_expected_type():
    from src.pipeline import pipeline_prepare_model_inputs

    result = pipeline_prepare_model_inputs(["foo"])
    try:
        assert(isinstance(result, list))
    except AssertionError:
        pytest.fail("`pipeline_prepare_model_inputs` did not return List")

