import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def test_can_call_text_length_transformer():
    import src.features.transformer

    try:
        src.features.transformer.TextLengthTransformer()
    except AttributeError:
        pytest.fail("Couldn't create instance of `TextLengthTransformer`")
    except Exception:
        pass


def test_text_length_transformer_pass():
    from src.features.transformer import TextLengthTransformer

    test_df = pd.DataFrame(
        {
            "name": ["batman", "robin", "joker"],
            "alignment": ["good", "good", "bad"],
        })

    expected = pd.DataFrame(
        {
            "alignment": ["good", "good", "bad"],
            "name_len": np.array([6, 5, 5], np.int64),
        }
    ).values

    pipeline = Pipeline([
        (
            "TextLengthTransformer",
            TextLengthTransformer(
                "len", ["name"]
            )
        ),
    ])

    result = pipeline.transform(test_df)

    try:
        assert(np.all(result == expected))
    except AssertionError:
        pytest.fail("TextLengthTransformer transform did not produce expected result")


def test_text_length_transformer_with_trim_pass():
    from src.features.transformer import TextLengthTransformer

    test_df = pd.DataFrame(
        {
            "name": ["batman     ", "    robin     ", "jo ker"],
            "alignment": ["good", "good", "bad"],
        })

    expected = pd.DataFrame(
        {
            "alignment": ["good", "good", "bad"],
            "name_len": np.array([6, 5, 6], np.int64),
        }
    ).values

    pipeline = Pipeline([
        (
            "TextLengthTransformer",
            TextLengthTransformer(
                "len", ["name"], True
            )
        ),
    ])

    result = pipeline.transform(test_df)

    try:
        assert(np.all(result == expected))
    except AssertionError:
        pytest.fail("TextLengthTransformer with trim transform did not produce expected result")
