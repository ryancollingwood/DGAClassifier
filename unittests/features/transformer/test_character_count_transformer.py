import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def test_can_call_character_count_transformer():
    import src.features.transformer

    try:
        src.features.transformer.CharacterCountTransformer()
    except AttributeError:
        pytest.fail("Couldn't create instance of `CharacterCountTransformer`")
    except Exception:
        pass


def test_character_count_transformer_pass():
    from src.features.transformer import CharacterCountTransformer

    test_df = pd.DataFrame(
        {
            "name": ["batman", "robin", "joker"],
            "alignment": ["good", "good", "bad"],
        })

    expected = pd.DataFrame(
        {
            "alignment": ["good", "good", "bad"],
            "name_character_count": [2, 0, 0]
        }
    ).values

    pipeline = Pipeline([
        (
            "CharacterCountTransformer",
            CharacterCountTransformer("character_count", ["name"], ["a"])
        ),
    ])

    result = pipeline.transform(test_df)
    try:
        assert(np.all(result == expected))
    except AssertionError:
        pytest.fail("CharacterCountTransformer transform did not produce expected result")

