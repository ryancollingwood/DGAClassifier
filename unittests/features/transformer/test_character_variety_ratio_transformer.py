import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def test_can_call_character_variety_ratio_transformer():
    import src.features.transformer

    try:
        src.features.transformer.CharacterVarietyRatioTransformer()
    except AttributeError:
        pytest.fail("Couldn't create instance of `CharacterVarietyRatioTransformer`")
    except Exception:
        pass


def test_character_variety_ratio_transformer_pass():
    from src.features.transformer import CharacterVarietyRatioTransformer

    test_df = pd.DataFrame(
        {
            "name": ["batman", "robin", "joker", "azrael"],
            "alignment": ["good", "good", "bad", "chaotic"],
        })

    expected = pd.DataFrame(
        {
            "name": ["batman", "robin", "joker", "azrael"],
            "alignment_character_variety_ratio": [1, 1, 1, 0.33]
        }
    ).values

    pipeline = Pipeline([
        (
            "CharacterVarietyRatioTransformer",
            CharacterVarietyRatioTransformer(
                "character_variety_ratio",
                ["alignment"], ["o"]
            )
        ),
    ])

    result = pipeline.transform(test_df)

    try:
        assert(np.all(result == expected))
    except AssertionError:
        pytest.fail("CharacterVarietyRatioTransformer transform did not produce expected result")
