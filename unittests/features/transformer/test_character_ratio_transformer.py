import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline


def test_can_call_character_ratio_transformer():
    import src.features.transformer

    try:
        src.features.transformer.CharacterRatioTransformer()
    except AttributeError:
        pytest.fail("Couldn't create CharacterRatioTransformer")
    except Exception:
        pass


def test_character_ratio_transformer_pass():
    from src.features.transformer import CharacterRatioTransformer

    test_df = pd.DataFrame(
        {
            "name": ["batman", "robin", "joker", "azrael"],
            "alignment": ["good", "good", "bad", "chaotic"],
        })

    expected = pd.DataFrame(
        {
            "alignment": ["good", "good", "bad", "chaotic"],
            "name_vowel_ratio": [0.33, 0.4, 0.4, 0.5],
        }
    ).values

    pipeline = Pipeline([
        (
            "CharacterRatioTransformer",
            CharacterRatioTransformer(
                "vowel_ratio",
                ["name"], ["a", "e", "i", "o", "u"]
            )
        ),
    ])

    result = pipeline.transform(test_df)

    try:
        assert(np.all(expected == result))
    except AssertionError:
        pytest.fail("CharacterRatioTransformer didn't produce expected result")
