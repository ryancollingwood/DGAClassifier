import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline


def test_can_call_character_distance_left_right_transformer():
    import src.features.transformer

    try:
        src.features.transformer.CharacterDistanceLeftRightTransformer()
    except AttributeError:
        pytest.fail("Couldn't create instance of `CharacterDistanceLeftRightTransformer`")
    except Exception:
        pass


def test_character_distance_left_right_transformer_no_agg_callback_pass():
    from src.features.transformer import CharacterDistanceLeftRightTransformer

    test_df = pd.DataFrame(
        {
            "name": ["batman", "robin", "joker", "azrael"],
            "alignment": ["good", "good", "bad", "chaotic"],
        })

    expected = pd.DataFrame(
        {
            "alignment": ["good", "good", "bad", "chaotic"],
            "name_vowel_distance": [
                np.array([1, 0, 2, 1, 0, 1], np.int64),
                np.array([1, 0, 1, 0, 1], np.int64),
                np.array([1, 0, 1, 0, 1], np.int64),
                np.array([0, 2, 1, 0, 0, 1], np.int64),
            ]
        }
    ).values

    pipeline = Pipeline([
        (
            "CharacterDistanceLeftRightTransformer",
            CharacterDistanceLeftRightTransformer(
                "vowel_distance",
                ["name"], ["a", "e", "i", "o", "u"]
            )
        ),
    ])

    result = pipeline.transform(test_df)

    try:
        # converting to str because of deprecation warning
        # "deprecationwarning: elementwise == comparison failed; this will raise an error in the future."
        assert (str(result) == str(expected))
    except AssertionError:
        pytest.fail("CharacterDistanceLeftRightTransformer with no agg callback didn't produce expected result")


def test_character_distance_left_right_transformer_max_agg_callback_pass():
    from src.features.transformer import CharacterDistanceLeftRightTransformer

    test_df = pd.DataFrame(
        {
            "name": ["batman", "robin", "joker", "azrael"],
            "alignment": ["good", "good", "bad", "chaotic"],
        })

    expected = pd.DataFrame(
        {
            "alignment": ["good", "good", "bad", "chaotic"],
            "name_vowel_distance_max": [2, 1, 1, 2],
        }
    ).values

    pipeline = Pipeline([
        (
            "CharacterDistanceLeftRightTransformer",
            CharacterDistanceLeftRightTransformer(
                "vowel_distance_max",
                ["name"], ["a", "e", "i", "o", "u"], np.max
            )
        ),
    ])

    result = pipeline.transform(test_df)

    try:
        assert (np.all(expected == result))
    except AssertionError:
        pytest.fail("CharacterDistanceLeftRightTransformer with `np.max` agg callback didn't produce expected result")


def test_character_distance_left_right_transformer_mode_agg_callback_pass():
    from src.features.transformer import CharacterDistanceLeftRightTransformer
    from src.features.character_distance_left_right import get_mode

    test_df = pd.DataFrame(
        {
            "name": ["batman", "robin", "joker", "azrael"],
            "alignment": ["good", "good", "bad", "chaotic"],
        })

    expected = pd.DataFrame(
        {
            "alignment": ["good", "good", "bad", "chaotic"],
            "name_vowel_distance_mode": [1, 1, 1, 0],
        }
    ).values

    pipeline = Pipeline([
        (
            "CharacterDistanceLeftRightTransformer",
            CharacterDistanceLeftRightTransformer(
                "vowel_distance_mode",
                ["name"], ["a", "e", "i", "o", "u"], get_mode
            )
        ),
    ])

    result = pipeline.transform(test_df)

    try:
        assert(np.all(result == expected))
    except AssertionError:
        pytest.fail("CharacterDistanceLeftRightTransformer with `get_mode` agg callback didn't produce expected result")
