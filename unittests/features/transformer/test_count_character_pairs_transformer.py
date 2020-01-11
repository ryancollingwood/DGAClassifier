import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def test_can_call_count_character_pairs_transformer():
    import src.features.transformer

    try:
        src.features.transformer.CountCharacterPairsTransformer()
    except AttributeError:
        pytest.fail("Couldn't create instance of `CountCharacterPairsTransformer`")
    except Exception:
        pass


def test_character_count_character_pairs_transformer_pass():
    from src.features.transformer import CountCharacterPairsTransformer

    test_df = pd.DataFrame(
        {
            "name": ["batman", "robin", "joker"],
            "alignment": ["good", "good", "bad"],
        })

    expected = pd.DataFrame(
        {
            "name_count_pair_ba": np.array([1, 0, 0], np.int64),
            "alignment_count_pair_ba": np.array([0, 0, 1], np.int64),
        }
    ).values

    pipeline = Pipeline([
        (
            "CountCharacterPairsTransformer",
            CountCharacterPairsTransformer(
                "count_pair", ["name", "alignment"],
                ["ba"]
            )
        ),
    ])

    result = pipeline.transform(test_df)

    try:
        assert(np.all(result == expected))
    except AssertionError:
        pytest.fail("CountCharacterPairsTransformer transform did not produce expected result")
