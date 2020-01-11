import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def test_can_call_binarize_character_pairs_transformer():
    import src.features.transformer

    try:
        src.features.transformer.BinarizeCharacterPairsTransformer()
    except AttributeError:
        pytest.fail("Couldn't create instance of `BinarizeCharacterPairsTransformer`")
    except Exception:
        pass


def test_binarize_character_pairs_transformer():
    from src.features.transformer import BinarizeCharacterPairsTransformer

    test_df = pd.DataFrame(
        {
            "name": ["batman", "robin", "joker"],
            "alignment": ["good", "good", "bad"],
        })

    expected = pd.DataFrame(
        {
            "name_pairs_ba": np.array([1, 0, 0], np.int32),
            "name_pairs_oo": np.array([0, 0, 0], np.int32),
            "alignment_pairs_ba": np.array([0, 0, 1], np.int32),
            "alignment_pairs_oo": np.array([1, 1, 0], np.int32),

        }
    ).values

    pipeline = Pipeline([
        (
            "BinarizeCharacterPairsTransformer",
            BinarizeCharacterPairsTransformer(
                "pairs", ["name", "alignment"],
                ["ba", "oo"]
            )
        ),
    ])

    result = pipeline.transform(test_df)

    try:
        assert(np.all(result == expected))
    except AssertionError:
        pytest.fail("BinarizeCharacterPairsTransformer transform did not produce expected result")
