from typing import Tuple
import numpy as np
from sklearn.pipeline import FeatureUnion

from src.features.transformer import CharacterVarietyRatioTransformer
from src.features.transformer import CharacterDistanceLeftRightTransformer
from src.features.transformer import CharacterRatioTransformer
from src.features.transformer import CountCharacterPairsTransformer
from src.features.transformer import TextLengthTransformer
from src.features import mode


def feature_generation() -> Tuple[str, FeatureUnion]:
    """
    After having pre-processed our inputs generate the features we're looking to use.
    This is returned as a FeatureUnion which should allow for parallel processing.

    :return: Tuple[str, FeatureUnion]
    """
    column = "domain_normed"

    return (
        "feature_generation",
        FeatureUnion([
            (
                "digit_ratio",
                CharacterRatioTransformer(
                    "digit_ratio", [column],
                    [x for x in "1234567890"]
                )
            ),
            (
                "len",
                TextLengthTransformer(
                    "len", [column],
                )
            ),
            (
                "vowel_distance_mode_ratio",
                CharacterDistanceLeftRightTransformer(
                    "vowel_distance_mode_ratio", [column],
                    [x for x in "aeiouy"], mode,
                    True
                )
            ),
            (
                "vowel_distance_std_ratio",
                CharacterDistanceLeftRightTransformer(
                    "vowel_distance_std_ratio", [column],
                    [x for x in "aeiouy"], np.std,
                    True
                )
            ),
            (
                "vowel_ratio",
                CharacterRatioTransformer(
                    "vowel_ratio", [column],
                    [x for x in "aeiouy"]
                )
            ),
            (
                "consonants_variety_ratio",
                CharacterVarietyRatioTransformer(
                    "consonants_variety_ratio", [column],
                    [x for x in "aeiouy"]
                )
            ),
            (
                "character_pairs",
                CountCharacterPairsTransformer(
                    "character_pair", [column],
                    [
                        'an', 'ff', 'ma', 've', 'to', 'me', 'er',
                        'gg', 'we', 'in', '12', 'ar', 'on',
                        'ra', 'st', 'oo', 'li', 'le', 'vv',
                        'te', 'jj', 'en', 'wc', 'al', 'es',
                        'ne', 'ss', 're', 'ct', 'he', 'di',
                        'ti', 'qq', 'pv', '36', 'ga', 'or'
                    ]
                )
            )
        ]),
    )

