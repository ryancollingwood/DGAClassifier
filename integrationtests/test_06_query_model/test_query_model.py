import pytest
from src.model import QueryModel


def test_query_model():
    print("\ntest_query_model")
    path = "integrationtests"

    test_model_path = f"{path}/models/trained.model"

    query_model = QueryModel(test_model_path, "legit")

    legit_examples = {
        "domain": ["richmondfc", "media-allrecipes", "reddit"]
    }

    legit_result = query_model.predict(legit_examples)

    try:
        assert legit_result
    except AssertionError:
        pytest.fail(f"Didn't get expected prediction of all 'True' for: {legit_examples['domain']}")

    dga_examples = {
        "domain": ["cgeoiyxoradbymu", "kbcejbpbduxyxrcqzxlxwdwclrqk", "dgrnntdplbrtg"]
    }

    dga_result = query_model.predict(dga_examples)

    try:
        assert not dga_result
    except AssertionError:
        pytest.fail(f"Didn't get expected prediction of all 'False' for: {dga_examples['domain']}")

    mixed_examples = {
        "domain": ["cgeoiyxoradbymu", "shipspotting", "rweulvobduttpzkbxsenfj"]
    }

    mixed_result = query_model.predict(mixed_examples)

    try:
        assert not mixed_result
    except AssertionError:
        pytest.fail(f"Didn't get expected prediction of `False` for {mixed_examples['domain']}")

