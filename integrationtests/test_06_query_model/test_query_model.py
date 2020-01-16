import logging
import pytest
from src.model import QueryModel
from src.logging import  setup_logging

def test_query_model():
    setup_logging(logging.DEBUG)
    logging.debug("\ntest_query_model")

    path = "integrationtests"
    test_model_path = f"{path}/models/trained.model"

    logging.debug(f"test_model_path: {test_model_path}")
    logging.debug("Creating instance of QueryModel")

    query_model = QueryModel(test_model_path, "legit")

    legit_examples = {
        "domain": ["richmondfc", "media-allrecipes", "reddit"]
    }

    legit_result = query_model.predict(legit_examples)

    logging.debug(f"legit_result: {legit_result}")

    try:
        assert legit_result
    except AssertionError:
        message = f"Didn't get expected prediction of all 'True' for: {legit_examples['domain']}"
        logging.exception(message)
        pytest.fail(message)

    dga_examples = {
        "domain": ["cgeoiyxoradbymu", "kbcejbpbduxyxrcqzxlxwdwclrqk", "dgrnntdplbrtg"]
    }

    dga_result = query_model.predict(dga_examples)

    logging.debug(f"dga_result: {dga_result}")

    try:
        assert not dga_result
    except AssertionError:
        message = f"Didn't get expected prediction of all 'False' for: {dga_examples['domain']}"
        logging.exception(message)
        pytest.fail(message)

    mixed_examples = {
        "domain": ["cgeoiyxoradbymu", "shipspotting", "rweulvobduttpzkbxsenfj"]
    }

    mixed_result = query_model.predict(mixed_examples)

    logging.debug(f"mixed_result: {mixed_result}")

    try:
        assert not mixed_result
    except AssertionError:
        message = f"Didn't get expected prediction of `False` for {mixed_examples['domain']}"
        logging.exception(message)
        pytest.fail(message)

