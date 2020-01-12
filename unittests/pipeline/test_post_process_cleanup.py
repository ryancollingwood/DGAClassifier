import pytest
import pandas as pd


def test_can_call_post_process_cleanup():
    import src.pipeline
    try:
        src.pipeline.post_process_cleanup()
    except AttributeError:
        pytest.fail("Couldn't call `post_process_cleanup`")
    except Exception:
        pass


def test_post_process_cleanup():
    from src.pipeline import post_process_cleanup

    input_df = pd.DataFrame(
        {
            "name": ["William", "James", "Billy", "", "Corgan"],
        },
        index = [1, 2, 3, 4, 5]
    )

    expected_df = pd.DataFrame(
        {
            "name": ["William", "James", "Billy", "Corgan"]
        },
        index = [1, 2, 3, 5]
    )

    result_df = post_process_cleanup(input_df)

    try:
        assert(result_df.equals(expected_df))
    except AssertionError:
        pytest.fail("`post_process_cleanup` did not remove empty rows as expected")


def test_post_process_no_side_effects():
    from src.pipeline import post_process_cleanup

    input_df = pd.DataFrame(
        {
            "name": ["William", "James", "Billy", "", "Corgan"],
        },
        index = [1, 2, 3, 4, 5]
    )
    copy_input_df = input_df.copy()

    post_process_cleanup(input_df)

    try:
        assert(copy_input_df.equals(input_df))
    except AssertionError:
        pytest.fail("`post_process_cleanup` modified input dataframe in place - introducing side effects")

