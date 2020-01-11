import pytest


def test_can_call_normalise_text_to_only_regex_matches():
    try:
        import src.preprocessing.text
        src.preprocessing.text.normalise_text_to_only_regex_matches()
    except AttributeError:
        pytest.fail("Couldn't call normalise_text_to_only_regex_matches")
    except Exception:
        pass
