from unidecode import unidecode
import re


def normalise_text_to_ascii(text: str) -> str:
    """
    For the given text return an ASCII-fied version of it, dropping characters with accents
    and other Unicode embellishments and replacing them with ASCII compliant characters

    :param text: str
    :return: str
    """
    result = unidecode(text)
    # Not all characters are resolved by `unicode` e.g. Romanian S Comma - https://en.wikipedia.org/wiki/S-comma
    result = result.encode('ascii', 'ignore').decode('ascii')
    return result


def get_regex_for_az_digits_underscores() -> re.Pattern:
    """
    Return a compiled Regex Pattern that will match:
        - lower cased letter `a-z`, does not account for accented characters
        - digits `0-9`, does not account for `.`
        - hyphens `-`

    :return: re.Pattern
    """
    return re.compile(r"[a-z\d\-]")


def get_regex_for_non_whitespace() -> re.Pattern:
    """
    Return a compiled Regex Pattern that will match characters that are NOT:
        - Whitespace
        - Braille Pattern Blank - https://www.compart.com/en/unicode/U+2800
    :return:
    """
    # fun fact `\s` this won't work for all "whitespace" characters
    # there are non-printable characters like braille whitespace that will still make it through
    # spent hours once trying to figure that one out
    return re.compile(r"[^\xA0\s]")


def normalise_text_to_only_regex_matches(text: str, matcher: re.Pattern) -> str:
    """
    Filter out the characters in the text that do not match the given matcher

    :param text: str
    :param matcher: re.Pattern
    :return: str
    """
    return "".join(matcher.findall(text))
