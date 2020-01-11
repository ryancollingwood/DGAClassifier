from itertools import islice


def window(seq, n=2):
    """
    For an iterable (str, list, etc.) return tuples of windows of size `n`.

    As found on https://docs.python.org/release/2.3.5/lib/itertools-example.html

    :param seq: iterable
    :param n: int size of the window
    :return: generator
    """

    if len(seq) == 0:
        raise ValueError("Cannot create window: empty sequence provided")

    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
