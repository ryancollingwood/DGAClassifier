import numpy as np
from scipy import stats

def mode(x: np.array) -> np.float64:
    """
    Helper function to return the mode of a numpy.array.
    Only if there is if the mode result appears more than once
        e.g. [1,1,1,1,1] will return 1
             [1,2,2,3,4] will return 2
             [1,2,3,4,5] will return np.nan

    scipy.stats.mode returns a ModeResult of ( {element}, {n_observations} )

    :param x: numpy.array
    :return: np.float64
    """
    result = stats.mode(x)

    if result[1][0] > 1:
        return result[0][0]
    else:
        # there was no identifiable mode return np.nan
        return np.nan
