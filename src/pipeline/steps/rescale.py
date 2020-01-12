from typing import Tuple
from sklearn.preprocessing import StandardScaler


def rescale() -> Tuple[str, StandardScaler]:
    return (
        "rescale",
        StandardScaler()
    )
