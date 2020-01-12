from typing import Tuple
from sklearn.preprocessing import StandardScaler


def pipeline_step_rescale() -> Tuple[str, StandardScaler]:
    return (
        "rescale",
        StandardScaler()
    )
