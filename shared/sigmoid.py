import numpy as np

def compute_sigmoid(
    z: np.typing.NDArray[np.float64]
) -> np.typing.NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-z))
