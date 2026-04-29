import numpy as np

def compute_gradient(
    inputs: np.typing.NDArray[np.float64],
    weights: np.typing.NDArray[np.float64],
    outputs: np.typing.NDArray[np.float64],
) -> np.typing.NDArray[np.float64]:
    m: int = inputs.shape[0]
    errors = inputs @ weights - outputs
    return (inputs.transpose() @ errors) / m
