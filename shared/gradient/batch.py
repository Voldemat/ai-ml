import numpy as np

from .computation import compute_gradient

from ..training import ComputeNewWeightsFunction


def compute_new_weights(
    inputs: np.typing.NDArray[np.float64],
    initial_weights: np.typing.NDArray[np.float64],
    outputs: np.typing.NDArray[np.float64],
    learning_rate: float,
) -> np.typing.NDArray[np.float64]:
    gradient = compute_gradient(inputs, initial_weights, outputs)
    return initial_weights - learning_rate * gradient


def create_compute_new_weights_function(
    learning_rate: float,
) -> ComputeNewWeightsFunction:
    def wrapper(
        inputs: np.typing.NDArray[np.float64],
        weights: np.typing.NDArray[np.float64],
        outputs: np.typing.NDArray[np.float64],
    ):
        return compute_new_weights(
            inputs, weights, outputs, learning_rate
        )

    return wrapper
