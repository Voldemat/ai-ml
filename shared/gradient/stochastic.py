import numpy as np

from .computation import compute_gradient

from ..training import ComputeNewWeightsFunction


def compute_new_weights(
    inputs: np.typing.NDArray[np.float64],
    initial_weights: np.typing.NDArray[np.float64],
    outputs: np.typing.NDArray[np.float64],
    learning_rate: float,
    random_generator: np.random.Generator,
) -> np.typing.NDArray[np.float64]:
    m: int = inputs.shape[0]
    random_indices = random_generator.permutation(m)
    working_weights = initial_weights
    random_index: np.int64
    for random_index in random_indices:
        input_array = np.array([inputs[random_index]])
        output_array = np.array([outputs[random_index]])
        gradient = compute_gradient(input_array, working_weights, output_array)
        working_weights -= learning_rate * gradient
    return working_weights


def create_compute_new_weights_function(
    learning_rate: float,
    random_generator: np.random.Generator,
) -> ComputeNewWeightsFunction:
    def wrapper(
        inputs: np.typing.NDArray[np.float64],
        weights: np.typing.NDArray[np.float64],
        outputs: np.typing.NDArray[np.float64],
    ):
        return compute_new_weights(
            inputs,
            weights,
            outputs,
            learning_rate,
            random_generator,
        )

    return wrapper
