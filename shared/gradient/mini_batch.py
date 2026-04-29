import numpy as np

from .computation import compute_gradient

from ..training import ComputeNewWeightsFunction


def compute_new_weights(
    inputs: np.typing.NDArray[np.float64],
    initial_weights: np.typing.NDArray[np.float64],
    outputs: np.typing.NDArray[np.float64],
    learning_rate: float,
    random_generator: np.random.Generator,
    batch_size: int,
) -> np.typing.NDArray[np.float64]:
    m: int = inputs.shape[0]
    random_indexes = random_generator.permutation(m)
    working_weights = initial_weights
    for start_index in range(0, m, batch_size):
        batch_indexes = random_indexes[start_index : start_index + batch_size]
        batch_inputs = inputs[batch_indexes]
        batch_outputs = outputs[batch_indexes]
        gradient = compute_gradient(
            batch_inputs, working_weights, batch_outputs
        )
        working_weights -= learning_rate * gradient
    return working_weights


def create_compute_new_weights_function(
    learning_rate: float, random_generator: np.random.Generator, batch_size: int
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
            batch_size,
        )

    return wrapper
