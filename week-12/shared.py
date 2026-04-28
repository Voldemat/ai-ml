import time
from dataclasses import dataclass
from typing import Callable, cast
import numpy as np


def compute_gradient(
    inputs: np.typing.NDArray[np.float64],
    weights: np.typing.NDArray[np.float64],
    outputs: np.typing.NDArray[np.float64],
) -> np.typing.NDArray[np.float64]:
    m: int = inputs.shape[0]
    errors = inputs @ weights - outputs
    return (inputs.transpose() @ errors) / m


def compute_new_weights_using_batch_gradient_descent(
    inputs: np.typing.NDArray[np.float64],
    initial_weights: np.typing.NDArray[np.float64],
    outputs: np.typing.NDArray[np.float64],
    learning_rate: float,
) -> np.typing.NDArray[np.float64]:
    gradient = compute_gradient(inputs, initial_weights, outputs)
    return initial_weights - learning_rate * gradient


def compute_new_weights_using_stochastic_gradient_descent(
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


def compute_new_weights_using_mini_batch_gradient_descent(
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


ComputeNewWeightsFunction = Callable[
    [
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
    ],
    np.typing.NDArray[np.float64],
]


def create_batch_gradient_descent_compute_new_weights_function(
    learning_rate: float,
) -> ComputeNewWeightsFunction:
    def wrapper(
        inputs: np.typing.NDArray[np.float64],
        weights: np.typing.NDArray[np.float64],
        outputs: np.typing.NDArray[np.float64],
    ):
        return compute_new_weights_using_batch_gradient_descent(
            inputs, weights, outputs, learning_rate
        )

    return wrapper


def create_mini_batch_gradient_descent_compute_new_weights_function(
    learning_rate: float, random_generator: np.random.Generator, batch_size: int
) -> ComputeNewWeightsFunction:
    def wrapper(
        inputs: np.typing.NDArray[np.float64],
        weights: np.typing.NDArray[np.float64],
        outputs: np.typing.NDArray[np.float64],
    ):
        return compute_new_weights_using_mini_batch_gradient_descent(
            inputs,
            weights,
            outputs,
            learning_rate,
            random_generator,
            batch_size,
        )

    return wrapper


def create_stochastic_gradient_descent_compute_new_weights_function(
    learning_rate: float,
    random_generator: np.random.Generator,
) -> ComputeNewWeightsFunction:
    def wrapper(
        inputs: np.typing.NDArray[np.float64],
        weights: np.typing.NDArray[np.float64],
        outputs: np.typing.NDArray[np.float64],
    ):
        return compute_new_weights_using_stochastic_gradient_descent(
            inputs,
            weights,
            outputs,
            learning_rate,
            random_generator,
        )

    return wrapper


def compute_mean_squared_loss(
    errors: np.typing.NDArray[np.float64],
) -> float:
    return float(np.mean(errors**2))


@dataclass
class TrainingResult:
    weights: np.typing.NDArray[np.float64]
    loss_history: list[float]
    training_time: float


def train_weights(
    inputs: np.typing.NDArray[np.float64],
    initial_weights: np.typing.NDArray[np.float64],
    outputs: np.typing.NDArray[np.float64],
    epochs: int,
    compute_new_weights_function: ComputeNewWeightsFunction,
) -> TrainingResult:
    assert epochs > 0
    start = time.perf_counter()
    loss_history: list[float] = []
    working_weights = initial_weights
    for _ in range(epochs):
        working_weights = compute_new_weights_function(
            inputs, working_weights, outputs
        )
        outputs_prediction = inputs @ working_weights
        mean_squared_loss = compute_mean_squared_loss(
            outputs - outputs_prediction
        )
        loss_history.append(mean_squared_loss)
    end = time.perf_counter()
    return TrainingResult(
        weights=working_weights,
        loss_history=loss_history,
        training_time=end - start,
    )


def add_bias_term_to_inputs(
    inputs: np.typing.NDArray[np.float64],
) -> np.typing.NDArray[np.float64]:
    return cast(
        np.typing.NDArray[np.float64],
        np.c_[np.ones((inputs.shape[0], 1)), inputs],
    )
