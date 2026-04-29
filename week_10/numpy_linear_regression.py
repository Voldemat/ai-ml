from dataclasses import dataclass
from typing import cast
import numpy as np


def add_intercept_to_inputs(
    inputs: np.typing.NDArray[np.float64],
) -> np.typing.NDArray[np.float64]:
    return np.concatenate((np.ones((inputs.shape[0], 1)), inputs), axis=1)


def predict(
    inputs: np.typing.NDArray[np.float64],
    weights: np.typing.NDArray[np.float64],
) -> np.typing.NDArray[np.float64]:
    return cast(np.typing.NDArray[np.float64], inputs.dot(weights))


def compute_errors(
    prediction: np.typing.NDArray[np.float64],
    reference: np.typing.NDArray[np.float64],
) -> np.typing.NDArray[np.float64]:
    return prediction - reference


def compute_mean_squared_loss(
    errors: np.typing.NDArray[np.float64],
) -> np.typing.NDArray[np.float64]:
    return cast(
        np.typing.NDArray[np.float64], cast(object, np.mean(errors**2) / 2.0)
    )


def compute_gradient(
    number_of_samples: int,
    inputs: np.typing.NDArray[np.float64],
    errors: np.typing.NDArray[np.float64],
) -> np.typing.NDArray[np.float64]:
    return cast(
        np.typing.NDArray[np.float64],
        (1.0 / number_of_samples) * inputs.T.dot(errors),
    )


def compute_new_weights(
    weights: np.typing.NDArray[np.float64],
    gradient: np.typing.NDArray[np.float64],
    learning_rate: float,
) -> np.typing.NDArray[np.float64]:
    print(weights)
    print(gradient)
    return weights - learning_rate * gradient


def should_stop_training(
    loss_history: list[np.typing.NDArray[np.float64]], min_change: float
) -> bool:
    return bool(
        len(loss_history) >= 2
        and abs(loss_history[-1] - loss_history[-2]) < min_change
    )


def train_weights(
    inputs: np.typing.NDArray[np.float64],
    reference: np.typing.NDArray[np.float64],
    initial_weights: np.typing.NDArray[np.float64],
    max_iterations: int,
    learning_rate: float,
    min_change: float,
) -> np.typing.NDArray[np.float64]:
    weights = initial_weights
    loss_history: list[np.typing.NDArray[np.float64]] = []
    for _ in range(max_iterations):
        predictions = predict(inputs, weights)
        errors = predictions - reference
        loss = compute_mean_squared_loss(errors)
        loss_history.append(loss)
        gradient = compute_gradient(cast(int, inputs.shape[0]), inputs, errors)
        weights = compute_new_weights(
            weights, gradient, learning_rate=learning_rate
        )
        if should_stop_training(loss_history, min_change=min_change):
            break
    return weights


def main() -> None:
    initial_inputs = np.linspace(0, 10, 200).reshape(-1, 1)
    test_dataset = np.asarray(
        3 * initial_inputs.squeeze() + 5 + np.random.normal(0, 1, 200),
        dtype=float,
    ).ravel()
    inputs = add_intercept_to_inputs(initial_inputs)
    initial_weights = np.zeros(inputs.shape[1])
    weights = train_weights(
        inputs,
        test_dataset,
        initial_weights,
        max_iterations=3000,
        learning_rate=0.05,
        min_change=1e-10,
    )
    print(predict(inputs, weights))
