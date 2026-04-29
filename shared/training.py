import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .loss import compute_mean_squared_loss

ComputeNewWeightsFunction = Callable[
    [
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
    ],
    np.typing.NDArray[np.float64],
]


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
            test_dataset=outputs, predicted_dataset=outputs_prediction
        )
        loss_history.append(mean_squared_loss)
    end = time.perf_counter()
    return TrainingResult(
        weights=working_weights,
        loss_history=loss_history,
        training_time=end - start,
    )
