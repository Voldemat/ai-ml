from typing import Any, cast
import numpy as np

from sklearn.model_selection import train_test_split  # pyright: ignore [reportUnknownVariableType,reportMissingTypeStubs]

random_generator = np.random.default_rng(42)
inputs = np.linspace(0, 10, 200).reshape(-1, 1)
outputs = np.sin(inputs).ravel() + random_generator.normal(
    scale=0.2, size=inputs.shape[0]
)

inputs_train, inputs_test, outputs_train, outputs_test = cast(
    tuple[
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
    ],
    cast(
        Any,  # pyright: ignore[reportExplicitAny]
        train_test_split(inputs, outputs, test_size=0.2, random_state=42),
    ),
)

sort_idx = np.argsort(inputs_test.ravel())
inputs_test_sorted = inputs_test[sort_idx]
outputs_test_sorted = outputs_test[sort_idx]
