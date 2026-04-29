from typing import cast

import numpy as np


def add_bias_term_to_inputs(
    inputs: np.typing.NDArray[np.float64],
) -> np.typing.NDArray[np.float64]:
    return cast(
        np.typing.NDArray[np.float64],
        np.c_[np.ones((inputs.shape[0], 1)), inputs],
    )
