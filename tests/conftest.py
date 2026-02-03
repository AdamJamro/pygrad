"""
@Author Adam Jamroziński

"""

from typing import Callable

import numpy as np
from numpy import allclose


def numeric_derivative(
    func: Callable[[np.ndarray], np.ndarray], x: np.ndarray, eps: float = 1e-6
) -> float:
    """Compute the numeric derivative of a function at a given point using central difference."""
    # remark that this is not even a real derivative formula
    return (func(x + eps) - func(x - eps)) / (2 * eps)


def does_not_contribute(value) -> None:
    """
    The variable graph actually splits into two cases when gradient is zero:
    - an actuall zero if it was used in computing the top node
    - None if it didn't
    """
    return value is None or allclose(value, zeros_like(value))
