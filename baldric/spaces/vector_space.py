import numpy as np
from .space import Space


class VectorSpace(Space):
    """R^n"""

    def __init__(
        self,
        low: np.ndarray,
        high: np.ndarray,
    ):
        super().__init__(low=low, high=high)

    @staticmethod
    def unit_box(d):
        return VectorSpace(np.zeros(d), np.ones(d))
