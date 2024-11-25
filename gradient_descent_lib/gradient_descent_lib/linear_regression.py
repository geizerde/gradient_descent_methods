from typing import List

from .matrix_operations import dot_product

def linear_regression_predict(
        X: List[List[float]],
        weights: List[float],
        bias: float
) -> List[float]:
    """
    Выполняет предсказание для линейной регрессии.
    """
    return [dot_product(row, weights) + bias for row in X]