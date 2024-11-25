import numpy as np
from typing import Tuple

def generate_synthetic_data(
        num_samples: int = 100,
        num_features: int = 3,
        noise: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    true_weights = np.random.randn(num_features)
    bias = np.random.randn()

    X = np.random.rand(num_samples, num_features)
    y = X.dot(true_weights) + bias + noise * np.random.randn(num_samples)

    return X, y, true_weights, bias