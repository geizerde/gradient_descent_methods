from typing import List, Any, Tuple
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from .linear_regression import linear_regression_predict
from .optimizers import sgd, nesterov, rmsprop, adam

def train_linear_regression(
        X: List[List[float]],
        y: List[float],
        method: str = 'sgd',
        **kwargs: Any
) -> Tuple[List[float], float]:
    """
    Обучение линейной регрессии с использованием выбранного алгоритма оптимизации.
    """
    match method:
        case 'sgd':
            weights, bias = sgd(X, y, **kwargs)
        case 'nesterov':
            weights, bias = nesterov(X, y, **kwargs)
        case 'rmsprop':
            weights, bias = rmsprop(X, y, **kwargs)
        case 'adam':
            weights, bias = adam(X, y, **kwargs)
        case _:
            raise ValueError('Неизвестный метод оптимизации.')

    return weights, bias

def visualize(
        X: List[List[float]],
        y: List[float],
        method: str = 'sgd',
        **kwargs: Any
) -> None:
    weights, bias = train_linear_regression(X, y, method, **kwargs)

    pca = PCA(n_components=1)
    X_reduced = pca.fit_transform(X)
    y_pred = linear_regression_predict(X, weights, bias)

    plt.scatter(X_reduced, y, color='blue', label='Истинные значения')
    plt.scatter(X_reduced, y_pred, color='red', label='Предсказанные значения')
    plt.xlabel('Компонента (PCA)')
    plt.ylabel('Значение целевой переменной')
    plt.legend()
    plt.title(f'Результаты с использованием {method}')
    plt.show()
