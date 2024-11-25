from typing import List, Tuple
from .matrix_operations import vector_add, scalar_multiply, vector_magnitude, elementwise_multiply, subtract_vectors
from .linear_regression import linear_regression_predict

def sgd(
        X: List[List[float]],
        y: List[float],
        lr: float = 0.01,
        num_epochs: int = 100
) -> Tuple[List[float], float]:
    """
    - X: матрица признаков размерностью (num_samples, num_features)
    - y: целевой вектор размерностью (num_samples,)
    - lr: скорость обучения
    - num_epochs: количество эпох
    """
    num_samples = len(X)
    num_features = len(X[0])
    weights = [0] * num_features
    bias = 0

    for epoch in range(num_epochs):
        total_loss = 0

        for i in range(num_samples):
            y_pred = linear_regression_predict([X[i]], weights, bias)[0]
            error = y_pred - y[i]

            total_loss += error ** 2

            gradients_w = [error * X[i][j] for j in range(num_features)]
            gradients_b = error

            weights = vector_add(weights, scalar_multiply(-lr, gradients_w))
            bias -= lr * gradients_b

            mean_loss = total_loss / num_samples
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {mean_loss}")

    return weights, bias

def nesterov(
        X: List[List[float]],
        y: List[float],
        lr: float = 0.01,
        num_epochs: int = 100,
        momentum: float = 0.9
) -> Tuple[List[float], float]:
    """
    - X: матрица признаков размерностью (num_samples, num_features)
    - y: целевой вектор размерностью (num_samples,)
    - lr: скорость обучения
    - num_epochs: количество эпох
    - momentum: Коэффициент инерции (momentum).
    """
    num_samples = len(X)
    num_features = len(X[0])
    weights = [0.0] * num_features
    bias = 0.0

    velocity_w = [0.0] * num_features
    velocity_b = 0.0

    for epoch in range(num_epochs):
        total_loss = 0

        for i in range(num_samples):
            temp_weights = vector_add(weights, scalar_multiply(momentum, velocity_w))
            temp_bias = bias + (momentum * velocity_b)
            y_pred = linear_regression_predict([X[i]], temp_weights, temp_bias)[0]
            error = y_pred - y[i]

            total_loss += error ** 2

            gradients_w = [error * X[i][j] for j in range(num_features)]
            gradients_b = error

            # Обновление скорости с учетом градиента
            velocity_w = vector_add(scalar_multiply(momentum, velocity_w), scalar_multiply(lr, gradients_w))
            velocity_b = momentum * velocity_b + lr * gradients_b

            weights = subtract_vectors(weights, velocity_w)
            bias -= velocity_b

            mean_loss = total_loss / num_samples
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {mean_loss}")

    return weights, bias

def rmsprop(
        X: List[List[float]],
        y: List[float],
        lr: float = 0.01,
        num_epochs: int = 100,
        beta: float = 0.9,
        epsilon: float = 1e-8
) -> Tuple[List[float], float]:
    """
    - X: матрица признаков размерностью (num_samples, num_features)
    - y: целевой вектор размерностью (num_samples,)
    - lr: скорость обучения
    - num_epochs: количество эпох
    - beta: коэффициент для усреднения квадратов градиентов
    - epsilon: маленькое значение для предотвращения деления на ноль
    """
    num_samples = len(X)
    num_features = len(X[0])
    weights = [0] * num_features
    bias = 0
    Eg_weights = [0] * num_features
    Eg_bias = 0

    for epoch in range(num_epochs):
        total_loss = 0

        for i in range(num_samples):
            y_pred = linear_regression_predict([X[i]], weights, bias)[0]
            error = y_pred - y[i]

            total_loss += error ** 2

            gradients_w = [error * X[i][j] for j in range(num_features)]
            gradients_b = error

            # Обновление усредненных квадратов градиентов
            Eg_weights = vector_add(scalar_multiply(beta, Eg_weights),
                                    scalar_multiply(1 - beta, elementwise_multiply(gradients_w, gradients_w)))
            Eg_bias = beta * Eg_bias + (1 - beta) * gradients_b ** 2

            weights = vector_add(weights, scalar_multiply(
                -lr,
                scalar_multiply(
                    1 / (epsilon + vector_magnitude(Eg_weights)),
                    gradients_w
                    )
                )
            )

            bias -= lr * (gradients_b / (epsilon + Eg_bias ** 0.5))

            mean_loss = total_loss / num_samples
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {mean_loss}")

    return weights, bias

def adam(
        X: List[List[float]],
        y: List[float],
        lr: float = 0.01,
        num_epochs: int = 100,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
) -> Tuple[List[float], float]:
    """
    - X: матрица признаков размерностью (num_samples, num_features)
    - y: целевой вектор размерностью (num_samples,)
    - lr: скорость обучения
    - num_epochs: количество эпох
    - beta1: коэффициент для моментума
    - beta2: коэффициент для усреднения квадратов градиентов
    - epsilon: маленькое значение для предотвращения деления на ноль
    """
    num_samples = len(X)
    num_features = len(X[0])
    weights = [0] * num_features
    bias = 0
    m_weights = [0] * num_features
    v_weights = [0] * num_features
    m_bias = 0
    v_bias = 0

    for epoch in range(1, num_epochs + 1):
        total_loss = 0

        for i in range(num_samples):
            y_pred = linear_regression_predict([X[i]], weights, bias)[0]
            error = y_pred - y[i]

            total_loss += error ** 2

            gradients_w = [error * X[i][j] for j in range(num_features)]
            gradients_b = error

            # Обновление моментума и усреднения квадратов градиентов
            m_weights = vector_add(scalar_multiply(beta1, m_weights), scalar_multiply(1 - beta1, gradients_w))
            v_weights = vector_add(scalar_multiply(beta2, v_weights),
                                   scalar_multiply(1 - beta2, elementwise_multiply(gradients_w, gradients_w)))
            m_bias = beta1 * m_bias + (1 - beta1) * gradients_b
            v_bias = beta2 * v_bias + (1 - beta2) * gradients_b ** 2

            # Коррекция смещения
            m_hat_weights = scalar_multiply(1 / (1 - beta1 ** epoch), m_weights)
            v_hat_weights = scalar_multiply(1 / (1 - beta2 ** epoch), v_weights)
            m_hat_bias = m_bias / (1 - beta1 ** epoch)
            v_hat_bias = v_bias / (1 - beta2 ** epoch)

            weights = vector_add(weights, scalar_multiply(-lr, scalar_multiply(
                1 / (epsilon + vector_magnitude(v_hat_weights)), m_hat_weights)))
            bias -= lr * m_hat_bias / (epsilon + v_hat_bias ** 0.5)

            mean_loss = total_loss / num_samples
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {mean_loss}")

    return weights, bias
