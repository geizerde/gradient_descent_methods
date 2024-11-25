from typing import List

def dot_product(a: List[float], b: List[float]) -> float:
    """
    Возвращает скалярное произведение двух векторов a и b.
    """
    return sum(ai * bi for ai, bi in zip(a, b))

def matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """
    Умножение матрицы A на матрицу B (матрицы должны быть совместимы для умножения).
    """
    return [[dot_product(row, col) for col in zip(*B)] for row in A]

def vector_add(v1: List[float], v2: List[float]) -> List[float]:
    """
    Сложение двух векторов v1 и v2.
    """
    return [a + b for a, b in zip(v1, v2)]

def scalar_multiply(scalar: float, vector: List[float]) -> List[float]:
    """
    Умножение вектора на скаляр.
    """
    return [scalar * v for v in vector]

def subtract_vectors(v1: List[float], v2: List[float]) -> List[float]:
    """
    Вычитание двух векторов.
    """
    return [a - b for a, b in zip(v1, v2)]

def vector_magnitude(v: List[float]) -> float:
    """
    Вычисление длины вектора.
    """
    return sum(x ** 2 for x in v) ** 0.5

def elementwise_multiply(v1: List[float], v2: List[float]) -> List[float]:
    """
    Поэлементное умножение двух векторов.
    """
    return [a * b for a, b in zip(v1, v2)]