from .linear_regression import linear_regression_predict
from .matrix_operations import dot_product, matmul, vector_add, scalar_multiply, subtract_vectors, vector_magnitude, elementwise_multiply
from .data_generation import generate_synthetic_data
from .optimizers import sgd, nesterov, rmsprop, adam
from . import datasets

__version__ = '1.0.0'

__doc__ = """
Эта библиотека включает в себя реализацию методов градиентного спуска (SGD, Nesterov, RMSProp, Adam)
и предназначена для выполнения линейной регрессии на синтетических и реальных данных с n-мерными признаками,
а также визуализации результатов работы (А также много много часов работы :( ).
"""