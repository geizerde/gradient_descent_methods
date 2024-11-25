import numpy as np
from sklearn.datasets import make_regression

from Beschastnov_A_A.lab1.gradient_descent_lib.gradient_descent_lib import generate_synthetic_data
from Beschastnov_A_A.lab1.gradient_descent_lib.gradient_descent_lib.datasets import dataset_2_features, \
    dataset_3_features, dataset_10_features
from Beschastnov_A_A.lab1.gradient_descent_lib.gradient_descent_lib.utils import visualize

"""
    Пример с рандомно сгенерированными данными
"""

# X, y, _, _ = generate_synthetic_data(num_samples=100, num_features=7)
#
# visualize(X, y, method='sgd', lr=0.01, num_epochs=100)
# visualize(X, y, method='nesterov', lr=0.01, num_epochs=100)
# visualize(X, y, method='rmsprop', lr=0.01, num_epochs=100)
# visualize(X, y, method='adam', lr=0.01, num_epochs=100)

"""
    Пример с двумя переменными
"""

# X = dataset_2_features.X
# y = dataset_2_features.y
#
# visualize(X, y, method='sgd', lr=0.01, num_epochs=8)
# visualize(X, y, method='nesterov', lr=0.01, num_epochs=2, momentum=0.9)
# visualize(X, y, method='rmsprop', lr=0.5, num_epochs=65, beta=0.9)
# visualize(X, y, method='adam', lr=0.1, num_epochs=80,  beta1= 0.99, beta2= 0.45)

"""
    Пример с тремя переменными
"""

# X = dataset_3_features.X
# y = dataset_3_features.y
#
# visualize(X, y, method='sgd', lr=0.01, num_epochs=100)
# visualize(X, y, method='nesterov', lr=0.01, num_epochs=100, momentum=0.9)
# visualize(X, y, method='rmsprop', lr=0.5, num_epochs=350, beta=0.9)
# visualize(X, y, method='adam', lr=0.1, num_epochs=100,  beta1= 0.99, beta2= 0.40)

"""
    Пример с 10-ю переменными
"""

# X = dataset_10_features.X
# y = dataset_10_features.y
#
# visualize(X, y, method='sgd', lr=0.01, num_epochs=100)
# visualize(X, y, method='nesterov', lr=0.01, num_epochs=100, momentum=0.9)
# visualize(X, y, method='rmsprop', lr=0.5, num_epochs=350, beta=0.9)
# visualize(X, y, method='adam', lr=0.1, num_epochs=100,  beta1= 0.99, beta2= 0.40)


"""
    Пример генерации линейного датасета
"""

# X, y = make_regression(n_samples=100, n_features=10, noise=0.1)
#
# X_str = np.array2string(X, separator=',')
# y_str = np.array2string(y, separator=',')
#
#
# print("X =", X_str)
# print("y =", y_str)