from setuptools import setup, find_packages

setup(
    name='gradient_descent_lib',
    version='1.0.0',
    description='Библиотека для реализации и визуализации методов градиентного спуска для линейной регрессии.',
    author='Anton Beschatnov',
    author_email='geizerde@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy>=2.1.3',
        'matplotlib>=3.9.2',
        'scikit-learn>=1.5.2'
    ],
    python_requires='>=3.12',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
