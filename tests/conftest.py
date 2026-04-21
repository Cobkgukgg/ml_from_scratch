"""Shared fixtures for all tests."""
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression, make_blobs


@pytest.fixture(scope="session")
def binary_clf_data():
    X, y = make_classification(n_samples=200, n_features=10, n_informative=5,
                                random_state=42)
    return X, y


@pytest.fixture(scope="session")
def multiclass_clf_data():
    X, y = make_classification(n_samples=300, n_features=10, n_classes=3,
                                n_informative=5, n_redundant=2, random_state=42)
    return X, y


@pytest.fixture(scope="session")
def regression_data():
    X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
    return X, y


@pytest.fixture(scope="session")
def blob_data():
    X, y = make_blobs(n_samples=300, centers=3, n_features=4, random_state=42)
    return X, y
