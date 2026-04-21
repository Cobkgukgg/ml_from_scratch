"""Tests for SVM."""
import numpy as np
from mlengine.svm import SVC, SVR


class TestSVC:
    def test_binary_accuracy(self):
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=4, random_state=0)
        y = np.where(y == 0, -1, 1)
        svm = SVC(C=1.0, kernel="rbf", gamma=0.5, max_iter=50).fit(X, y)
        acc = svm.score(X, y)
        assert acc > 0.7

    def test_linear_kernel(self):
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=80, n_features=4, random_state=1)
        y = np.where(y == 0, -1, 1)
        svm = SVC(C=1.0, kernel="linear", max_iter=50).fit(X, y)
        assert svm.score(X, y) > 0.65

    def test_predict_labels_are_pm1(self):
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=50, n_features=4, random_state=2)
        y = np.where(y == 0, -1, 1)
        svm = SVC(max_iter=20).fit(X, y)
        preds = svm.predict(X)
        assert set(preds).issubset({-1.0, 1.0})


class TestSVR:
    def test_fit_predict(self):
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=80, n_features=4, noise=0.5, random_state=0)
        svr = SVR(C=1.0, kernel="rbf", gamma=0.1, max_iter=50).fit(X, y)
        preds = svr.predict(X)
        assert preds.shape == y.shape
