"""Tests for linear models."""
import numpy as np
import pytest
from mlengine.linear import (
    LinearRegression, RidgeRegression, LassoRegression,
    LogisticRegression, ElasticNet,
)


class TestLinearRegression:
    def test_fit_predict_normal(self, regression_data):
        X, y = regression_data
        model = LinearRegression(method="normal")
        model.fit(X, y)
        assert model.score(X, y) > 0.9

    def test_fit_predict_gd(self, regression_data):
        X, y = regression_data
        from sklearn.preprocessing import StandardScaler
        Xs = StandardScaler().fit_transform(X)
        model = LinearRegression(method="gd", lr=0.01, n_iter=2000)
        model.fit(Xs, y)
        assert model.score(Xs, y) > 0.755

    def test_no_intercept(self, regression_data):
        X, y = regression_data
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        assert hasattr(model, "coef_")

    def test_coef_shape(self, regression_data):
        X, y = regression_data
        model = LinearRegression()
        model.fit(X, y)
        assert model.coef_.shape == (X.shape[1],)


class TestRidgeRegression:
    def test_higher_alpha_shrinks_coef(self, regression_data):
        X, y = regression_data
        m1 = RidgeRegression(alpha=0.01).fit(X, y)
        m2 = RidgeRegression(alpha=1000.0).fit(X, y)
        assert np.linalg.norm(m2.coef_) < np.linalg.norm(m1.coef_)

    def test_score_reasonable(self, regression_data):
        X, y = regression_data
        model = RidgeRegression(alpha=1.0).fit(X, y)
        assert model.score(X, y) > 0.85


class TestLassoRegression:
    def test_sparsity(self):
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=200, n_features=20, n_informative=5, noise=0.1, random_state=42)
        model = LassoRegression(alpha=5.0, n_iter=3000).fit(X, y)
        assert np.sum(model.coef_ == 0) > 0

    def test_predict_shape(self, regression_data):
        X, y = regression_data
        model = LassoRegression(alpha=0.01).fit(X, y)
        assert model.predict(X).shape == y.shape


class TestLogisticRegression:
    def test_binary_accuracy(self, binary_clf_data):
        X, y = binary_clf_data
        from mlengine.preprocessing import StandardScaler
        Xs = StandardScaler().fit_transform(X)
        model = LogisticRegression(C=1.0, n_iter=500)
        model.fit(Xs, y)
        assert model.score(Xs, y) > 0.75

    def test_predict_proba_sums_to_one(self, binary_clf_data):
        X, y = binary_clf_data
        model = LogisticRegression().fit(X, y)
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_multiclass_softmax(self, multiclass_clf_data):
        X, y = multiclass_clf_data
        from mlengine.preprocessing import StandardScaler
        Xs = StandardScaler().fit_transform(X)
        model = LogisticRegression(multi_class="softmax", n_iter=300)
        model.fit(Xs, y)
        assert model.score(Xs, y) > 0.60

    def test_classes_attribute(self, binary_clf_data):
        X, y = binary_clf_data
        model = LogisticRegression().fit(X, y)
        assert len(model.classes_) == 2


class TestElasticNet:
    def test_fit_predict(self, regression_data):
        X, y = regression_data
        model = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape

    def test_pure_ridge(self, regression_data):
        X, y = regression_data
        model = ElasticNet(alpha=0.1, l1_ratio=0.0).fit(X, y)
        assert np.sum(model.coef_ == 0) == 0  # no sparsity with l1_ratio=0
