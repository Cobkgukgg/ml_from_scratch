"""Placeholder: naive_bayes tests (covered in conftest fixtures)."""
import numpy as np
import pytest
from mlengine.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB


class TestGaussianNB:
    def test_binary_accuracy(self, binary_clf_data):
        X, y = binary_clf_data
        gnb = GaussianNB().fit(X, y)
        assert gnb.score(X, y) > 0.75

    def test_predict_proba_sums_to_one(self, binary_clf_data):
        X, y = binary_clf_data
        gnb = GaussianNB().fit(X, y)
        proba = gnb.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_multiclass(self, multiclass_clf_data):
        X, y = multiclass_clf_data
        gnb = GaussianNB().fit(X, y)
        assert gnb.score(X, y) > 0.5
        assert len(gnb.classes_) == 3


class TestMultinomialNB:
    def test_count_features(self):
        rng = np.random.default_rng(0)
        X = rng.integers(0, 10, size=(100, 20)).astype(float)
        y = rng.integers(0, 3, size=100)
        mnb = MultinomialNB(alpha=1.0).fit(X, y)
        assert mnb.score(X, y) > 0.5

    def test_log_prob_shape(self):
        rng = np.random.default_rng(0)
        X = rng.integers(0, 5, size=(50, 10)).astype(float)
        y = rng.integers(0, 2, size=50)
        mnb = MultinomialNB().fit(X, y)
        lp = mnb.predict_log_proba(X)
        assert lp.shape == (50, 2)


class TestBernoulliNB:
    def test_binary_features(self):
        rng = np.random.default_rng(0)
        X = rng.integers(0, 2, size=(100, 15)).astype(float)
        y = rng.integers(0, 2, size=100)
        bnb = BernoulliNB().fit(X, y)
        assert bnb.score(X, y) > 0.5
