"""Tests for tree-based models."""
import numpy as np
import pytest
from mlengine.trees import (
    DecisionTree, RandomForest,
    GradientBoostingClassifier, GradientBoostingRegressor,
    XGBoostClassifier,
)


class TestDecisionTree:
    def test_clf_accuracy(self, binary_clf_data):
        X, y = binary_clf_data
        tree = DecisionTree(task="classification", max_depth=5).fit(X, y)
        assert tree.score(X, y) > 0.90

    def test_reg_r2(self, regression_data):
        X, y = regression_data
        from mlengine.preprocessing import StandardScaler
        Xs = StandardScaler().fit_transform(X)
        tree = DecisionTree(task="regression", max_depth=8).fit(Xs, y)
        assert tree.score(Xs, y) > 0.7

    def test_predict_shape(self, binary_clf_data):
        X, y = binary_clf_data
        tree = DecisionTree().fit(X, y)
        assert tree.predict(X).shape == y.shape

    def test_feature_importances_sum_to_one(self, binary_clf_data):
        X, y = binary_clf_data
        tree = DecisionTree(max_depth=4).fit(X, y)
        np.testing.assert_allclose(tree.feature_importances_.sum(), 1.0, atol=1e-6)

    def test_max_depth_zero(self, binary_clf_data):
        X, y = binary_clf_data
        tree = DecisionTree(max_depth=0).fit(X, y)
        preds = tree.predict(X)
        assert len(np.unique(preds)) == 1  # all same prediction


class TestRandomForest:
    def test_clf_accuracy(self, binary_clf_data):
        X, y = binary_clf_data
        rf = RandomForest(n_estimators=20, max_depth=5, random_state=0).fit(X, y)
        assert rf.score(X, y) > 0.88

    def test_predict_proba_valid(self, binary_clf_data):
        X, y = binary_clf_data
        rf = RandomForest(n_estimators=10, random_state=0).fit(X, y)
        proba = rf.predict_proba(X)
        assert proba.shape == (len(y), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_oob_score(self, binary_clf_data):
        X, y = binary_clf_data
        rf = RandomForest(n_estimators=30, oob_score=True, random_state=1).fit(X, y)
        assert hasattr(rf, "oob_score_")
        assert 0 < rf.oob_score_ < 1

    def test_reg_r2(self, regression_data):
        X, y = regression_data
        rf = RandomForest(n_estimators=20, task="regression", max_depth=5, random_state=0)
        rf.fit(X, y)
        assert rf.score(X, y) > 0.85


class TestGradientBoosting:
    def test_clf_accuracy(self, binary_clf_data):
        X, y = binary_clf_data
        gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1,
                                         max_depth=3, random_state=0)
        gb.fit(X, y)
        assert gb.score(X, y) > 0.82

    def test_predict_proba_valid(self, binary_clf_data):
        X, y = binary_clf_data
        gb = GradientBoostingClassifier(n_estimators=20, random_state=0).fit(X, y)
        proba = gb.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_reg_r2(self, regression_data):
        X, y = regression_data
        from mlengine.preprocessing import StandardScaler
        Xs = StandardScaler().fit_transform(X)
        gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                        max_depth=3, random_state=0)
        gb.fit(Xs, y)
        assert gb.score(Xs, y) > 0.75

    def test_loss_decreases(self, binary_clf_data):
        X, y = binary_clf_data
        gb = GradientBoostingClassifier(n_estimators=30, random_state=0).fit(X, y)
        losses = gb.train_losses_
        assert losses[-1] < losses[0]


class TestXGBoost:
    def test_accuracy(self, binary_clf_data):
        X, y = binary_clf_data
        xgb = XGBoostClassifier(n_estimators=30, learning_rate=0.1,
                                 max_depth=3, random_state=0)
        xgb.fit(X, y)
        assert xgb.score(X, y) > 0.85

    def test_predict_proba(self, binary_clf_data):
        X, y = binary_clf_data
        xgb = XGBoostClassifier(n_estimators=10, random_state=0).fit(X, y)
        proba = xgb.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
