"""Tests for evaluation metrics and cross-validation."""
import numpy as np
import pytest
from mlengine.evaluation import (
    accuracy_score, confusion_matrix, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss, mean_squared_error, r2_score,
    KFold, StratifiedKFold, cross_val_score, GridSearchCV,
)
from mlengine.linear import LogisticRegression, LinearRegression


class TestClassificationMetrics:
    def test_accuracy(self):
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        assert accuracy_score(y_true, y_pred) == pytest.approx(0.8)

    def test_confusion_matrix_binary(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        cm = confusion_matrix(y_true, y_pred)
        assert cm.shape == (2, 2)
        assert cm.sum() == 4

    def test_perfect_f1(self):
        y = np.array([0, 0, 1, 1])
        assert f1_score(y, y) == pytest.approx(1.0)

    def test_roc_auc_perfect(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        assert roc_auc_score(y_true, y_score) == pytest.approx(1.0)

    def test_roc_auc_random(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 100)
        y_score = rng.uniform(0, 1, 100)
        auc = roc_auc_score(y_true, y_score)
        assert 0.3 < auc < 0.7  # roughly random

    def test_log_loss_perfect(self):
        y_true = np.array([0, 1])
        y_prob = np.array([0.0001, 0.9999])
        assert log_loss(y_true, y_prob) < 0.01


class TestRegressionMetrics:
    def test_mse_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mean_squared_error(y, y) == pytest.approx(0.0)

    def test_r2_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert r2_score(y, y) == pytest.approx(1.0)

    def test_r2_constant_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        y_pred = np.full_like(y, y.mean())
        assert r2_score(y, y_pred) == pytest.approx(0.0, abs=1e-10)


class TestCrossValidation:
    def test_kfold_splits(self, binary_clf_data):
        X, y = binary_clf_data
        cv = KFold(n_splits=5)
        splits = list(cv.split(X))
        assert len(splits) == 5
        sizes = [len(v) for _, v in splits]
        assert abs(max(sizes) - min(sizes)) <= 1

    def test_stratified_kfold(self, binary_clf_data):
        X, y = binary_clf_data
        cv = StratifiedKFold(n_splits=5)
        for train_idx, val_idx in cv.split(X, y):
            train_ratio = y[train_idx].mean()
            val_ratio = y[val_idx].mean()
            assert abs(train_ratio - val_ratio) < 0.15

    def test_cross_val_score_shape(self, binary_clf_data):
        X, y = binary_clf_data
        model = LogisticRegression(n_iter=200)
        scores = cross_val_score(model, X, y, cv=3)
        assert len(scores) == 3
        assert all(0 < s < 1 for s in scores)


class TestGridSearchCV:
    def test_finds_best_params(self, binary_clf_data):
        X, y = binary_clf_data
        model = LogisticRegression(n_iter=200)
        grid = GridSearchCV(model, param_grid={"C": [0.1, 1.0, 10.0]}, cv=3)
        grid.fit(X, y)
        assert "C" in grid.best_params_
        assert grid.best_score_ > 0.7
        assert hasattr(grid, "best_estimator_")

    def test_all_param_combos_evaluated(self, binary_clf_data):
        X, y = binary_clf_data
        model = LogisticRegression(n_iter=100)
        grid = GridSearchCV(model, param_grid={"C": [0.1, 1.0], "lr": [0.05, 0.1]}, cv=2)
        grid.fit(X, y)
        assert len(grid.cv_results_) == 4  # 2 x 2
