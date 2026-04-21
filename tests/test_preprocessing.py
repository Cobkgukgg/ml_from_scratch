"""Tests for preprocessing."""
import numpy as np
import pytest
from mlengine.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder,
    SimpleImputer, KNNImputer,
    SMOTE, RandomOverSampler, RandomUnderSampler,
    SelectKBest, VarianceThreshold,
)


class TestScalers:
    def test_standard_zero_mean(self, regression_data):
        X, _ = regression_data
        Xt = StandardScaler().fit_transform(X)
        np.testing.assert_allclose(Xt.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(Xt.std(axis=0), 1.0, atol=1e-10)

    def test_minmax_range(self, regression_data):
        X, _ = regression_data
        Xt = MinMaxScaler().fit_transform(X)
        np.testing.assert_allclose(Xt.min(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(Xt.max(axis=0), 1.0, atol=1e-10)

    def test_inverse_transform_standard(self, regression_data):
        X, _ = regression_data
        sc = StandardScaler()
        X_rec = sc.inverse_transform(sc.fit_transform(X))
        np.testing.assert_allclose(X_rec, X, atol=1e-8)

    def test_robust_scaler(self, regression_data):
        X, _ = regression_data
        Xt = RobustScaler().fit_transform(X)
        assert Xt.shape == X.shape


class TestEncoders:
    def test_label_encoder_roundtrip(self):
        y = np.array(["cat", "dog", "cat", "bird"])
        le = LabelEncoder()
        encoded = le.fit_transform(y)
        assert set(encoded) == {0, 1, 2}
        np.testing.assert_array_equal(le.inverse_transform(encoded), y)

    def test_ohe_shape(self):
        X = np.array([["a"], ["b"], ["a"], ["c"]])
        ohe = OneHotEncoder()
        Xt = ohe.fit_transform(X)
        assert Xt.shape == (4, 3)

    def test_ohe_drop_first(self):
        X = np.array([["a"], ["b"], ["c"]])
        Xt = OneHotEncoder(drop="first").fit_transform(X)
        assert Xt.shape == (3, 2)


class TestImputers:
    def test_simple_mean(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 6.0]])
        Xt = SimpleImputer(strategy="mean").fit_transform(X)
        assert not np.isnan(Xt).any()
        assert Xt[0, 1] == pytest.approx(5.0)

    def test_simple_median(self):
        X = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
        Xt = SimpleImputer(strategy="median").fit_transform(X)
        assert Xt[1, 0] == pytest.approx(3.0)

    def test_knn_imputer(self):
        X = np.array([[1.0, 2.0], [1.1, 2.1], [np.nan, 2.05], [10.0, 10.0]])
        ki = KNNImputer(n_neighbors=2)
        Xt = ki.fit_transform(X)
        assert not np.isnan(Xt).any()
        assert abs(Xt[2, 0] - 1.05) < 0.5


class TestImbalanced:
    def _imbalanced(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((150, 5))
        y = np.array([0] * 100 + [1] * 50)
        return X, y

    def test_oversample_balances(self):
        X, y = self._imbalanced()
        X_res, y_res = RandomOverSampler(random_state=0).fit_resample(X, y)
        vals, counts = np.unique(y_res, return_counts=True)
        assert counts[0] == counts[1]

    def test_undersample_balances(self):
        X, y = self._imbalanced()
        X_res, y_res = RandomUnderSampler(random_state=0).fit_resample(X, y)
        vals, counts = np.unique(y_res, return_counts=True)
        assert counts[0] == counts[1]

    def test_smote_balances(self):
        X, y = self._imbalanced()
        X_res, y_res = SMOTE(random_state=0).fit_resample(X, y)
        vals, counts = np.unique(y_res, return_counts=True)
        assert counts[0] == counts[1]

    def test_smote_synthetic_only_minority(self):
        X, y = self._imbalanced()
        X_res, y_res = SMOTE(random_state=0).fit_resample(X, y)
        assert len(X_res) > len(X)


class TestFeatureSelection:
    def test_variance_threshold(self, regression_data):
        X, _ = regression_data
        X_const = np.hstack([X, np.zeros((len(X), 2))])
        vt = VarianceThreshold(threshold=0.0).fit(X_const)
        Xt = vt.transform(X_const)
        assert Xt.shape[1] == X.shape[1]

    def test_select_k_best(self, binary_clf_data):
        X, y = binary_clf_data
        sel = SelectKBest(k=5).fit(X, y)
        Xt = sel.transform(X)
        assert Xt.shape[1] == 5
