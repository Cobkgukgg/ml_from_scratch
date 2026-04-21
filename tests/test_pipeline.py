"""Tests for Pipeline and FeatureUnion."""
import numpy as np
import pytest
from mlengine.pipeline import Pipeline, make_pipeline, FeatureUnion, make_union
from mlengine.preprocessing import StandardScaler
from mlengine.decomposition import PCA
from mlengine.linear import LogisticRegression, LinearRegression


class TestPipeline:
    def test_fit_predict(self, binary_clf_data):
        X, y = binary_clf_data
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(n_iter=200)),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == y.shape

    def test_score(self, binary_clf_data):
        X, y = binary_clf_data
        pipe = make_pipeline(StandardScaler(), LogisticRegression(n_iter=200))
        pipe.fit(X, y)
        assert pipe.score(X, y) > 0.75

    def test_transform_chain(self, binary_clf_data):
        X, _ = binary_clf_data
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=3)),
        ])
        Xt = pipe.fit_transform(X)
        assert Xt.shape == (len(X), 3)

    def test_named_steps(self, binary_clf_data):
        X, y = binary_clf_data
        pipe = make_pipeline(StandardScaler(), LogisticRegression())
        pipe.fit(X, y)
        assert "standardscaler" in pipe.named_steps

    def test_set_params(self, binary_clf_data):
        X, y = binary_clf_data
        pipe = make_pipeline(StandardScaler(), LogisticRegression())
        pipe.set_params(logisticregression__C=10.0)
        assert pipe.named_steps["logisticregression"].C == 10.0

    def test_invalid_intermediate_step(self):
        with pytest.raises(TypeError):
            Pipeline([("bad", LogisticRegression()), ("clf", LogisticRegression())])

    def test_predict_proba(self, binary_clf_data):
        X, y = binary_clf_data
        pipe = make_pipeline(StandardScaler(), LogisticRegression(n_iter=200))
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


class TestFeatureUnion:
    def test_output_shape(self, binary_clf_data):
        X, y = binary_clf_data
        union = make_union(PCA(n_components=3), PCA(n_components=2))
        Xt = union.fit_transform(X)
        assert Xt.shape == (len(X), 5)

    def test_weighted_union(self, binary_clf_data):
        X, _ = binary_clf_data
        union = FeatureUnion(
            [("pca1", PCA(n_components=2)), ("pca2", PCA(n_components=2))],
            weights=[2.0, 1.0],
        )
        Xt = union.fit_transform(X)
        assert Xt.shape[1] == 4
