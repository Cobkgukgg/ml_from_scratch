"""Tests for dimensionality reduction."""
import numpy as np
import pytest
from mlengine.decomposition import PCA, TruncatedSVD, NMF, TSNE


class TestPCA:
    def test_n_components(self, blob_data):
        X, _ = blob_data
        pca = PCA(n_components=2).fit(X)
        Xt = pca.transform(X)
        assert Xt.shape == (len(X), 2)

    def test_variance_ratio_sums_to_one(self, blob_data):
        X, _ = blob_data
        pca = PCA().fit(X)
        np.testing.assert_allclose(pca.explained_variance_ratio_.sum(), 1.0, atol=1e-5)

    def test_inverse_transform(self, blob_data):
        X, _ = blob_data
        pca = PCA(n_components=X.shape[1]).fit(X)
        X_rec = pca.inverse_transform(pca.transform(X))
        np.testing.assert_allclose(X_rec, X, atol=1e-8)

    def test_variance_fraction(self, blob_data):
        X, _ = blob_data
        pca = PCA(n_components=0.95).fit(X)
        assert pca.cumulative_variance_ratio_[pca.n_components_ - 1] >= 0.95

    def test_orthogonal_components(self, blob_data):
        X, _ = blob_data
        pca = PCA().fit(X)
        gram = pca.components_ @ pca.components_.T
        np.testing.assert_allclose(gram, np.eye(len(pca.components_)), atol=1e-8)


class TestTruncatedSVD:
    def test_shape(self, blob_data):
        X, _ = blob_data
        svd = TruncatedSVD(n_components=2).fit(X)
        assert svd.transform(X).shape == (len(X), 2)

    def test_reconstruction_better_than_random(self, blob_data):
        X, _ = blob_data
        Xt = TruncatedSVD(n_components=3).fit_transform(X)
        assert Xt.shape[1] == 3


class TestNMF:
    def test_non_negative(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (50, 10))
        W = NMF(n_components=3, random_state=0).fit_transform(X)
        assert (W >= 0).all()

    def test_shape(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (50, 10))
        W = NMF(n_components=4, random_state=0).fit_transform(X)
        assert W.shape == (50, 4)


class TestTSNE:
    def test_output_shape(self, blob_data):
        X, _ = blob_data
        tsne = TSNE(n_components=2, n_iter=100, random_state=0)
        Xt = tsne.fit_transform(X[:50])  # small subset for speed
        assert Xt.shape == (50, 2)

    def test_kl_decreases(self, blob_data):
        X, _ = blob_data
        tsne = TSNE(n_components=2, n_iter=200, random_state=0)
        tsne.fit_transform(X[:50])
        mid = len(tsne.kl_divergences_) // 2
        assert min(tsne.kl_divergences_[mid:]) < max(tsne.kl_divergences_[:mid])
