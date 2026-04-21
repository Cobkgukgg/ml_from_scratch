"""Tests for clustering algorithms."""
import numpy as np
import pytest
from mlengine.clustering import KMeans, MiniBatchKMeans, DBSCAN, GaussianMixture


class TestKMeans:
    def test_correct_n_clusters(self, blob_data):
        X, y = blob_data
        km = KMeans(n_clusters=3, random_state=0).fit(X)
        assert len(np.unique(km.labels_)) == 3

    def test_inertia_decreases_with_more_iters(self, blob_data):
        X, _ = blob_data
        km1 = KMeans(n_clusters=3, max_iter=1, n_init=1, random_state=0).fit(X)
        km2 = KMeans(n_clusters=3, max_iter=300, n_init=1, random_state=0).fit(X)
        assert km2.inertia_ <= km1.inertia_

    def test_predict_consistency(self, blob_data):
        X, _ = blob_data
        km = KMeans(n_clusters=3, random_state=0).fit(X)
        preds = km.predict(X)
        np.testing.assert_array_equal(preds, km.labels_)

    def test_centers_shape(self, blob_data):
        X, _ = blob_data
        km = KMeans(n_clusters=3, random_state=0).fit(X)
        assert km.cluster_centers_.shape == (3, X.shape[1])

    def test_kmeanspp_better_than_random(self, blob_data):
        X, _ = blob_data
        inertias_pp, inertias_rand = [], []
        for seed in range(5):
            km_pp = KMeans(n_clusters=3, init="kmeans++", n_init=1, random_state=seed).fit(X)
            km_rand = KMeans(n_clusters=3, init="random", n_init=1, random_state=seed).fit(X)
            inertias_pp.append(km_pp.inertia_)
            inertias_rand.append(km_rand.inertia_)
        assert np.mean(inertias_pp) <= np.mean(inertias_rand)


class TestDBSCAN:
    def test_finds_clusters(self, blob_data):
        X, _ = blob_data
        db = DBSCAN(eps=1.5, min_samples=5).fit(X)
        n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        assert n_clusters >= 2

    def test_noise_label(self):
        X = np.array([[0, 0], [0, 0.1], [100, 100]])  # last point is outlier
        db = DBSCAN(eps=0.5, min_samples=2).fit(X)
        assert db.labels_[-1] == -1


class TestGaussianMixture:
    def test_n_components(self, blob_data):
        X, _ = blob_data
        gmm = GaussianMixture(n_components=3, random_state=0).fit(X)
        assert gmm.weights_.shape == (3,)
        np.testing.assert_allclose(gmm.weights_.sum(), 1.0, atol=1e-5)

    def test_predict_proba_shape(self, blob_data):
        X, _ = blob_data
        gmm = GaussianMixture(n_components=3, random_state=0).fit(X)
        proba = gmm.predict_proba(X)
        assert proba.shape == (len(X), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_sample(self, blob_data):
        X, _ = blob_data
        gmm = GaussianMixture(n_components=3, random_state=0).fit(X)
        X_new, labels = gmm.sample(50)
        assert X_new.shape == (50, X.shape[1])
        assert set(labels).issubset({0, 1, 2})
