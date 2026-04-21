"""
Benchmark mlengine against scikit-learn on standard datasets.
Measures accuracy, training time, and prediction time.
"""

import time
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

import mlengine as mle

try:
    from sklearn.linear_model import LogisticRegression as SKLogistic
    from sklearn.linear_model import Ridge as SKRidge
    from sklearn.ensemble import RandomForestClassifier as SKForest
    from sklearn.ensemble import GradientBoostingClassifier as SKGB
    from sklearn.naive_bayes import GaussianNB as SKGNB
    from sklearn.svm import SVC as SKSVC
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    print("scikit-learn not installed — skipping comparison benchmarks.")


def benchmark(name_a, model_a, name_b, model_b, X_tr, X_te, y_tr, y_te):
    results = []
    for name, model in [(name_a, model_a), (name_b, model_b)]:
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        fit_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        score = model.score(X_te, y_te)
        pred_time = time.perf_counter() - t0
        results.append((name, score, fit_time, pred_time))
    print(f"\n{'Model':40s} {'Score':>8} {'Fit (s)':>10} {'Pred (s)':>10}")
    print("-" * 72)
    for name, score, fit, pred in results:
        print(f"{name:40s} {score:>8.4f} {fit:>10.3f} {pred:>10.4f}")
    speed = results[0][2] / (results[1][2] + 1e-9)
    print(f"  → mlengine is {speed:.1f}× vs sklearn (fit time)")


def main():
    print("=" * 72)
    print("mlengine vs scikit-learn Benchmark")
    print("=" * 72)

    # Classification dataset
    X, y = make_classification(n_samples=2000, n_features=20,
                                n_informative=10, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    from mlengine.preprocessing import StandardScaler
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    if _HAS_SKLEARN:
        print("\n[Logistic Regression]")
        benchmark(
            "mlengine.LogisticRegression",
            mle.LogisticRegression(C=1.0, n_iter=500),
            "sklearn.LogisticRegression",
            SKLogistic(C=1.0, max_iter=500, random_state=42),
            X_tr_s, X_te_s, y_tr, y_te,
        )

        print("\n[Random Forest (50 trees)]")
        benchmark(
            "mlengine.RandomForest",
            mle.RandomForest(n_estimators=50, max_depth=5, random_state=0),
            "sklearn.RandomForestClassifier",
            SKForest(n_estimators=50, max_depth=5, random_state=0),
            X_tr, X_te, y_tr, y_te,
        )

        print("\n[Gaussian Naive Bayes]")
        benchmark(
            "mlengine.GaussianNB",
            mle.GaussianNB(),
            "sklearn.GaussianNB",
            SKGNB(),
            X_tr_s, X_te_s, y_tr, y_te,
        )

    # Regression
    X_r, y_r = make_regression(n_samples=1000, n_features=20, noise=0.5, random_state=42)
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

    if _HAS_SKLEARN:
        print("\n[Ridge Regression]")
        benchmark(
            "mlengine.RidgeRegression",
            mle.RidgeRegression(alpha=1.0),
            "sklearn.Ridge",
            SKRidge(alpha=1.0),
            X_tr_r, X_te_r, y_tr_r, y_te_r,
        )

    print("\n" + "=" * 72)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
