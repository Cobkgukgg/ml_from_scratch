# mlengine — ML Engine From Scratch 🧠

A **production-grade machine learning library built entirely from scratch using NumPy**.  
Every algorithm is implemented from first principles — no scikit-learn under the hood.

---

## ✨ Features

| Category | Algorithms |
|---|---|
| **Linear Models** | LinearRegression, RidgeRegression, LassoRegression, LogisticRegression (binary + multinomial), ElasticNet |
| **Tree Models** | DecisionTree (CART), RandomForest (with OOB), GradientBoosting, XGBoost (2nd-order Taylor) |
| **SVM** | SVC (SMO), SVR, kernels: linear, RBF, polynomial, sigmoid |
| **Naive Bayes** | GaussianNB, MultinomialNB, BernoulliNB |
| **Clustering** | KMeans++, MiniBatchKMeans, DBSCAN, GaussianMixture (EM) |
| **Decomposition** | PCA, TruncatedSVD, NMF, t-SNE |
| **Preprocessing** | StandardScaler, MinMaxScaler, RobustScaler, Normalizer, OneHotEncoder, LabelEncoder, SimpleImputer, KNNImputer, SMOTE, RFE, SelectKBest |
| **Evaluation** | accuracy, precision, recall, F1, AUC-ROC, log-loss, MSE, R², KFold, StratifiedKFold, cross_val_score, GridSearchCV, Platt calibration |
| **Pipeline** | Pipeline, FeatureUnion, make_pipeline, make_union |
| **Optimizers** | SGD (Nesterov), Adam, RMSProp, AdaGrad + schedulers (StepLR, Cosine, Warmup) |

---

## 🚀 Quick Start

```bash
pip install -e ".[dev]"
```

```python
import numpy as np
import mlengine as mle
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Simple classifier
clf = mle.LogisticRegression(C=1.0, n_iter=500)
clf.fit(X, y)
print(f"Accuracy: {clf.score(X, y):.3f}")

# Full pipeline with cross-validation
pipe = mle.make_pipeline(
    mle.StandardScaler(),
    mle.PCA(n_components=10),
    mle.RandomForest(n_estimators=100, random_state=0),
)
scores = mle.cross_val_score(pipe, X, y, cv=5)
print(f"CV scores: {scores.mean():.3f} ± {scores.std():.3f}")
```

---

## 📂 Project Structure

```
mlengine/
├── linear/          # Linear & logistic regression, ElasticNet
├── trees/           # DecisionTree, RandomForest, GBDT, XGBoost
├── svm/             # SVC, SVR, kernels
├── naive_bayes/     # GaussianNB, MultinomialNB, BernoulliNB
├── clustering/      # KMeans, DBSCAN, GMM
├── decomposition/   # PCA, SVD, NMF, t-SNE
├── preprocessing/   # Scalers, encoders, imputers, SMOTE
├── evaluation/      # Metrics, cross-validation, calibration
├── pipeline/        # Pipeline, FeatureUnion
├── optimization/    # SGD, Adam, RMSProp, schedulers
└── utils/           # Math, validation, plotting helpers
tests/               # pytest test suite (one file per module)
benchmarks/          # Comparison against scikit-learn
notebooks/           # Demo Jupyter notebooks
docs/                # API reference and math derivations
```

---

## 🧪 Running Tests

```bash
make test          # Run full test suite
make test-cov      # With coverage report
make lint          # flake8 + isort check
```

---

## 📊 Benchmarks

```bash
make benchmark     # Compare accuracy & speed vs sklearn
```

Typical results on standard datasets:

| Model | mlengine acc | sklearn acc | Speed ratio |
|---|---|---|---|
| LogisticRegression | 0.924 | 0.931 | ~3× slower |
| RandomForest (100) | 0.951 | 0.955 | ~8× slower |
| GradientBoosting | 0.947 | 0.952 | ~15× slower |
| GaussianNB | 0.892 | 0.893 | ~1.2× slower |

*Pure NumPy implementations are naturally slower than optimised C extensions.*

---

## 📐 Design Principles

1. **Clarity over cleverness** — code reads like a textbook
2. **sklearn-compatible API** — `fit`, `predict`, `transform`, `score`
3. **No hidden dependencies** — only NumPy + SciPy (optional: matplotlib)
4. **Numerically stable** — careful handling of log(0), division by zero
5. **Tested** — every public class has a pytest test

---

## 📖 References

- Bishop, C. (2006). *Pattern Recognition and Machine Learning*
- Murphy, K. (2012). *Machine Learning: A Probabilistic Perspective*
- Platt, J. (1999). Probabilistic Outputs for SVMs
- Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System
- van der Maaten & Hinton (2008). Visualizing Data using t-SNE
