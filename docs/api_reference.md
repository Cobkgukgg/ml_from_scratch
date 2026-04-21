# API Reference

## mlengine.linear

### `LinearRegression(method='normal', lr=1e-3, n_iter=1000, fit_intercept=True)`
OLS linear regression. `method='normal'` uses the closed-form normal equation; `method='gd'` uses gradient descent.

**Methods:** `fit(X, y)`, `predict(X)`, `score(X, y)` → R²

---

### `RidgeRegression(alpha=1.0, fit_intercept=True)`
L2-regularised regression. Solved analytically via `(XᵀX + αI)⁻¹Xᵀy`.

---

### `LassoRegression(alpha=1.0, n_iter=1000, tol=1e-4)`
L1-regularised regression via coordinate descent with soft-thresholding.

---

### `LogisticRegression(C=1.0, multi_class='ovr', lr=0.1, n_iter=1000)`
Binary (`multi_class='ovr'`) or multinomial (`multi_class='softmax'`) logistic regression.  
**Extra methods:** `predict_proba(X)`, `decision_function(X)`

---

### `ElasticNet(alpha=1.0, l1_ratio=0.5, n_iter=1000)`
Combined L1+L2 penalty via coordinate descent.

---

## mlengine.trees

### `DecisionTree(task, criterion, max_depth, min_samples_split, min_samples_leaf, max_features)`
CART tree for classification (`criterion='gini'|'entropy'`) or regression (`criterion='mse'`).  
**Properties:** `feature_importances_`

---

### `RandomForest(n_estimators=100, task, max_depth, max_features='sqrt', bootstrap=True, oob_score=False)`
Bagged ensemble of decision trees.  
**Extra methods:** `predict_proba(X)`  
**Properties:** `oob_score_`, `feature_importances_`

---

### `GradientBoostingClassifier / GradientBoostingRegressor`
`(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0)`  
Additive model using pseudo-residuals (MSE for regression, log-loss for classification).

---

### `XGBoostClassifier(n_estimators, learning_rate, max_depth, reg_lambda, gamma, colsample)`
Second-order Taylor expansion boosting with gain-based split criterion.

---

## mlengine.svm

### `SVC(C=1.0, kernel='rbf', gamma=1.0, tol=1e-3, max_iter=200)`
Binary SVM via Sequential Minimal Optimisation (SMO). Labels must be ±1.  
Kernels: `'linear'`, `'rbf'`, `'poly'`, `'sigmoid'`

---

### `SVR(C=1.0, epsilon=0.1, kernel='rbf', gamma=1.0)`
Support Vector Regression with ε-insensitive loss.

---

## mlengine.naive_bayes

### `GaussianNB(var_smoothing=1e-9)`
Assumes Gaussian feature distributions per class.

### `MultinomialNB(alpha=1.0)`
For count/frequency features (e.g. bag-of-words).

### `BernoulliNB(alpha=1.0, binarize=0.0)`
For binary features.

---

## mlengine.clustering

### `KMeans(n_clusters=8, init='kmeans++', n_init=10, max_iter=300)`
K-Means++ initialisation for better convergence.

### `MiniBatchKMeans(n_clusters, batch_size=256, max_iter=100)`
Faster variant using stochastic mini-batch updates.

### `DBSCAN(eps=0.5, min_samples=5, metric='euclidean')`
Density-based clustering. Label `-1` = noise.

### `GaussianMixture(n_components=1, covariance_type='full', max_iter=100)`
EM algorithm. `covariance_type`: `'full'`, `'diag'`, `'spherical'`  
**Extra:** `sample(n)` → (X, labels)

---

## mlengine.decomposition

### `PCA(n_components, whiten=False)`
`n_components` can be int, float (variance fraction), or `None`.  
**Properties:** `explained_variance_ratio_`, `cumulative_variance_ratio_`

### `TruncatedSVD(n_components, n_iter=5)`
Randomized SVD — works on non-centred matrices (good for text/sparse data).

### `NMF(n_components, max_iter=200)`
Non-negative factorisation via multiplicative updates.

### `TSNE(n_components=2, perplexity=30, n_iter=1000)`
Exact t-SNE with perplexity-based bandwidth search.

---

## mlengine.preprocessing

| Class | Parameters |
|---|---|
| `StandardScaler` | `with_mean`, `with_std` |
| `MinMaxScaler` | `feature_range=(0,1)` |
| `RobustScaler` | `quantile_range=(25,75)` |
| `Normalizer` | `norm='l2'|'l1'|'max'` |
| `LabelEncoder` | — |
| `OneHotEncoder` | `drop=None|'first'`, `handle_unknown` |
| `OrdinalEncoder` | — |
| `TargetEncoder` | `smoothing=10` |
| `SimpleImputer` | `strategy='mean'|'median'|'most_frequent'|'constant'` |
| `KNNImputer` | `n_neighbors=5` |
| `VarianceThreshold` | `threshold=0.0` |
| `SelectKBest` | `score_func`, `k=10` |
| `RFE` | `estimator`, `n_features_to_select`, `step` |
| `SMOTE` | `k_neighbors=5` |
| `RandomOverSampler` | — |
| `RandomUnderSampler` | — |

---

## mlengine.evaluation

### Metrics
`accuracy_score`, `precision_score`, `recall_score`, `f1_score(average='binary'|'macro'|'weighted'|'none')`,
`roc_auc_score`, `roc_curve`, `precision_recall_curve`, `log_loss`, `matthews_corrcoef`,
`confusion_matrix`, `classification_report`,
`mean_squared_error(squared=True)`, `root_mean_squared_error`,
`mean_absolute_error`, `r2_score`, `huber_loss`, `explained_variance_score`

### Cross-Validation
`KFold(n_splits, shuffle, random_state)`, `StratifiedKFold`, `LeaveOneOut`  
`cross_val_score(estimator, X, y, cv, scoring)` → ndarray of scores  
`cross_validate(estimator, X, y, cv, scoring)` → dict  
`GridSearchCV(estimator, param_grid, cv, scoring, refit)` → `.best_params_`, `.best_score_`, `.best_estimator_`

### Calibration
`CalibratedClassifier(base_estimator, method='platt'|'isotonic', cv=5)`  
`calibration_curve(y_true, y_prob, n_bins=10)` → (fraction_of_positives, mean_predicted)  
`expected_calibration_error(y_true, y_prob)` → float

---

## mlengine.pipeline

### `Pipeline(steps: list[tuple[str, estimator]])`
Chain transformers + final estimator.  
`make_pipeline(*steps)` — auto-names from class names.

### `FeatureUnion(transformer_list, weights=None)`
Run transformers in parallel and concatenate outputs.  
`make_union(*transformers)` — convenience constructor.

---

## mlengine.optimization

### Optimizers
`SGD(lr, momentum, nesterov, weight_decay)`,
`Adam(lr, beta1, beta2, eps, weight_decay)`,
`RMSProp(lr, decay, eps)`,
`AdaGrad(lr, eps)`

All optimizers: `.step(params_dict, grads_dict)` → updated params dict.

### Schedulers
`StepLR(optimizer, step_size, gamma)`,
`ExponentialLR(optimizer, gamma)`,
`CosineAnnealingLR(optimizer, T_max, eta_min)`,
`WarmupScheduler(optimizer, warmup_steps, after_scheduler)`
