"""
mlengine — A production-grade ML engine built from scratch using NumPy.

Modules
-------
linear          : LinearRegression, RidgeRegression, LassoRegression,
                  LogisticRegression, ElasticNet
trees           : DecisionTree, RandomForest,
                  GradientBoostingClassifier/Regressor, XGBoostClassifier
svm             : SVC, SVR
naive_bayes     : GaussianNB, MultinomialNB, BernoulliNB
clustering      : KMeans, MiniBatchKMeans, DBSCAN, GaussianMixture
decomposition   : PCA, TruncatedSVD, NMF, TSNE
preprocessing   : StandardScaler, MinMaxScaler, RobustScaler, Normalizer,
                  LabelEncoder, OneHotEncoder, SimpleImputer, KNNImputer,
                  VarianceThreshold, SelectKBest, RFE, SMOTE
evaluation      : accuracy_score, f1_score, roc_auc_score, cross_val_score,
                  GridSearchCV, KFold, StratifiedKFold, CalibratedClassifier
pipeline        : Pipeline, make_pipeline, FeatureUnion, make_union
optimization    : SGD, Adam, RMSProp, AdaGrad, StepLR, CosineAnnealingLR
"""

__version__ = "0.1.0"
__author__ = "ML Engine Contributors"

from .linear import (
    LinearRegression, RidgeRegression, LassoRegression,
    LogisticRegression, ElasticNet,
)
from .trees import (
    DecisionTree, RandomForest,
    GradientBoostingRegressor, GradientBoostingClassifier,
    XGBoostClassifier,
)
from .svm import SVC, SVR
from .naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from .clustering import KMeans, MiniBatchKMeans, DBSCAN, GaussianMixture
from .decomposition import PCA, TruncatedSVD, NMF, TSNE
from .preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, Normalizer,
    LabelEncoder, OneHotEncoder, OrdinalEncoder,
    SimpleImputer, KNNImputer,
    VarianceThreshold, SelectKBest, RFE,
    SMOTE, RandomOverSampler, RandomUnderSampler,
)
from .evaluation import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, log_loss, confusion_matrix,
    classification_report, mean_squared_error, r2_score,
    KFold, StratifiedKFold, cross_val_score, cross_validate, GridSearchCV,
    CalibratedClassifier,
)
from .pipeline import Pipeline, make_pipeline, FeatureUnion, make_union
from .optimization import SGD, Adam, RMSProp, AdaGrad

__all__ = [
    # linear
    "LinearRegression", "RidgeRegression", "LassoRegression",
    "LogisticRegression", "ElasticNet",
    # trees
    "DecisionTree", "RandomForest",
    "GradientBoostingRegressor", "GradientBoostingClassifier", "XGBoostClassifier",
    # svm
    "SVC", "SVR",
    # naive bayes
    "GaussianNB", "MultinomialNB", "BernoulliNB",
    # clustering
    "KMeans", "MiniBatchKMeans", "DBSCAN", "GaussianMixture",
    # decomposition
    "PCA", "TruncatedSVD", "NMF", "TSNE",
    # preprocessing
    "StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer",
    "LabelEncoder", "OneHotEncoder", "OrdinalEncoder",
    "SimpleImputer", "KNNImputer",
    "VarianceThreshold", "SelectKBest", "RFE",
    "SMOTE", "RandomOverSampler", "RandomUnderSampler",
    # evaluation
    "accuracy_score", "precision_score", "recall_score", "f1_score",
    "roc_auc_score", "roc_curve", "log_loss", "confusion_matrix",
    "classification_report", "mean_squared_error", "r2_score",
    "KFold", "StratifiedKFold", "cross_val_score", "cross_validate", "GridSearchCV",
    "CalibratedClassifier",
    # pipeline
    "Pipeline", "make_pipeline", "FeatureUnion", "make_union",
    # optimization
    "SGD", "Adam", "RMSProp", "AdaGrad",
]
