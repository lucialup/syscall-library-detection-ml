from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import spmatrix
from sklearn.metrics import f1_score, precision_recall_curve
from imblearn.ensemble import BalancedRandomForestClassifier

from src.models.base import BaseDetector
from src.config import ModelConfig


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
    """
    Find the optimal classification threshold that maximizes F1 score.

    Returns:
        Tuple of (optimal_threshold, best_f1_score)
    """
    # Get precision-recall curve points
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    # Calculate F1 for each threshold
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * (precision * recall) / (precision + recall),
        0
    )

    # Find best threshold (excluding the last point which has threshold=1)
    if len(thresholds) > 0:
        best_idx = np.argmax(f1_scores[:-1])
        return float(thresholds[best_idx]), float(f1_scores[best_idx])

    return 0.5, 0.0


class BalancedRandomForestDetector(BaseDetector):
    """
    Balanced Random Forest classifier from imbalanced-learn.

    Key features:
    - Random undersampling of majority class in each bootstrap sample
    - Each tree sees balanced data
    - Optional threshold tuning to maximize F1 score

    This addresses the core issue with standard RF on imbalanced data:
    bootstrap samples may contain few/no minority class samples.
    """

    def __init__(
        self,
        config: ModelConfig = None,
        n_estimators: int = 200,
        max_depth: int = 20,
        min_samples_leaf: int = 2,
        max_features: str = "sqrt",
        tune_threshold: bool = True,
        sampling_strategy: str = "auto",
    ):
        config = config or ModelConfig()
        self.model = BalancedRandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            sampling_strategy=sampling_strategy,
            replacement=True,
            bootstrap=True,
            random_state=config.random_state,
            n_jobs=-1,
        )
        self.tune_threshold = tune_threshold
        self.optimal_threshold = 0.5
        self._fitted = False

    def fit(self, X: spmatrix | np.ndarray, y: np.ndarray) -> "BalancedRandomForestDetector":
        self.model.fit(X, y)
        self._fitted = True

        # Tune threshold on training data (in practice, use validation set)
        if self.tune_threshold and hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X)[:, 1]
            self.optimal_threshold, _ = find_optimal_threshold(y, y_proba)

        return self

    def predict(self, X: spmatrix | np.ndarray) -> np.ndarray:
        if self.tune_threshold:
            y_proba = self.model.predict_proba(X)[:, 1]
            return (y_proba >= self.optimal_threshold).astype(int)
        return self.model.predict(X)

    def predict_proba(self, X: spmatrix | np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None,
        top_k: int = 15,
    ) -> List[tuple]:
        if not self._fitted or not hasattr(self.model, "feature_importances_"):
            return []

        importances = self.model.feature_importances_

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        n_features = min(len(importances), len(feature_names))
        indices = np.argsort(importances[:n_features])[-top_k:][::-1]

        return [(feature_names[i], float(importances[i])) for i in indices if importances[i] > 0]

    @property
    def name(self) -> str:
        return "BalancedRandomForest"


class ThresholdTuningWrapper:
    """
    Wrapper that adds threshold tuning to any sklearn-compatible classifier.
    Used for cross-validation where we need a fresh threshold per fold.
    """

    def __init__(self, base_estimator, tune_threshold: bool = True):
        self.base_estimator = base_estimator
        self.tune_threshold = tune_threshold
        self.optimal_threshold = 0.5
        self.classes_ = None

    def fit(self, X, y):
        self.base_estimator.fit(X, y)
        self.classes_ = self.base_estimator.classes_

        if self.tune_threshold:
            y_proba = self.base_estimator.predict_proba(X)[:, 1]
            self.optimal_threshold, _ = find_optimal_threshold(y, y_proba)

        return self

    def predict(self, X):
        if self.tune_threshold:
            y_proba = self.base_estimator.predict_proba(X)[:, 1]
            return (y_proba >= self.optimal_threshold).astype(int)
        return self.base_estimator.predict(X)

    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)

    def get_params(self, deep=True):
        return {
            'base_estimator': self.base_estimator,
            'tune_threshold': self.tune_threshold,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
