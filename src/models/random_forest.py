from typing import List, Optional

import numpy as np
from scipy.sparse import spmatrix
from sklearn.ensemble import RandomForestClassifier

from src.models.base import BaseDetector
from src.config import ModelConfig


class RandomForestDetector(BaseDetector):
    def __init__(
        self,
        config: ModelConfig = None,
        n_estimators: int = 200,
        max_depth: int = 20,
        min_samples_leaf: int = 3,
        max_features: str = "sqrt",
    ):
        config = config or ModelConfig()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=6,
            max_features=max_features,
            class_weight="balanced_subsample",
            bootstrap=True,
            oob_score=True,
            random_state=config.random_state,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, X: spmatrix | np.ndarray, y: np.ndarray) -> "RandomForestDetector":
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: spmatrix | np.ndarray) -> np.ndarray:
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
        return "RandomForest"
