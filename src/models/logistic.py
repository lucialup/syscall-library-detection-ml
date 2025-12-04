from typing import Dict, List, Optional

import numpy as np
from scipy.sparse import spmatrix
from sklearn.linear_model import LogisticRegression

from src.models.base import BaseDetector
from src.config import ModelConfig


class LogisticDetector(BaseDetector):
    def __init__(self, config: ModelConfig = None):
        config = config or ModelConfig()
        self.model = LogisticRegression(
            max_iter=config.max_iter,
            class_weight="balanced",
            solver=config.solver,
            random_state=config.random_state,
        )
        self._fitted = False

    def fit(self, X: spmatrix | np.ndarray, y: np.ndarray) -> "LogisticDetector":
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
        if not self._fitted or not hasattr(self.model, "coef_"):
            return []

        coefs = self.model.coef_[0]

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(coefs))]

        n_features = min(len(coefs), len(feature_names))
        indices = np.argsort(np.abs(coefs[:n_features]))[-top_k:][::-1]

        return [(feature_names[i], float(coefs[i])) for i in indices if coefs[i] != 0]

    @property
    def name(self) -> str:
        return "LogisticRegression"
