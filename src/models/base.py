from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
from scipy.sparse import spmatrix


class BaseDetector(ABC):
    @abstractmethod
    def fit(self, X: spmatrix | np.ndarray, y: np.ndarray) -> "BaseDetector":
        pass

    @abstractmethod
    def predict(self, X: spmatrix | np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: spmatrix | np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
