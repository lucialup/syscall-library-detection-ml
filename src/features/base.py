from abc import ABC, abstractmethod
from typing import List

import numpy as np
from scipy.sparse import spmatrix


class FeatureExtractor(ABC):
    @abstractmethod
    def fit(self, data: List) -> "FeatureExtractor":
        pass

    @abstractmethod
    def transform(self, data: List) -> spmatrix | np.ndarray:
        pass

    def fit_transform(self, data: List) -> spmatrix | np.ndarray:
        return self.fit(data).transform(data)

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def n_features(self) -> int:
        pass
