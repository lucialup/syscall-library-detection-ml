from typing import List

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler

from src.features.base import FeatureExtractor
from src.data.loader import AppData


class StatisticalFeatureExtractor(FeatureExtractor):
    def __init__(self):
        self.scaler = StandardScaler()
        self._feature_names: List[str] = []
        self._fitted = False

    def _extract_features(self, data: List[AppData]) -> np.ndarray:
        if not data:
            return np.array([])

        self._feature_names = sorted(data[0].counts.keys())

        features = np.array([
            [d.counts[k] for k in self._feature_names]
            for d in data
        ])
        return features

    def fit(self, data: List[AppData]) -> "StatisticalFeatureExtractor":
        features = self._extract_features(data)
        self.scaler.fit(features)
        self._fitted = True
        return self

    def transform(self, data: List[AppData]) -> csr_matrix:
        features = self._extract_features(data)
        scaled = self.scaler.transform(features)
        return csr_matrix(scaled)

    def get_feature_names(self) -> List[str]:
        return [f"stat:{name}" for name in self._feature_names]

    @property
    def n_features(self) -> int:
        return len(self._feature_names)
