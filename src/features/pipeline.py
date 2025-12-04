from typing import List, Dict

from scipy.sparse import hstack, spmatrix

from src.features.base import FeatureExtractor
from src.features.text import (
    PathFeatureExtractor,
    ThreadFeatureExtractor,
    ThreadSyscallFeatureExtractor,
    SyscallBigramFeatureExtractor,
    PathSyscallFeatureExtractor,
)
from src.features.statistical import StatisticalFeatureExtractor
from src.data.loader import AppData
from src.config import FeatureConfig


class FeaturePipeline:
    def __init__(self, config: FeatureConfig = None):
        config = config or FeatureConfig()

        self.extractors: Dict[str, FeatureExtractor] = {
            "path": PathFeatureExtractor(config),
            "thread": ThreadFeatureExtractor(config),
            "thread_syscall": ThreadSyscallFeatureExtractor(config),
            "syscall_bigram": SyscallBigramFeatureExtractor(config),
            "path_syscall": PathSyscallFeatureExtractor(config),
            "statistical": StatisticalFeatureExtractor(),
        }
        self._fitted = False

    def fit(self, data: List[AppData]) -> "FeaturePipeline":
        for extractor in self.extractors.values():
            extractor.fit(data)
        self._fitted = True
        return self

    def transform(self, data: List[AppData]) -> spmatrix:
        matrices = [ext.transform(data) for ext in self.extractors.values()]
        return hstack(matrices)

    def fit_transform(self, data: List[AppData]) -> spmatrix:
        return self.fit(data).transform(data)

    def get_feature_names(self) -> List[str]:
        names = []
        for extractor in self.extractors.values():
            names.extend(extractor.get_feature_names())
        return names

    def get_feature_breakdown(self) -> Dict[str, int]:
        return {name: ext.n_features for name, ext in self.extractors.items()}

    @property
    def n_features(self) -> int:
        return sum(ext.n_features for ext in self.extractors.values())
