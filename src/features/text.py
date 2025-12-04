from typing import List, Callable

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix

from src.features.base import FeatureExtractor
from src.data.loader import AppData
from src.config import FeatureConfig


class TfidfFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        field: str,
        prefix: str,
        max_features: int,
        ngram_range: tuple = (1, 1),
        min_df: int = 1,
        max_df: float = 1.0,
    ):
        self.field = field
        self.prefix = prefix
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
        )
        self._fitted = False

    def _get_texts(self, data: List[AppData]) -> List[str]:
        return [getattr(d, self.field) for d in data]

    def fit(self, data: List[AppData]) -> "TfidfFeatureExtractor":
        self.vectorizer.fit(self._get_texts(data))
        self._fitted = True
        return self

    def transform(self, data: List[AppData]) -> spmatrix:
        return self.vectorizer.transform(self._get_texts(data))

    def get_feature_names(self) -> List[str]:
        return [f"{self.prefix}:{name}" for name in self.vectorizer.get_feature_names_out()]

    @property
    def n_features(self) -> int:
        if not self._fitted:
            return 0
        return len(self.vectorizer.get_feature_names_out())


class PathFeatureExtractor(TfidfFeatureExtractor):
    def __init__(self, config: FeatureConfig):
        super().__init__(
            field="paths",
            prefix="path",
            max_features=config.path_max_features,
            ngram_range=config.path_ngram_range,
            min_df=config.path_min_df,
            max_df=config.path_max_df,
        )


class ThreadFeatureExtractor(TfidfFeatureExtractor):
    def __init__(self, config: FeatureConfig):
        super().__init__(
            field="threads",
            prefix="thread",
            max_features=config.thread_max_features,
            ngram_range=(1, 1),
            min_df=config.thread_min_df,
        )


class ThreadSyscallFeatureExtractor(TfidfFeatureExtractor):
    def __init__(self, config: FeatureConfig):
        super().__init__(
            field="thread_syscalls",
            prefix="thread_syscall",
            max_features=config.thread_syscall_max_features,
            ngram_range=(1, 1),
            min_df=config.thread_syscall_min_df,
        )


class SyscallBigramFeatureExtractor(TfidfFeatureExtractor):
    def __init__(self, config: FeatureConfig):
        super().__init__(
            field="syscall_bigrams",
            prefix="syscall_bigram",
            max_features=config.syscall_bigram_max_features,
            ngram_range=(1, 1),
            min_df=config.syscall_bigram_min_df,
        )


class PathSyscallFeatureExtractor(TfidfFeatureExtractor):
    def __init__(self, config: FeatureConfig):
        super().__init__(
            field="path_syscalls",
            prefix="path_syscall",
            max_features=config.path_syscall_max_features,
            ngram_range=(1, 1),
            min_df=config.path_syscall_min_df,
        )
