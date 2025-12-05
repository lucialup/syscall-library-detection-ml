"""
Syscall N-gram baseline for library detection.

A simple baseline using syscall bigrams with bag-of-words (CountVectorizer)
and Multinomial Naive Bayes classifier.

This is a classic NLP sequence classification approach applied to syscall logs.
"""

from typing import List, Optional, Tuple
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

from src.models.base import BaseDetector
from src.config import ModelConfig
from src.data.loader import AppData


class SyscallNgramBaseline(BaseDetector):
    """
    Syscall N-gram baseline - classic sequence classification approach.

    Uses syscall bigrams (e.g., "openat_read", "read_close") with
    simple bag-of-words (counts) and Multinomial Naive Bayes classifier.

    This is the simplest meaningful NLP baseline - no TF-IDF, just raw counts.
    """

    def __init__(
        self,
        config: ModelConfig = None,
        max_features: int = 500,
        ngram_range: Tuple[int, int] = (1, 2),
    ):
        config = config or ModelConfig()
        self.random_state = config.random_state
        self.max_features = max_features
        self.ngram_range = ngram_range

        self.vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95,
        )
        self.model = MultinomialNB(alpha=1.0)
        self._fitted = False
        self.classes_ = np.array([0, 1])

    def _extract_text(self, app_data_list: List[AppData]) -> List[str]:
        """Extract syscall bigrams as text."""
        return [app.syscall_bigrams for app in app_data_list]

    def fit(self, X: List[AppData], y: np.ndarray) -> "SyscallNgramBaseline":
        texts = self._extract_text(X)
        features = self.vectorizer.fit_transform(texts)
        self.model.fit(features, y)
        self._fitted = True
        return self

    def predict(self, X: List[AppData]) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: List[AppData]) -> np.ndarray:
        if not self._fitted:
            return np.zeros((len(X), 2))

        texts = self._extract_text(X)
        features = self.vectorizer.transform(texts)
        return self.model.predict_proba(features)

    def get_feature_importance(self, feature_names=None, top_k=15):
        if not self._fitted:
            return []

        feature_names = self.vectorizer.get_feature_names_out()
        log_prob_diff = self.model.feature_log_prob_[1] - self.model.feature_log_prob_[0]
        indices = np.argsort(np.abs(log_prob_diff))[-top_k:][::-1]
        return [(feature_names[i], float(log_prob_diff[i])) for i in indices]

    @property
    def name(self) -> str:
        return "SyscallNgram-NB"
