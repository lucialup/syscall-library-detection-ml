from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


@dataclass
class ClassificationMetrics:
    f1: float
    precision: float
    recall: float
    accuracy: float

    @classmethod
    def from_predictions(cls, y_true: np.ndarray, y_pred: np.ndarray) -> "ClassificationMetrics":
        return cls(
            f1=f1_score(y_true, y_pred, zero_division=0),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            accuracy=accuracy_score(y_true, y_pred),
        )


@dataclass
class CVMetrics:
    mean_f1: float
    std_f1: float
    scores: List[float]
    support: int

    @property
    def tier(self) -> str:
        if self.mean_f1 >= 0.70 and self.std_f1 < 0.15:
            return "TIER 1"
        if self.mean_f1 >= 0.40:
            return "TIER 2"
        return "TIER 3"
