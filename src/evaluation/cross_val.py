from typing import Callable

import numpy as np
from scipy.sparse import spmatrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer

from src.models.base import BaseDetector
from src.evaluation.metrics import CVMetrics
from src.config import EvalConfig


def run_cross_validation(
    model: BaseDetector,
    X: spmatrix | np.ndarray,
    y: np.ndarray,
    config: EvalConfig = None,
) -> CVMetrics:
    config = config or EvalConfig()

    cv = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    scorer = make_scorer(f1_score, zero_division=0)

    scores = cross_val_score(model.model, X, y, cv=cv, scoring=scorer)

    return CVMetrics(
        mean_f1=float(scores.mean()),
        std_f1=float(scores.std()),
        scores=scores.tolist(),
        support=int(y.sum()),
    )
