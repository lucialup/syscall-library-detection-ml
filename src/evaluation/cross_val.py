from typing import List

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from src.evaluation.metrics import CVMetrics
from src.config import EvalConfig, FeatureConfig
from src.features import FeaturePipeline


def run_cross_validation(
    model_class: type,
    data: List,
    y: np.ndarray,
    feature_config: FeatureConfig,
    model_config,
    eval_config: EvalConfig = None,
    model_kwargs: dict = None,
) -> CVMetrics:
    eval_config = eval_config or EvalConfig()
    model_kwargs = model_kwargs or {}

    cv = StratifiedKFold(
        n_splits=eval_config.cv_folds,
        shuffle=True,
        random_state=eval_config.random_state,
    )

    fold_scores = []

    for train_idx, test_idx in cv.split(data, y):
        train_data = [data[i] for i in train_idx]
        test_data = [data[i] for i in test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline = FeaturePipeline(feature_config)
        X_train = pipeline.fit_transform(train_data)

        X_test = pipeline.transform(test_data)

        model = model_class(model_config, **model_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        fold_scores.append(f1)

    return CVMetrics(
        mean_f1=float(np.mean(fold_scores)),
        std_f1=float(np.std(fold_scores)),
        scores=fold_scores,
        support=int(y.sum()),
    )
