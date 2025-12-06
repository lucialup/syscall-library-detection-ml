import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DataConfig,
    FeatureConfig,
    ModelConfig,
    EvalConfig,
    TARGET_LIBRARIES,
)
from src.data import load_dataset
from src.features import FeaturePipeline
from src.models import LogisticDetector, RandomForestDetector, BalancedRandomForestDetector
from src.evaluation import run_cross_validation, ResultsReport


MODELS = {
    "logistic": LogisticDetector,
    "random_forest": RandomForestDetector,
    "balanced_rf": BalancedRandomForestDetector,
}

MODEL_NAMES = {
    "logistic": "Logistic Regression",
    "random_forest": "Random Forest",
    "balanced_rf": "Balanced Random Forest",
}


def main():
    parser = argparse.ArgumentParser(
        description="Library Detection from Android Syscall Traces"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Path to ground truth data directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/output.json"),
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=10,
        help="Minimum samples required to evaluate a library",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS.keys()),
        default="logistic",
        help="Model to use for classification",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees for Random Forest",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=20,
        help="Max depth for Random Forest trees",
    )
    args = parser.parse_args()

    model_name = MODEL_NAMES.get(args.model, args.model)
    print("=" * 70)
    print("LIBRARY DETECTION FROM SYSCALL TRACES")
    print(f"Model: {model_name} with TF-IDF Features")
    print("=" * 70)

    data_config = DataConfig()
    if args.data_dir:
        data_config.data_dir = args.data_dir

    feature_config = FeatureConfig()
    model_config = ModelConfig()
    eval_config = EvalConfig(cv_folds=args.cv_folds)

    print("\nLoading dataset...")
    data = load_dataset(data_config)
    print(f"Loaded {len(data)} apps")

    print("\nExtracting features...")
    pipeline = FeaturePipeline(feature_config)
    X = pipeline.fit_transform(data)

    print("\nFeature breakdown:")
    for name, count in pipeline.get_feature_breakdown().items():
        print(f"  {name}: {count}")
    print(f"  Total: {pipeline.n_features}")

    labels_df = pd.DataFrame([d.labels for d in data])
    feature_names = pipeline.get_feature_names()

    report = ResultsReport()

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 70)
    print(f"\n{'Library':<20} {'F1':>8} {'Std':>8} {'Support':>10} {'Tier':>10}")
    print("-" * 60)

    for library in TARGET_LIBRARIES:
        y = labels_df[library].values
        support = int(y.sum())

        if support < args.min_support:
            print(f"{library:<20} {'N/A':>8} {'N/A':>8} {support:>10} {'skip':>10}")
            continue

        if args.model == "random_forest":
            model = RandomForestDetector(
                model_config,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
            )
        elif args.model == "balanced_rf":
            model = BalancedRandomForestDetector(
                model_config,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
            )
        else:
            model = LogisticDetector(model_config)
        metrics = run_cross_validation(model, X, y, eval_config)

        model.fit(X, y)
        top_features = model.get_feature_importance(feature_names)

        report.add_result(library, metrics, top_features)

        print(f"{library:<20} {metrics.mean_f1:>8.3f} {metrics.std_f1:>8.3f} "
              f"{metrics.support:>10} {metrics.tier:>10}")

    report.print_top_features()
    report.print_tier_summary()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.save(args.output)


if __name__ == "__main__":
    main()
