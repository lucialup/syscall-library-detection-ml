import argparse
import json
import logging
from pathlib import Path
from typing import List

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score

from src.config import DataConfig, TARGET_LIBRARIES
from src.data import load_dataset
from src.models.hybrid_cnn_wordlevel import HybridCNNWordLevelDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_hybrid_cnn_cv(
    app_data,
    y: np.ndarray,
    library: str,
    n_folds: int = 5,
    n_repeats: int = 3,
    **cnn_params
) -> dict:
    """Run repeated stratified K-fold cross-validation."""

    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)
    fold_scores = []
    fold_precisions = []
    fold_recalls = []

    total_folds = n_folds * n_repeats
    logger.info(f"Running {n_folds}-fold × {n_repeats}-repeat CV for {library} ({total_folds} total folds)...")

    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(np.arange(len(app_data)), y)):
        y_train, y_test = y[train_idx], y[test_idx]

        if y_test.sum() == 0:
            fold_scores.append(0.0)
            fold_precisions.append(0.0)
            fold_recalls.append(0.0)
            logger.info(f"  Fold {fold_idx+1}/{total_folds}: F1=0.000 (no positives in test)")
            continue

        train_data = [app_data[i] for i in train_idx]
        test_data = [app_data[i] for i in test_idx]

        try:
            detector = HybridCNNWordLevelDetector(**cnn_params)
            detector.fit(train_data, y_train)
            y_pred = detector.predict(test_data)

            f1 = f1_score(y_test, y_pred, zero_division=0)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)

            fold_scores.append(f1)
            fold_precisions.append(prec)
            fold_recalls.append(rec)

            logger.info(f"  Fold {fold_idx+1}/{total_folds}: F1={f1:.3f}, Prec={prec:.3f}, Rec={rec:.3f}")

        except Exception as e:
            logger.error(f"  Fold {fold_idx+1}/{total_folds}: FAILED - {str(e)}")
            fold_scores.append(0.0)
            fold_precisions.append(0.0)
            fold_recalls.append(0.0)

    metrics = {
        'library': library,
        'mean_f1': float(np.mean(fold_scores)),
        'std_f1': float(np.std(fold_scores)),
        'mean_precision': float(np.mean(fold_precisions)),
        'mean_recall': float(np.mean(fold_recalls)),
        'support': int(y.sum()),
        'fold_scores': fold_scores,
    }

    if metrics['mean_f1'] >= 0.70 and metrics['std_f1'] <= 0.15:
        metrics['tier'] = "TIER 1"
    elif metrics['mean_f1'] >= 0.40:
        metrics['tier'] = "TIER 2"
    else:
        metrics['tier'] = "TIER 3"

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Hybrid CNN word-level library detector")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--cv-repeats", type=int, default=3, help="Number of CV repeats")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--vocab-size", type=int, default=2000, help="Max vocabulary size")
    parser.add_argument("--tfidf-dim", type=int, default=500, help="TF-IDF feature dimension")
    parser.add_argument("--embed-dim", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--num-filters", type=int, default=32, help="Number of CNN filters")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--output", type=Path, default=Path("results/hybrid_cnn_wordlevel_output.json"))
    parser.add_argument("--min-support", type=int, default=10, help="Minimum samples per library")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])

    args = parser.parse_args()

    print("=" * 70)
    print("HYBRID CNN WORD-LEVEL LIBRARY DETECTOR")
    print("=" * 70)

    logger.info("Loading dataset...")
    data_config = DataConfig()
    data = load_dataset(data_config)
    logger.info(f"Loaded {len(data)} apps")

    import pandas as pd
    labels_df = pd.DataFrame([d.labels for d in data])

    results = {
        "model": "Hybrid-CNN-WordLevel",
        "config": {
            "cv_folds": args.cv_folds,
            "cv_repeats": args.cv_repeats,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "vocab_size": args.vocab_size,
            "tfidf_dim": args.tfidf_dim,
            "embed_dim": args.embed_dim,
            "num_filters": args.num_filters,
            "learning_rate": args.learning_rate,
            "device": args.device,
        },
        "results": []
    }

    print("==" * 45)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 90)
    print(f"\n{'Library':<20} {'F1':>8} {'Std':>8} {'Support':>10} {'Tier':>10}")
    print("-" * 70)

    total_f1 = 0
    lib_count = 0

    for library in TARGET_LIBRARIES:
        if library not in labels_df.columns:
            continue

        y = labels_df[library].values
        support = int(y.sum())

        if support < args.min_support:
            print(f"{library:<20} {'N/A':>8} {'N/A':>8} {support:>10} {'skip':>10}")
            continue

        logger.info(f"\n{'='*70}")
        logger.info(f"Training: {library} (support={support})")
        logger.info(f"{'='*70}")

        try:
            metrics = run_hybrid_cnn_cv(
                data, y, library,
                n_folds=args.cv_folds,
                n_repeats=args.cv_repeats,
                vocab_size=args.vocab_size,
                tfidf_dim=args.tfidf_dim,
                embed_dim=args.embed_dim,
                num_filters=args.num_filters,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=args.device,
            )

            results['results'].append(metrics)

            print(f"{library:<20} {metrics['mean_f1']:>8.3f} {metrics['std_f1']:>8.3f} "
                  f"{support:>10} {metrics['tier']:>10}")

            total_f1 += metrics['mean_f1']
            lib_count += 1

            # Save partial results
            args.output.parent.mkdir(parents=True, exist_ok=True)
            results['status'] = 'in_progress'
            with open(args.output.with_suffix('.partial.json'), 'w') as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to train {library}: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"{library:<20} {'ERROR':>8} {'ERROR':>8} {support:>10} {'ERROR':>10}")

    print("-" * 70)
    if lib_count > 0:
        avg_f1 = total_f1 / lib_count
        print(f"{'AVERAGE':<20} {avg_f1:>8.3f}")
    print("=" * 70)

    results['status'] = 'completed'
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
