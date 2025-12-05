import argparse
import json
import warnings
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score

from src.config import DataConfig, ModelConfig, TARGET_LIBRARIES
from src.data import load_dataset
from src.data.loader import AppData
from src.models.syscall_ngram import SyscallNgramBaseline

warnings.filterwarnings('ignore')


def run_cv(
    app_data: List[AppData],
    y: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
    **kwargs,
) -> Dict[str, Any]:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    fold_scores = []
    fold_precisions = []
    fold_recalls = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(app_data, y)):
        train_data = [app_data[i] for i in train_idx]
        test_data = [app_data[i] for i in test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if y_test.sum() == 0:
            fold_scores.append(0.0)
            fold_precisions.append(0.0)
            fold_recalls.append(0.0)
            continue

        config = ModelConfig(random_state=random_state + fold)
        model = SyscallNgramBaseline(config=config, **kwargs)
        model.fit(train_data, y_train)

        y_pred = model.predict(test_data)

        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        fold_scores.append(f1)
        fold_precisions.append(prec)
        fold_recalls.append(rec)

        print(f"    Fold {fold + 1}: F1={f1:.3f} P={prec:.3f} R={rec:.3f}")

    return {
        'mean_f1': float(np.mean(fold_scores)) if fold_scores else 0.0,
        'std_f1': float(np.std(fold_scores)) if fold_scores else 0.0,
        'mean_precision': float(np.mean(fold_precisions)) if fold_precisions else 0.0,
        'mean_recall': float(np.mean(fold_recalls)) if fold_recalls else 0.0,
        'fold_scores': fold_scores,
    }


def get_tier(f1: float, std: float) -> str:
    if f1 >= 0.70 and std < 0.15:
        return "TIER 1"
    elif f1 >= 0.40:
        return "TIER 2"
    else:
        return "TIER 3"

"""
A simple baseline using syscall bigrams with bag-of-words and Naive Bayes.
"""
def main():
    parser = argparse.ArgumentParser(description="Syscall N-gram baseline")
    parser.add_argument("--data-dir", type=Path, help="Path to data")
    parser.add_argument("--output", type=Path, default=Path("results/syscall_ngram_output.json"))
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=500)
    parser.add_argument("--libraries", nargs="+", help="Specific libraries to test")
    args = parser.parse_args()

    print("=" * 70)
    print("LIBRARY DETECTION - Syscall N-gram Baseline")
    print("Bag-of-words + Multinomial Naive Bayes")
    print("=" * 70)

    data_config = DataConfig()
    if args.data_dir:
        data_config.data_dir = args.data_dir

    print("\nLoading dataset...")
    app_data = load_dataset(data_config)
    print(f"Loaded {len(app_data)} apps")

    labels_df = pd.DataFrame([d.labels for d in app_data])

    print(f"\nConfig: max_features={args.max_features}, cv_folds={args.cv_folds}")

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 70)

    results = []
    libraries_to_test = args.libraries if args.libraries else list(TARGET_LIBRARIES.keys())

    for library in libraries_to_test:
        if library not in labels_df.columns:
            continue

        y = labels_df[library].values
        support = int(y.sum())

        if support < args.min_support:
            print(f"\n{library}: skipped (support={support})")
            continue

        print(f"\n{library} (n={support}):")
        metrics = run_cv(
            app_data, y,
            n_folds=args.cv_folds,
            max_features=args.max_features,
        )

        tier = get_tier(metrics['mean_f1'], metrics['std_f1'])

        results.append({
            'library': library,
            'support': support,
            'tier': tier,
            **metrics,
        })

        print(f"  -> F1={metrics['mean_f1']:.3f}±{metrics['std_f1']:.3f} "
              f"P={metrics['mean_precision']:.3f} R={metrics['mean_recall']:.3f} [{tier}]")

    if results:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        for tier_name in ["TIER 1", "TIER 2", "TIER 3"]:
            tier_results = [r for r in results if r['tier'] == tier_name]
            if tier_results:
                print(f"\n{tier_name}: {len(tier_results)} libraries")
                for r in sorted(tier_results, key=lambda x: -x['mean_f1']):
                    print(f"  {r['library']:<20} F1={r['mean_f1']:.3f}±{r['std_f1']:.3f}")

        all_f1 = [r['mean_f1'] for r in results]
        print(f"\nOverall: Mean F1={np.mean(all_f1):.3f}, Median F1={np.median(all_f1):.3f}")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({'model': 'SyscallNgram-NB', 'results': results}, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
