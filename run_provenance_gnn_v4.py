import argparse
import json
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score

from src.config import DataConfig, ModelConfig, TARGET_LIBRARIES
from src.data import load_dataset
from src.data.loader import AppData
from src.models.provenance_gnn_v4 import ProvenanceGNNDetectorV4

warnings.filterwarnings('ignore')


def run_gnn_cv(app_data: List[AppData], y: np.ndarray, data_config: DataConfig,
               n_folds: int = 5, random_state: int = 42, **gnn_params) -> Tuple[float, float, List[float], dict]:
    """Run stratified K-fold cross-validation."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    fold_scores = []
    fold_precisions = []
    fold_recalls = []
    fold_thresholds = []
    indices = np.arange(len(app_data))

    for fold, (train_idx, test_idx) in enumerate(skf.split(indices, y)):
        train_data = [app_data[i] for i in train_idx]
        test_data = [app_data[i] for i in test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if y_test.sum() == 0:
            continue

        config = ModelConfig(random_state=random_state + fold)
        detector = ProvenanceGNNDetectorV4(
            config=config,
            data_config=data_config,
            **gnn_params
        )
        detector.fit(train_data, y_train)

        y_pred = detector.predict(test_data)

        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        fold_scores.append(f1)
        fold_precisions.append(prec)
        fold_recalls.append(rec)
        fold_thresholds.append(detector.optimal_threshold)

        print(f"    Fold {fold + 1}: F1={f1:.3f} P={prec:.3f} R={rec:.3f} thresh={detector.optimal_threshold:.3f}")

    if not fold_scores:
        return 0.0, 0.0, [], {}

    extra_metrics = {
        'mean_precision': float(np.mean(fold_precisions)),
        'mean_recall': float(np.mean(fold_recalls)),
        'mean_threshold': float(np.mean(fold_thresholds)),
        'std_threshold': float(np.std(fold_thresholds)),
    }

    return float(np.mean(fold_scores)), float(np.std(fold_scores)), fold_scores, extra_metrics


def get_tier(f1: float, std: float) -> str:
    if f1 >= 0.70 and std < 0.15:
        return "TIER 1"
    elif f1 >= 0.40:
        return "TIER 2"
    else:
        return "TIER 3"


def main():
    parser = argparse.ArgumentParser(description="Provenance GNN v4 - Production Ready")
    parser.add_argument("--data-dir", type=Path, help="Path to data")
    parser.add_argument("--output", type=Path, default=Path("results/provenance_gnn_v4_output.json"))
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--libraries", nargs="+", help="Specific libraries to test")
    args = parser.parse_args()

    print("=" * 70)
    print("LIBRARY DETECTION - Provenance GNN v4")
    print("=" * 70)

    data_config = DataConfig()
    if args.data_dir:
        data_config.data_dir = args.data_dir

    print("\nLoading dataset...")
    app_data = load_dataset(data_config)
    print(f"Loaded {len(app_data)} apps")

    import pandas as pd
    labels_df = pd.DataFrame([d.labels for d in app_data])

    gnn_params = {
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'patience': args.patience,
        'focal_gamma': args.focal_gamma,
        'val_fraction': args.val_fraction,
    }

    print(f"\nConfig: hidden={args.hidden_dim}, layers={args.num_layers}, "
          f"dropout={args.dropout}, focal_gamma={args.focal_gamma}")

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 70)

    results = []

    libraries_to_test = args.libraries if args.libraries else list(TARGET_LIBRARIES.keys())

    for library in libraries_to_test:
        if library not in labels_df.columns:
            print(f"{library}: not found in dataset")
            continue

        y = labels_df[library].values
        support = int(y.sum())

        if support < args.min_support:
            print(f"\n{library}: skipped (support={support})")
            continue

        print(f"\n{library} (n={support}):")
        mean_f1, std_f1, fold_scores, extra = run_gnn_cv(
            app_data, y,
            data_config=data_config,
            n_folds=args.cv_folds,
            **gnn_params
        )

        tier = get_tier(mean_f1, std_f1)

        results.append({
            'library': library,
            'mean_f1': mean_f1,
            'std_f1': std_f1,
            'mean_precision': extra.get('mean_precision', 0),
            'mean_recall': extra.get('mean_recall', 0),
            'mean_threshold': extra.get('mean_threshold', 0.5),
            'std_threshold': extra.get('std_threshold', 0),
            'support': support,
            'tier': tier,
            'fold_scores': fold_scores,
        })

        print(f"  -> F1={mean_f1:.3f}±{std_f1:.3f} P={extra.get('mean_precision', 0):.3f} "
              f"R={extra.get('mean_recall', 0):.3f} thresh_std={extra.get('std_threshold', 0):.3f} [{tier}]")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output.with_suffix('.partial.json'), 'w') as f:
            json.dump({'model': 'ProvenanceGNN-v4', 'config': gnn_params,
                      'results': results, 'status': 'in_progress'}, f, indent=2)

    # Summary
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

        with open(args.output, 'w') as f:
            json.dump({'model': 'ProvenanceGNN-v4', 'config': gnn_params,
                      'results': results}, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
