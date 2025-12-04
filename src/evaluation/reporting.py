import json
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

from src.evaluation.metrics import CVMetrics


@dataclass
class LibraryResult:
    library: str
    mean_f1: float
    std_f1: float
    support: int
    tier: str
    top_features: List[tuple] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["top_features"]:
            d["top_features"] = [(n, float(c)) for n, c in d["top_features"]]
        return d


class ResultsReport:
    def __init__(self):
        self.results: List[LibraryResult] = []

    def add_result(
        self,
        library: str,
        metrics: CVMetrics,
        top_features: List[tuple] = None,
    ):
        self.results.append(LibraryResult(
            library=library,
            mean_f1=metrics.mean_f1,
            std_f1=metrics.std_f1,
            support=metrics.support,
            tier=metrics.tier,
            top_features=top_features,
        ))

    def print_summary(self):
        print(f"\n{'Library':<20} {'F1':>8} {'Std':>8} {'Support':>10} {'Tier':>10}")
        print("-" * 60)

        for r in sorted(self.results, key=lambda x: -x.mean_f1):
            print(f"{r.library:<20} {r.mean_f1:>8.3f} {r.std_f1:>8.3f} {r.support:>10} {r.tier:>10}")

    def print_tier_summary(self):
        print("\n" + "=" * 60)
        print("TIER SUMMARY")
        print("=" * 60)

        for tier_name in ["TIER 1", "TIER 2", "TIER 3"]:
            tier_libs = [r for r in self.results if r.tier == tier_name]
            print(f"\n{tier_name}: {len(tier_libs)} libraries")
            for r in sorted(tier_libs, key=lambda x: -x.mean_f1):
                print(f"  {r.library:<20} F1={r.mean_f1:.3f}")

    def print_top_features(self, n_libraries: int = 5, n_features: int = 8):
        print("\n" + "=" * 60)
        print("TOP LEARNED FEATURES")
        print("=" * 60)

        tier1 = [r for r in self.results if r.tier == "TIER 1"]
        for r in sorted(tier1, key=lambda x: -x.mean_f1)[:n_libraries]:
            if r.top_features:
                print(f"\n{r.library} (F1={r.mean_f1:.3f}):")
                for name, coef in r.top_features[:n_features]:
                    sign = "+" if coef > 0 else ""
                    print(f"  {name}: {sign}{coef:.3f}")

    def save(self, path: Path):
        output = {
            "results": [r.to_dict() for r in self.results],
        }
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {path}")

    def get_by_tier(self, tier: str) -> List[LibraryResult]:
        return [r for r in self.results if r.tier == tier]
