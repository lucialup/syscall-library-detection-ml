import json
from pathlib import Path
from typing import Dict

from src.config import TARGET_LIBRARIES


def extract_labels(filepath: Path) -> Dict[str, int]:
    labels = {lib: 0 for lib in TARGET_LIBRARIES}

    try:
        with open(filepath, "r") as f:
            sbom = json.load(f)

        for lib in sbom.get("libraries", []):
            lib_name = lib.get("name", "").lower()
            lib_package = lib.get("package", "").lower()

            for target, patterns in TARGET_LIBRARIES.items():
                for pattern in patterns:
                    if pattern.lower() in lib_name or pattern.lower() in lib_package:
                        labels[target] = 1
                        break
    except (json.JSONDecodeError, FileNotFoundError):
        pass

    return labels
