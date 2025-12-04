from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from src.config import DataConfig
from src.data.parser import SyscallParser
from src.data.sbom import extract_labels


@dataclass
class AppData:
    package: str
    paths: str
    threads: str
    thread_syscalls: str
    syscall_bigrams: str
    path_syscalls: str
    counts: Dict[str, int]
    labels: Dict[str, int]


def load_dataset(config: Optional[DataConfig] = None) -> List[AppData]:
    config = config or DataConfig()
    parser = SyscallParser()

    syscall_files = list(config.syscall_dir.glob("*.syscall.log"))
    sbom_map = {f.stem.replace(".sbom", ""): f for f in config.sbom_dir.glob("*.sbom.json")}

    data = []
    for syscall_file in syscall_files:
        package = syscall_file.stem.replace(".syscall", "")

        if package not in sbom_map:
            continue

        features = parser.parse_file(syscall_file)
        labels = extract_labels(sbom_map[package])

        if features["counts"]["total_syscalls"] < config.min_syscalls:
            continue

        data.append(AppData(
            package=package,
            paths=features["paths"],
            threads=features["threads"],
            thread_syscalls=features["thread_syscalls"],
            syscall_bigrams=features["syscall_bigrams"],
            path_syscalls=features["path_syscalls"],
            counts=features["counts"],
            labels=labels,
        ))

    return data
