import re
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import Counter

from src.config import NOISE_PATTERNS


def parse_line(line: str) -> Optional[Dict[str, str]]:
    if not line.startswith("ts="):
        return None

    fields = {}
    for match in re.finditer(r'(\w+)=(?:"([^"]*)"|(\S+))', line):
        key = match.group(1)
        value = match.group(2) if match.group(2) is not None else match.group(3)
        fields[key] = value

    return fields if "syscall" in fields else None


def normalize_path(path: str) -> str:
    if not path:
        return ""
    path = re.sub(r"\d+", "N", path)
    path = re.sub(r"com\.[a-zA-Z0-9_.]+", "PKG", path)
    path = re.sub(r"org\.[a-zA-Z0-9_.]+", "PKG", path)
    path = re.sub(r"io\.[a-zA-Z0-9_.]+", "PKG", path)
    return path


def normalize_thread(comm: str) -> str:
    if not comm:
        return ""
    comm = re.sub(r"-?\d+$", "", comm)
    return comm.lower()


def extract_path_tokens(path: str) -> List[str]:
    if not path:
        return []

    tokens = []
    norm_path = normalize_path(path)

    parts = [p for p in norm_path.split("/") if p and p not in ["data", "user", "N", "PKG"]]
    tokens.extend(parts)

    if "." in path:
        ext = path.rsplit(".", 1)[-1].lower()
        if len(ext) <= 6:
            tokens.append(f"ext_{ext}")

    path_lower = path.lower()
    if "database" in path_lower or ".db" in path_lower:
        tokens.append("IS_DATABASE")
    if "cache" in path_lower:
        tokens.append("IS_CACHE")
    if "preference" in path_lower or "settings" in path_lower:
        tokens.append("IS_PREFERENCES")
    if ".so" in path:
        tokens.append("IS_NATIVE_LIB")

    return tokens


def is_noise(path: str) -> bool:
    return any(re.search(p, path) for p in NOISE_PATTERNS)


def get_path_category(path: str) -> Optional[str]:
    if not path:
        return None
    path_lower = path.lower()
    if ".db" in path_lower or "database" in path_lower:
        return "db"
    if "cache" in path_lower:
        return "cache"
    if ".so" in path:
        return "native"
    if "preference" in path_lower or "datastore" in path_lower:
        return "prefs"
    return None


class SyscallParser:
    def __init__(self):
        self.reset()

    def reset(self):
        self.path_tokens: List[str] = []
        self.thread_names: Set[str] = set()
        self.thread_syscalls: List[str] = []
        self.syscall_bigrams: List[str] = []
        self.path_syscalls: List[str] = []
        self.counts: Dict[str, int] = {
            "openat": 0, "read": 0, "write": 0, "close": 0,
            "mmap": 0, "clone": 0, "socket": 0, "connect": 0,
            "unlinkat": 0, "fstat": 0,
        }
        self.total_syscalls: int = 0
        self.has_tcp: bool = False
        self.has_udp: bool = False
        self._prev_syscall: Optional[str] = None

    def parse_file(self, filepath: Path) -> Dict:
        self.reset()

        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                self._process_line(line)

        return self._build_result()

    def _process_line(self, line: str):
        record = parse_line(line)
        if not record:
            return

        path = record.get("path", "")
        comm = record.get("comm", "")
        syscall = record.get("syscall", "")

        if is_noise(path):
            return

        self.total_syscalls += 1
        self.path_tokens.extend(extract_path_tokens(path))

        thread = normalize_thread(comm)
        if thread:
            self.thread_names.add(thread)
            if syscall:
                self.thread_syscalls.append(f"{thread}:{syscall}")

        if self._prev_syscall and syscall:
            self.syscall_bigrams.append(f"{self._prev_syscall}_{syscall}")
        self._prev_syscall = syscall

        category = get_path_category(path)
        if category and syscall:
            self.path_syscalls.append(f"{category}:{syscall}")

        if syscall in self.counts:
            self.counts[syscall] += 1

        if syscall == "socket":
            socket_type = record.get("type", "")
            if "STREAM" in socket_type:
                self.has_tcp = True
            elif "DGRAM" in socket_type:
                self.has_udp = True

    def _build_result(self) -> Dict:
        thread_syscall_counts = Counter(self.thread_syscalls)
        top_thread_syscalls = [ts for ts, _ in thread_syscall_counts.most_common(100)]

        return {
            "paths": " ".join(self.path_tokens),
            "threads": " ".join(self.thread_names),
            "thread_syscalls": " ".join(top_thread_syscalls),
            "syscall_bigrams": " ".join(self.syscall_bigrams[:1000]),
            "path_syscalls": " ".join(self.path_syscalls[:500]),
            "counts": {
                **{f"count_{k}": v for k, v in self.counts.items()},
                "total_syscalls": self.total_syscalls,
                "has_tcp_socket": int(self.has_tcp),
                "has_udp_socket": int(self.has_udp),
                "unique_threads": len(self.thread_names),
                "unique_files": len(set(self.path_tokens)),
            },
        }
