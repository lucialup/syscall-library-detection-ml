import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from src.config import NOISE_PATTERNS

NODE_TYPE_PROCESS = 0
NODE_TYPE_FILE = 1
NODE_TYPE_SOCKET = 2

EDGE_TYPE_READ = 0
EDGE_TYPE_WRITE = 1
EDGE_TYPE_OPEN = 2
EDGE_TYPE_CLOSE = 3
EDGE_TYPE_MMAP = 4
EDGE_TYPE_CLONE = 5
EDGE_TYPE_SOCKET = 6
EDGE_TYPE_CONNECT = 7
EDGE_TYPE_UNLINK = 8
EDGE_TYPE_EXECVE = 9

NUM_EDGE_TYPES = 10

SYSCALL_TO_EDGE_TYPE = {
    'read': EDGE_TYPE_READ,
    'write': EDGE_TYPE_WRITE,
    'openat': EDGE_TYPE_OPEN,
    'open': EDGE_TYPE_OPEN,
    'close': EDGE_TYPE_CLOSE,
    'mmap': EDGE_TYPE_MMAP,
    'clone': EDGE_TYPE_CLONE,
    'socket': EDGE_TYPE_SOCKET,
    'connect': EDGE_TYPE_CONNECT,
    'unlinkat': EDGE_TYPE_UNLINK,
    'unlink': EDGE_TYPE_UNLINK,
    'execve': EDGE_TYPE_EXECVE,
}

PATH_CATEGORIES = {
    'database': ['.db', 'database', 'sqlite', '-journal', '-wal', '-shm'],
    'cache': ['cache', 'http-cache', 'image_cache', 'glide', 'coil', 'fresco'],
    'preferences': ['shared_prefs', 'preferences', 'datastore', '.preferences_pb'],
    'native_lib': ['.so', '/lib/', '/lib64/', 'native-lib'],
    'framework': ['framework', '/apex/', 'system/framework'],
    'app_data': ['/data/data/', '/data/user/', '/files/', 'app_'],
    'system': ['/system/', '/vendor/', '/dev/', '/proc/'],
    'network': ['socket', 'pipe', 'binder', 'ashmem'],
    'dex': ['.dex', '.odex', '.vdex', '.art'],
    'temp': ['/tmp', '/temp', '.tmp'],
}

THREAD_PATTERNS = {
    'main': ['main', 'ui', 'activity'],
    'io': ['io', 'async', 'worker', 'pool'],
    'network': ['okhttp', 'retrofit', 'http', 'network', 'socket'],
    'database': ['room', 'sqlite', 'database', 'db'],
    'render': ['render', 'gl', 'graphics', 'hwui'],
    'gc': ['gc', 'heap', 'finalizer'],
    'binder': ['binder'],
    'jit': ['jit'],
}


def parse_syscall_line(line: str) -> Optional[Dict[str, str]]:
    if not line.startswith("ts="):
        return None

    fields = {}
    for match in re.finditer(r'(\w+)=(?:"([^"]*)"|(\S+))', line):
        key = match.group(1)
        value = match.group(2) if match.group(2) is not None else match.group(3)
        fields[key] = value

    return fields if "syscall" in fields else None


def is_noise_path(path: str) -> bool:
    """Check if path should be filtered as noise."""
    if not path:
        return True
    return any(re.search(p, path) for p in NOISE_PATTERNS)


def categorize_path(path: str) -> str:
    if not path:
        return 'unknown'

    path_lower = path.lower()

    for category, patterns in PATH_CATEGORIES.items():
        if any(p in path_lower for p in patterns):
            return category

    return 'other'


def categorize_thread(comm: str) -> str:
    if not comm:
        return 'unknown'

    comm_lower = comm.lower()

    for category, patterns in THREAD_PATTERNS.items():
        if any(p in comm_lower for p in patterns):
            return category

    return 'other'


def normalize_path_for_node(path: str) -> str:
    if not path:
        return ''

    # Remove numeric IDs that vary between runs
    normalized = re.sub(r'/\d+/', '/N/', path)
    # Normalize package names
    normalized = re.sub(r'com\.[a-zA-Z0-9_.]+', 'PKG', normalized)
    normalized = re.sub(r'org\.[a-zA-Z0-9_.]+', 'PKG', normalized)
    normalized = re.sub(r'io\.[a-zA-Z0-9_.]+', 'PKG', normalized)

    return normalized


@dataclass
class ProvenanceGraph:
    """Directed graph of information flow between processes and file system objects."""
    node_ids: List[str] = field(default_factory=list)
    node_types: List[int] = field(default_factory=list)
    node_categories: List[str] = field(default_factory=list)
    edge_src: List[int] = field(default_factory=list)
    edge_dst: List[int] = field(default_factory=list)
    edge_types: List[int] = field(default_factory=list)
    edge_weights: List[int] = field(default_factory=list)
    syscall_counts: Dict[str, int] = field(default_factory=dict)
    total_syscalls: int = 0


class ProvenanceGraphBuilder:
    def __init__(
        self,
        max_file_nodes: int = 200,
        max_process_nodes: int = 100,
        max_socket_nodes: int = 50,
        min_edge_weight: int = 1,
    ):
        self.max_file_nodes = max_file_nodes
        self.max_process_nodes = max_process_nodes
        self.max_socket_nodes = max_socket_nodes
        self.min_edge_weight = min_edge_weight

    def build_from_file(self, filepath: Path) -> ProvenanceGraph:
        """Build provenance graph from syscall trace file."""
        # Node tracking
        process_nodes: Dict[str, int] = {}  # tid:comm -> node_id
        file_nodes: Dict[str, int] = {}     # normalized_path -> node_id
        socket_nodes: Dict[str, int] = {}   # socket_key -> node_id

        node_ids: List[str] = []
        node_types: List[int] = []
        node_categories: List[str] = []

        # FD tracking per process (pid, fd) -> file_node_id
        fd_table: Dict[Tuple[str, str], int] = {}

        # Edge tracking (src, dst, type) -> count
        edge_counts: Dict[Tuple[int, int, int], int] = defaultdict(int)

        # Syscall counts
        syscall_counts: Dict[str, int] = defaultdict(int)
        total_syscalls = 0

        def get_or_create_process_node(tid: str, comm: str) -> Optional[int]:
            key = f"{tid}:{comm}"
            if key in process_nodes:
                return process_nodes[key]

            if len(process_nodes) >= self.max_process_nodes:
                for existing_key in process_nodes:
                    if comm.lower() in existing_key.lower():
                        return process_nodes[existing_key]
                return None

            node_id = len(node_ids)
            process_nodes[key] = node_id
            node_ids.append(key)
            node_types.append(NODE_TYPE_PROCESS)
            node_categories.append(categorize_thread(comm))
            return node_id

        def get_or_create_file_node(path: str) -> Optional[int]:
            norm_path = normalize_path_for_node(path)
            if norm_path in file_nodes:
                return file_nodes[norm_path]

            if len(file_nodes) >= self.max_file_nodes:
                cat = categorize_path(path)
                for existing_path, node_id in file_nodes.items():
                    if categorize_path(existing_path) == cat:
                        return node_id
                return None

            node_id = len(node_ids)
            file_nodes[norm_path] = node_id
            node_ids.append(norm_path)
            node_types.append(NODE_TYPE_FILE)
            node_categories.append(categorize_path(path))
            return node_id

        def get_or_create_socket_node(pid: str, fd: str, family: str = '', sock_type: str = '') -> Optional[int]:
            key = f"socket:{family}:{sock_type}"
            if key in socket_nodes:
                return socket_nodes[key]

            if len(socket_nodes) >= self.max_socket_nodes:
                return None

            node_id = len(node_ids)
            socket_nodes[key] = node_id
            node_ids.append(key)
            node_types.append(NODE_TYPE_SOCKET)
            node_categories.append('network')
            return node_id

        # Parse syscall trace
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                record = parse_syscall_line(line)
                if not record:
                    continue

                syscall = record.get('syscall', '')
                if not syscall:
                    continue

                total_syscalls += 1
                syscall_counts[syscall] += 1

                pid = record.get('pid', '')
                tid = record.get('tid', pid)
                comm = record.get('comm', 'unknown')
                path = record.get('path', '')
                fd = record.get('fd', '')

                if path and is_noise_path(path):
                    continue

                edge_type = SYSCALL_TO_EDGE_TYPE.get(syscall)
                if edge_type is None:
                    continue

                proc_node = get_or_create_process_node(tid, comm)
                if proc_node is None:
                    continue

                if syscall in ('openat', 'open'):
                    if path:
                        file_node = get_or_create_file_node(path)
                        if file_node is not None:
                            if fd:
                                fd_table[(pid, fd)] = file_node
                            edge_counts[(proc_node, file_node, edge_type)] += 1

                elif syscall == 'close':
                    file_node = fd_table.get((pid, fd))
                    if file_node is not None:
                        edge_counts[(proc_node, file_node, edge_type)] += 1
                        fd_table.pop((pid, fd), None)

                elif syscall == 'read':
                    file_node = fd_table.get((pid, fd))
                    if file_node is None and path:
                        file_node = get_or_create_file_node(path)
                    if file_node is not None:
                        edge_counts[(file_node, proc_node, edge_type)] += 1

                elif syscall == 'write':
                    file_node = fd_table.get((pid, fd))
                    if file_node is None and path:
                        file_node = get_or_create_file_node(path)
                    if file_node is not None:
                        edge_counts[(proc_node, file_node, edge_type)] += 1

                elif syscall == 'mmap':
                    file_node = fd_table.get((pid, fd))
                    if file_node is None and path:
                        file_node = get_or_create_file_node(path)
                    if file_node is not None:
                        edge_counts[(file_node, proc_node, edge_type)] += 1

                elif syscall in ('unlinkat', 'unlink'):
                    if path:
                        file_node = get_or_create_file_node(path)
                        if file_node is not None:
                            edge_counts[(proc_node, file_node, edge_type)] += 1

                elif syscall == 'clone':
                    child_pid = record.get('child_pid', '')
                    if child_pid:
                        child_node = get_or_create_process_node(child_pid, comm)
                        if child_node is not None:
                            edge_counts[(proc_node, child_node, edge_type)] += 1

                elif syscall == 'socket':
                    family = record.get('family', '')
                    sock_type = record.get('type', '')
                    socket_node = get_or_create_socket_node(pid, fd, family, sock_type)
                    if socket_node is not None:
                        if fd:
                            fd_table[(pid, fd)] = socket_node
                        edge_counts[(proc_node, socket_node, edge_type)] += 1

                elif syscall == 'connect':
                    socket_node = fd_table.get((pid, fd))
                    if socket_node is None:
                        socket_node = get_or_create_socket_node(pid, fd)
                    if socket_node is not None:
                        edge_counts[(proc_node, socket_node, edge_type)] += 1

                elif syscall == 'execve':
                    if path:
                        file_node = get_or_create_file_node(path)
                        if file_node is not None:
                            edge_counts[(proc_node, file_node, edge_type)] += 1

        edge_src, edge_dst, edge_types_list, edge_weights = [], [], [], []
        for (src, dst, etype), count in edge_counts.items():
            if count >= self.min_edge_weight:
                edge_src.append(src)
                edge_dst.append(dst)
                edge_types_list.append(etype)
                edge_weights.append(count)

        return ProvenanceGraph(
            node_ids=node_ids,
            node_types=node_types,
            node_categories=node_categories,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_types=edge_types_list,
            edge_weights=edge_weights,
            syscall_counts=dict(syscall_counts),
            total_syscalls=total_syscalls,
        )


PROCESS_CATEGORIES = ['main', 'io', 'network', 'database', 'render', 'gc', 'binder', 'jit', 'other', 'unknown']
FILE_CATEGORIES = ['database', 'cache', 'preferences', 'native_lib', 'framework', 'app_data', 'system', 'network', 'dex', 'temp', 'other', 'unknown']

PROCESS_CATEGORY_TO_IDX = {cat: i for i, cat in enumerate(PROCESS_CATEGORIES)}
FILE_CATEGORY_TO_IDX = {cat: i for i, cat in enumerate(FILE_CATEGORIES)}

NUM_PROCESS_CATEGORIES = len(PROCESS_CATEGORIES)
NUM_FILE_CATEGORIES = len(FILE_CATEGORIES)


def provenance_graph_to_pyg(
    graph: ProvenanceGraph,
    label: int = 0,
    include_edge_attr: bool = True,
    bidirectional: bool = True,
) -> Data:
    """Convert ProvenanceGraph to PyTorch Geometric Data.
    Node features: [type (3) | file_cat (12) | proc_cat (10) | degree (4)] = 29 dims
    Edge features: [type (10) | log_weight (1) | direction (1)] = 12 dims if bidirectional
    """
    num_nodes = len(graph.node_ids)
    node_feature_dim = 3 + NUM_FILE_CATEGORIES + NUM_PROCESS_CATEGORIES + 4
    edge_attr_dim = NUM_EDGE_TYPES + (2 if bidirectional else 1)

    if num_nodes == 0:
        return Data(
            x=torch.zeros(1, node_feature_dim),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0, edge_attr_dim) if include_edge_attr else None,
            y=torch.tensor([label], dtype=torch.long),
            num_nodes=1,
        )

    node_features = torch.zeros(num_nodes, node_feature_dim)

    for i, (ntype, category) in enumerate(zip(graph.node_types, graph.node_categories)):
        node_features[i, ntype] = 1.0

        if ntype == NODE_TYPE_PROCESS:
            cat_idx = PROCESS_CATEGORY_TO_IDX.get(category, PROCESS_CATEGORY_TO_IDX['unknown'])
            node_features[i, 3 + NUM_FILE_CATEGORIES + cat_idx] = 1.0
        elif ntype == NODE_TYPE_FILE:
            cat_idx = FILE_CATEGORY_TO_IDX.get(category, FILE_CATEGORY_TO_IDX['unknown'])
            node_features[i, 3 + cat_idx] = 1.0
        else:
            node_features[i, 3 + FILE_CATEGORY_TO_IDX['network']] = 1.0

    if len(graph.edge_src) == 0:
        edge_index = torch.tensor([[i for i in range(num_nodes)],
                                   [i for i in range(num_nodes)]], dtype=torch.long)
        edge_weights = torch.ones(num_nodes)
        edge_types = torch.zeros(num_nodes, dtype=torch.long)
    else:
        edge_index = torch.tensor([graph.edge_src, graph.edge_dst], dtype=torch.long)
        edge_weights = torch.tensor(graph.edge_weights, dtype=torch.float32)
        edge_types = torch.tensor(graph.edge_types, dtype=torch.long)

    if edge_index.size(1) > 0:
        src_nodes = edge_index[0].numpy()
        dst_nodes = edge_index[1].numpy()

        out_degree = np.bincount(src_nodes, minlength=num_nodes)
        in_degree = np.bincount(dst_nodes, minlength=num_nodes)

        total_degree = out_degree + in_degree
        degree_ratio = np.where(in_degree > 0, out_degree / (in_degree + 1e-8), 0)

        node_features[:, -4] = torch.tensor(np.log1p(out_degree), dtype=torch.float32)
        node_features[:, -3] = torch.tensor(np.log1p(in_degree), dtype=torch.float32)
        node_features[:, -2] = torch.tensor(np.log1p(total_degree), dtype=torch.float32)
        node_features[:, -1] = torch.tensor(degree_ratio, dtype=torch.float32)

    if bidirectional and edge_index.size(1) > 0:
        reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
        edge_weights = torch.cat([edge_weights, edge_weights], dim=0)
        edge_types = torch.cat([edge_types, edge_types], dim=0)

        num_original = edge_index.size(1) // 2
        direction_flags = torch.cat([torch.zeros(num_original), torch.ones(num_original)], dim=0)
    else:
        direction_flags = torch.zeros(edge_index.size(1)) if edge_index.size(1) > 0 else torch.zeros(0)

    if include_edge_attr:
        edge_attr = torch.zeros(edge_index.size(1), edge_attr_dim)
        for i, etype in enumerate(edge_types):
            edge_attr[i, etype] = 1.0
        edge_attr[:, -2] = torch.log1p(edge_weights)
        if bidirectional:
            edge_attr[:, -1] = direction_flags
    else:
        edge_attr = None

    return Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([label], dtype=torch.long),
        num_nodes=num_nodes,
    )
