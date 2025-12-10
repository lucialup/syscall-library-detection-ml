from typing import List, Optional, Tuple, Dict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINEConv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
    LayerNorm,
)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, f1_score
from scipy.sparse import hstack, spmatrix

from src.models.base import BaseDetector
from src.config import ModelConfig, DataConfig
from src.data.loader import AppData
from src.features.provenance import (
    ProvenanceGraphBuilder,
    provenance_graph_to_pyg,
    NUM_EDGE_TYPES,
)


class FocalLoss(nn.Module):
    """Focal Loss with class weighting for imbalanced datasets."""

    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=torch.tensor([self.pos_weight], device=logits.device),
            reduction='none'
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class GINEBlock(nn.Module):
    """Graph Isomorphism Network block with edge features and residual connections."""

    def __init__(self, in_channels: int, out_channels: int, edge_dim: int, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
        )
        self.conv = GINEConv(self.mlp, train_eps=True, edge_dim=edge_dim)
        self.norm = LayerNorm(out_channels)
        self.residual = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        identity = self.residual(x)
        out = self.conv(x, edge_index, edge_attr)
        out = self.norm(out)
        out = F.relu(out + identity)
        return self.dropout(out)


class HybridGNNv5(nn.Module):
    """Hybrid GNN with graph structure (GNN) and text features (TF-IDF)."""

    def __init__(
        self,
        input_dim: int,
        tfidf_dim: int,
        edge_dim: int = NUM_EDGE_TYPES + 2,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        tfidf_dropout: float = 0.3,
        num_syscall_types: int = 10,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.tfidf_dim = tfidf_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
        )

        self.convs = nn.ModuleList([
            GINEBlock(hidden_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        self.jk_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.graph_feat_proj = nn.Linear(num_syscall_types, hidden_dim // 4)

        # GNN output dim: hidden*3 (pools) + hidden//4 (graph features)
        gnn_output_dim = hidden_dim * 3 + hidden_dim // 4
        # Simple projection with dropout (prevent over-reliance on TF-IDF)
        self.tfidf_proj = nn.Sequential(
            nn.Linear(tfidf_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(tfidf_dropout),
        )
        # GNN: gnn_output_dim, TF-IDF: hidden_dim
        combined_dim = gnn_output_dim + hidden_dim

        self.pre_classifier_norm = nn.LayerNorm(combined_dim)

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: Data, tfidf_features: torch.Tensor) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else \
            torch.zeros(edge_index.size(1), NUM_EDGE_TYPES + 2, device=x.device)
        x = self.input_proj(x)
        edge_attr = self.edge_proj(edge_attr)

        layer_outputs = []
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            layer_outputs.append(x)

        layer_stack = torch.stack(layer_outputs, dim=1)
        attn_weights = F.softmax(self.jk_attention(layer_stack), dim=1)
        x = (layer_stack * attn_weights).sum(dim=1)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_add = global_add_pool(x, batch)
        gnn_emb = torch.cat([x_mean, x_max, x_add], dim=-1)

        if hasattr(data, 'graph_features') and data.graph_features is not None:
            graph_feat = self.graph_feat_proj(data.graph_features)
            gnn_emb = torch.cat([gnn_emb, graph_feat], dim=-1)
        else:
            batch_size = gnn_emb.size(0)
            graph_feat = torch.zeros(batch_size, self.hidden_dim // 4, device=gnn_emb.device)
            gnn_emb = torch.cat([gnn_emb, graph_feat], dim=-1)
        tfidf_emb = self.tfidf_proj(tfidf_features)
        combined = torch.cat([gnn_emb, tfidf_emb], dim=-1)
        combined = self.pre_classifier_norm(combined)

        logits = self.classifier(combined)
        return logits.squeeze(-1)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_thresh: float = 0.30,
    max_thresh: float = 0.65,
    min_recall: float = 0.3,
) -> Tuple[float, float]:
    """Find threshold maximizing F1 with bounds and minimum recall constraint."""
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.5, 0.0

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = np.where(
            (precision + recall) > 0,
            2 * (precision * recall) / (precision + recall),
            0
        )

        valid_mask = (thresholds >= min_thresh) & (thresholds <= max_thresh)

    # Apply minimum recall constraint (recall[:-1] aligns with thresholds)
    # This prevents selecting thresholds that would result in very low recall
    recall_mask = recall[:-1] >= min_recall
    valid_mask = valid_mask & recall_mask

    if not valid_mask.any():
        # use safe default
        return 0.5, 0.0

    valid_f1 = f1_scores[:-1][valid_mask]
    valid_thresh = thresholds[valid_mask]

    if len(valid_f1) == 0:
        return 0.5, 0.0

    best_idx = np.argmax(valid_f1)
    return float(valid_thresh[best_idx]), float(valid_f1[best_idx])


class TfidfFeatureExtractor:
    """TF-IDF feature extractor with SVD compression."""

    def __init__(
        self,
        max_features_per_field: int = 300,
        svd_components: int = 100,
        min_df: int = 2,
        max_df: float = 0.95,
    ):
        self.max_features = max_features_per_field
        self.svd_components = svd_components
        self.min_df = min_df
        self.max_df = max_df

        self.vectorizers: Dict[str, TfidfVectorizer] = {}
        self.svd: Optional[TruncatedSVD] = None
        self.scaler: Optional[StandardScaler] = None
        self._fitted = False
        self.text_fields = ['paths', 'threads', 'thread_syscalls', 'syscall_bigrams', 'path_syscalls']

    def fit(self, app_data_list: List[AppData]) -> 'TfidfFeatureExtractor':
        """Fit TF-IDF vectorizers and SVD on text fields."""
        matrices = []

        for field in self.text_fields:
            texts = [getattr(d, field) for d in app_data_list]

            if all(not t for t in texts):
                continue

            vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df,
                ngram_range=(1, 1),
            )

            try:
                matrix = vectorizer.fit_transform(texts)
                self.vectorizers[field] = vectorizer
                matrices.append(matrix)
            except ValueError:
                continue

        if not matrices:
            self._fitted = True
            return self

        combined = hstack(matrices)

        n_components = min(self.svd_components, combined.shape[1] - 1, combined.shape[0] - 1)
        if n_components > 0:
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            reduced = self.svd.fit_transform(combined)

            self.scaler = StandardScaler()
            self.scaler.fit(reduced)

        self._fitted = True
        return self

    def transform(self, app_data_list: List[AppData]) -> np.ndarray:
        """Transform text data using fitted TF-IDF and SVD."""
        if not self._fitted:
            raise RuntimeError("TfidfFeatureExtractor not fitted")

        if not self.vectorizers:
            return np.zeros((len(app_data_list), self.svd_components), dtype=np.float32)

        matrices = []
        for field, vectorizer in self.vectorizers.items():
            texts = [getattr(d, field) for d in app_data_list]
            matrix = vectorizer.transform(texts)
            matrices.append(matrix)

        combined = hstack(matrices)

        if self.svd is None:
            return combined.toarray().astype(np.float32)

        reduced = self.svd.transform(combined)

        if self.scaler is not None:
            reduced = self.scaler.transform(reduced)

        return reduced.astype(np.float32)

    @property
    def output_dim(self) -> int:
        """Return dimensionality of transformed features."""
        if self.svd is not None:
            return self.svd.n_components
        return self.svd_components


class ProvenanceGNNDetectorV5(BaseDetector):
    """Hybrid detector combining GNN and TF-IDF features."""

    def __init__(
        self,
        config: ModelConfig = None,
        data_config: DataConfig = None,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        weight_decay: float = 5e-4,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 20,
        focal_gamma: float = 3.0,
        val_fraction: float = 0.2,
        use_bidirectional: bool = True,
        tfidf_max_features: int = 3000,
        tfidf_svd_components: int = 150,
        tfidf_dropout: float = 0.3,
    ):
        config = config or ModelConfig()
        data_config = data_config or DataConfig()
        self.config = config
        self.data_config = data_config
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.focal_gamma = focal_gamma
        self.val_fraction = val_fraction
        self.use_bidirectional = use_bidirectional
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_svd_components = tfidf_svd_components
        self.tfidf_dropout = tfidf_dropout

        self.model: Optional[HybridGNNv5] = None
        self.tfidf_extractor: Optional[TfidfFeatureExtractor] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._fitted = False
        self.optimal_threshold = 0.5

        self.graph_builder = ProvenanceGraphBuilder(
            max_file_nodes=200,
            max_process_nodes=100,
            max_socket_nodes=50,
        )

        self.classes_ = np.array([0, 1])
        self.syscall_types = ['read', 'write', 'openat', 'close', 'mmap',
                              'clone', 'socket', 'connect', 'unlinkat', 'execve']

    def _prepare_graphs(self, app_data_list: List[AppData], y: np.ndarray) -> List[Data]:
        """Build provenance graphs with graph-level features."""
        graphs = []
        edge_dim = NUM_EDGE_TYPES + 2 if self.use_bidirectional else NUM_EDGE_TYPES + 1

        for app_data, label in zip(app_data_list, y):
            syscall_file = self.data_config.syscall_dir / f"{app_data.package}.syscall.log"

            if syscall_file.exists():
                prov_graph = self.graph_builder.build_from_file(syscall_file)
                pyg_data = provenance_graph_to_pyg(
                    prov_graph, int(label),
                    bidirectional=self.use_bidirectional
                )

                syscall_counts = np.zeros(len(self.syscall_types), dtype=np.float32)
                total = max(prov_graph.total_syscalls, 1)
                for i, sc in enumerate(self.syscall_types):
                    syscall_counts[i] = prov_graph.syscall_counts.get(sc, 0) / total
                pyg_data.graph_features = torch.tensor(syscall_counts).unsqueeze(0)
            else:
                pyg_data = Data(
                    x=torch.zeros(1, 29),
                    edge_index=torch.zeros(2, 0, dtype=torch.long),
                    edge_attr=torch.zeros(0, edge_dim),
                    y=torch.tensor([int(label)], dtype=torch.long),
                    num_nodes=1,
                    graph_features=torch.zeros(1, len(self.syscall_types)),
                )

            graphs.append(pyg_data)

        return graphs

    def fit(self, X: List[AppData], y: np.ndarray) -> "ProvenanceGNNDetectorV5":
        """Train hybrid model with TF-IDF fitted on training data only."""
        if len(X) == 0:
            return self

        self.tfidf_extractor = TfidfFeatureExtractor(
            max_features_per_field=self.tfidf_max_features,
            svd_components=self.tfidf_svd_components,
        )
        self.tfidf_extractor.fit(X)
        tfidf_features = self.tfidf_extractor.transform(X)
        tfidf_dim = self.tfidf_extractor.output_dim
        graphs = self._prepare_graphs(X, y)

        if not graphs:
            return self
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]

        np.random.seed(self.config.random_state)
        np.random.shuffle(pos_indices)
        np.random.shuffle(neg_indices)

        n_pos = len(pos_indices)
        if n_pos <= 5:
            n_pos_val = 1
        elif n_pos <= 15:
            n_pos_val = max(1, int(n_pos * 0.15))
        else:
            n_pos_val = max(1, int(n_pos * self.val_fraction))

        n_neg_val = max(1, int(len(neg_indices) * self.val_fraction))

        val_indices = np.concatenate([pos_indices[:n_pos_val], neg_indices[:n_neg_val]])
        train_indices = np.concatenate([pos_indices[n_pos_val:], neg_indices[n_neg_val:]])

        train_graphs = [graphs[i] for i in train_indices]
        val_graphs = [graphs[i] for i in val_indices]
        train_tfidf = tfidf_features[train_indices]
        val_tfidf = tfidf_features[val_indices]
        y_train = y[train_indices]
        y_val = y[val_indices]
        input_dim = 29
        for g in graphs:
            if g.x is not None and g.x.size(0) > 0:
                input_dim = g.x.size(1)
                break

        edge_dim = NUM_EDGE_TYPES + 2 if self.use_bidirectional else NUM_EDGE_TYPES + 1

        self.model = HybridGNNv5(
            input_dim=input_dim,
            tfidf_dim=tfidf_dim,
            edge_dim=edge_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            tfidf_dropout=self.tfidf_dropout,
            num_syscall_types=len(self.syscall_types),
        ).to(self.device)
        pos_count = max((y_train == 1).sum(), 1)
        neg_count = max((y_train == 0).sum(), 1)
        pos_weight = min(float(neg_count / pos_count), 10.0)
        criterion = FocalLoss(gamma=self.focal_gamma, pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2
        )

        train_tfidf_tensor = torch.tensor(train_tfidf, dtype=torch.float32)
        val_tfidf_tensor = torch.tensor(val_tfidf, dtype=torch.float32)

        train_loader = DataLoader(
            list(range(len(train_graphs))),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            list(range(len(val_graphs))),
            batch_size=self.batch_size,
        )
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            for batch_indices in train_loader:
                batch_indices = batch_indices.numpy()
                batch_graphs = [train_graphs[i] for i in batch_indices]
                batch_tfidf = train_tfidf_tensor[batch_indices].to(self.device)

                batch = self._collate_graphs(batch_graphs)
                batch = batch.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch, batch_tfidf)
                loss = criterion(logits, batch.y.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_indices in val_loader:
                    batch_indices = batch_indices.numpy()
                    batch_graphs = [val_graphs[i] for i in batch_indices]
                    batch_tfidf = val_tfidf_tensor[batch_indices].to(self.device)

                    batch = self._collate_graphs(batch_graphs)
                    batch = batch.to(self.device)

                    logits = self.model(batch, batch_tfidf)
                    loss = criterion(logits, batch.y.float())
                    val_loss += loss.item()

            val_loss /= max(len(val_loader), 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            val_probs = np.concatenate([
                torch.sigmoid(self.model(
                    self._collate_graphs([val_graphs[i] for i in batch_indices.numpy()]).to(self.device),
                    val_tfidf_tensor[batch_indices.numpy()].to(self.device)
                )).cpu().numpy()
                for batch_indices in val_loader
            ])

        n_pos_val = int(y_val.sum())
        n_neg_val = len(y_val) - n_pos_val

        if n_pos_val >= 5 and n_neg_val >= 5:
            self.optimal_threshold, _ = find_optimal_threshold(y_val, val_probs)
        elif n_pos_val > 0 and n_neg_val > 0:
            pos_median = np.median(val_probs[y_val == 1])
            neg_median = np.median(val_probs[y_val == 0])
            self.optimal_threshold = np.clip((pos_median + neg_median) / 2, 0.3, 0.7)
        else:
            self.optimal_threshold = 0.5

        self._fitted = True
        return self

    def _collate_graphs(self, graphs: List[Data]) -> Data:
        from torch_geometric.data import Batch
        return Batch.from_data_list(graphs)

    def predict(self, X: List[AppData]) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.optimal_threshold).astype(int)

    def predict_proba(self, X: List[AppData]) -> np.ndarray:
        if not self._fitted or self.model is None or self.tfidf_extractor is None:
            return np.zeros((len(X), 2))

        tfidf_features = self.tfidf_extractor.transform(X)
        tfidf_tensor = torch.tensor(tfidf_features, dtype=torch.float32)

        graphs = self._prepare_graphs(X, np.zeros(len(X)))

        loader = DataLoader(list(range(len(graphs))), batch_size=self.batch_size)

        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for batch_indices in loader:
                batch_indices = batch_indices.numpy()
                batch_graphs = [graphs[i] for i in batch_indices]
                batch_tfidf = tfidf_tensor[batch_indices].to(self.device)

                batch = self._collate_graphs(batch_graphs)
                batch = batch.to(self.device)

                logits = self.model(batch, batch_tfidf)
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())

        probs_pos = np.concatenate(all_probs)
        probs_neg = 1 - probs_pos

        return np.stack([probs_neg, probs_pos], axis=1)

    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None,
        top_k: int = 15,
    ) -> List[Tuple[str, float]]:
        return []

    @property
    def name(self) -> str:
        return "ProvenanceGNN-v5-Hybrid"
