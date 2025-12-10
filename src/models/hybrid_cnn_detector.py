import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


class HybridSyscallCNN(nn.Module):
    """Hybrid CNN with three branches: learned CNN, explicit TF-IDF, and global stats."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        num_filters: int = 64,
        kernel_sizes: List[int] = [3, 5, 7],
        dropout: float = 0.5,
        tfidf_dim: int = 500,
        stats_dim: int = 10,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )

        filters_per_kernel = num_filters // len(kernel_sizes)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=filters_per_kernel,
                kernel_size=k,
                padding=k//2
            )
            for k in kernel_sizes
        ])

        self.actual_filters = filters_per_kernel * len(kernel_sizes)
        self.bn = nn.BatchNorm1d(self.actual_filters)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        cnn_output_dim = self.actual_filters * 2

        self.stats_fc = nn.Sequential(
            nn.Linear(stats_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        combined_dim = cnn_output_dim + tfidf_dim + 16

        self.classifier = nn.Sequential(
            nn.Dropout(dropout * 0.6),
            nn.Linear(combined_dim, 1)
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        x_tfidf: torch.Tensor,
        x_stats: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass combining CNN, TF-IDF, and stats branches."""
        x = self.embedding(x_seq)
        x = x.permute(0, 2, 1)

        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))
            conv_outputs.append(conv_out)

        x = torch.cat(conv_outputs, dim=1)
        x = self.bn(x)

        max_pooled = self.max_pool(x).squeeze(2)
        avg_pooled = self.avg_pool(x).squeeze(2)
        cnn_features = torch.cat([max_pooled, avg_pooled], dim=1)

        tfidf_features = x_tfidf
        stats_features = self.stats_fc(x_stats)

        combined = torch.cat([cnn_features, tfidf_features, stats_features], dim=1)
        logits = self.classifier(combined)
        return logits


class HybridCNNLibraryDetector:
    """Wrapper with dual vectorizers: one for CNN embeddings, one for explicit TF-IDF."""

    def __init__(
        self,
        max_features_cnn: int = 2000,
        max_features_tfidf: int = 500,
        max_seq_len: int = 500,
        embed_dim: int = 32,
        num_filters: int = 64,
        epochs: int = 30,
        batch_size: int = 16,
        learning_rate: float = 0.002,
        device: str = "auto",
    ):
        self.max_features_cnn = max_features_cnn
        self.max_features_tfidf = max_features_tfidf
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if device == "auto":
            self.device = torch.device("mps" if torch.backends.mps.is_available()
                                      else "cuda" if torch.cuda.is_available()
                                      else "cpu")
        else:
            self.device = torch.device(device)

        self.vectorizer_cnn = TfidfVectorizer(
            max_features=max_features_cnn,
            analyzer='char_wb',
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.95
        )

        self.vectorizer_tfidf = TfidfVectorizer(
            max_features=max_features_tfidf,
            analyzer='char_wb',
            ngram_range=(2, 5),
            min_df=1,
            max_df=0.98,
            sublinear_tf=True,
            norm='l2'
        )

        self.stats_scaler = StandardScaler()
        self.model = None
        self.vocab_size = None

    def _tokenize_batch(self, texts: List[str]) -> np.ndarray:
        """Convert texts to integer sequences for CNN branch."""
        tfidf_matrix = self.vectorizer_cnn.transform(texts)

        indices = []
        for i in range(len(texts)):
            row = tfidf_matrix[i].toarray()[0]
            top_k = np.argsort(row)[-self.max_seq_len:][::-1]
            top_k = [idx+1 for idx in top_k if row[idx] > 0]

            if len(top_k) < self.max_seq_len:
                top_k = top_k + [0] * (self.max_seq_len - len(top_k))
            else:
                top_k = top_k[:self.max_seq_len]

            indices.append(top_k)

        return np.array(indices, dtype=np.int64)

    def _extract_tfidf_features(self, texts: List[str]) -> np.ndarray:
        """Extract explicit TF-IDF features without embedding."""
        tfidf_matrix = self.vectorizer_tfidf.transform(texts)
        return tfidf_matrix.toarray().astype(np.float32)

    def _extract_stats(self, app_data_list) -> np.ndarray:
        """Extract global statistics features."""
        stats = []
        for app in app_data_list:
            counts = app.counts
            stat_vec = [
                float(counts.get('total_syscalls', 0)),
                float(counts.get('count_openat', 0)),
                float(counts.get('count_read', 0)),
                float(counts.get('count_write', 0)),
                float(counts.get('count_close', 0)),
                float(counts.get('count_mmap', 0)),
                float(counts.get('unique_files', 0)),
                float(counts.get('unique_threads', 0)),
                float(len(app.threads.split()) if app.threads else 0),
                float(len(app.paths.split()) if app.paths else 0),
            ]
            stats.append(stat_vec)

        return np.array(stats, dtype=np.float32)

    def fit(self, app_data_list, y: np.ndarray):
        """Train hybrid CNN on app data."""
        texts = [
            (app.threads or "") + " " + (app.paths or "")
            for app in app_data_list
        ]

        self.vectorizer_cnn.fit(texts)
        self.vectorizer_tfidf.fit(texts)
        self.vocab_size = len(self.vectorizer_cnn.vocabulary_) + 1

        X_seq = self._tokenize_batch(texts)
        X_tfidf = self._extract_tfidf_features(texts)
        X_stats = self._extract_stats(app_data_list)

        self.stats_scaler.fit(X_stats)
        X_stats = self.stats_scaler.transform(X_stats).astype(np.float32)

        n_train = int(0.8 * len(X_seq))
        indices = np.random.permutation(len(X_seq))
        train_idx, val_idx = indices[:n_train], indices[n_train:]

        X_seq_train, X_seq_val = X_seq[train_idx], X_seq[val_idx]
        X_tfidf_train, X_tfidf_val = X_tfidf[train_idx], X_tfidf[val_idx]
        X_stats_train, X_stats_val = X_stats[train_idx], X_stats[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_seq_train_t = torch.LongTensor(X_seq_train).to(self.device)
        X_tfidf_train_t = torch.FloatTensor(X_tfidf_train).to(self.device)
        X_stats_train_t = torch.FloatTensor(X_stats_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)

        X_seq_val_t = torch.LongTensor(X_seq_val).to(self.device)
        X_tfidf_val_t = torch.FloatTensor(X_tfidf_val).to(self.device)
        X_stats_val_t = torch.FloatTensor(X_stats_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        self.model = HybridSyscallCNN(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_filters=self.num_filters,
            tfidf_dim=X_tfidf.shape[1],
            stats_dim=X_stats.shape[1]
        ).to(self.device)

        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        raw_weight = neg_count / (pos_count + 1e-6)
        pos_weight = torch.tensor([raw_weight ** 0.7], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )

        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 10

        for epoch in range(self.epochs):
            self.model.train()

            n_batches = (len(X_seq_train) + self.batch_size - 1) // self.batch_size
            train_loss = 0

            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(X_seq_train))

                batch_seq = X_seq_train_t[start_idx:end_idx]
                batch_tfidf = X_tfidf_train_t[start_idx:end_idx]
                batch_stats = X_stats_train_t[start_idx:end_idx]
                batch_y = y_train_t[start_idx:end_idx]

                optimizer.zero_grad()
                outputs = self.model(batch_seq, batch_tfidf, batch_stats)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_seq_val_t, X_tfidf_val_t, X_stats_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    break

        return self

    def predict_proba(self, app_data_list) -> np.ndarray:
        """Predict probabilities."""
        self.model.eval()

        texts = [(app.threads or "") + " " + (app.paths or "") for app in app_data_list]
        X_seq = self._tokenize_batch(texts)
        X_tfidf = self._extract_tfidf_features(texts)
        X_stats = self._extract_stats(app_data_list)
        X_stats = self.stats_scaler.transform(X_stats).astype(np.float32)

        X_seq_t = torch.LongTensor(X_seq).to(self.device)
        X_tfidf_t = torch.FloatTensor(X_tfidf).to(self.device)
        X_stats_t = torch.FloatTensor(X_stats).to(self.device)

        with torch.no_grad():
            logits = self.model(X_seq_t, X_tfidf_t, X_stats_t)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        return probs

    def predict(self, app_data_list) -> np.ndarray:
        """Predict binary labels with lowered threshold."""
        return (self.predict_proba(app_data_list) > 0.4).astype(int)
