import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class HybridSyscallCNN(nn.Module):
    """Hybrid CNN with word-level convolutions and TF-IDF features."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        num_filters: int = 32,
        kernel_sizes: List[int] = [1, 2, 3],
        tfidf_dim: int = 500,
        stats_dim: int = 10
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=num_filters,
                      kernel_size=k,
                      padding=k//2)
            for k in kernel_sizes
        ])

        cnn_output_dim = num_filters * len(kernel_sizes)

        self.tfidf_project = nn.Linear(tfidf_dim, 64)
        self.stats_project = nn.Linear(stats_dim, 16)

        fusion_dim = cnn_output_dim + 64 + 16

        self.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x_seq, x_tfidf, x_stats):
        """Forward pass combining CNN, TF-IDF, and stats branches."""
        emb = self.embedding(x_seq)
        emb = emb.permute(0, 2, 1)

        conv_outs = []
        for conv in self.convs:
            feat = F.relu(conv(emb))
            feat = F.adaptive_max_pool1d(feat, 1).squeeze(2)
            conv_outs.append(feat)

        cnn_feat = torch.cat(conv_outs, dim=1)
        tfidf_feat = F.relu(self.tfidf_project(x_tfidf))
        stats_feat = F.relu(self.stats_project(x_stats))

        combined = torch.cat([cnn_feat, tfidf_feat, stats_feat], dim=1)
        logits = self.classifier(combined)
        return logits


class HybridCNNWordLevelDetector:
    """Wrapper with word-level tokenization using dot-preserving regex."""

    def __init__(
        self,
        vocab_size: int = 2000,
        max_len: int = 500,
        tfidf_dim: int = 500,
        embed_dim: int = 32,
        num_filters: int = 32,
        epochs: int = 30,
        batch_size: int = 32,
        learning_rate: float = 5e-4,
        device: str = "auto"
    ):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tfidf_dim = tfidf_dim
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

        self.tokenizer = CountVectorizer(
            analyzer='word',
            token_pattern=r"(?u)[\w\.-]+",
            max_features=vocab_size,
            lowercase=True
        )

        self.tfidf = TfidfVectorizer(
            analyzer='word',
            token_pattern=r"(?u)[\w\.-]+",
            max_features=tfidf_dim,
            lowercase=True,
            sublinear_tf=True,
            norm='l2'
        )

        self.stats_scaler = StandardScaler()
        self.model = None
        self.vocab = None
        self.save_dir = Path("checkpoints/wordlevel")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _tokenize_batch(self, texts: List[str]) -> np.ndarray:
        """Convert texts to padded integer sequences using word-level tokens."""
        analyzer = self.tokenizer.build_analyzer()
        vocab_get = self.tokenizer.vocabulary_.get

        sequences = []
        for text in texts:
            seq = []
            tokens = analyzer(text)
            for token in tokens:
                idx = vocab_get(token)
                if idx is not None:
                    seq.append(idx + 1)

            if len(seq) > self.max_len:
                seq = seq[:self.max_len]
            else:
                seq = seq + [0] * (self.max_len - len(seq))

            sequences.append(seq)

        return np.array(sequences, dtype=np.int64)

    def _extract_tfidf_features(self, texts: List[str]) -> np.ndarray:
        """Extract word-level TF-IDF features without embedding."""
        return self.tfidf.transform(texts).toarray().astype(np.float32)

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

    def fit(self, app_data_list, y_train: np.ndarray):
        """Train hybrid CNN with word-level tokenization."""
        texts = [(app.threads or "") + " " + (app.paths or "") for app in app_data_list]

        train_idx, val_idx = train_test_split(
            np.arange(len(texts)),
            test_size=0.2,
            random_state=42,
            stratify=y_train
        )

        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        train_apps = [app_data_list[i] for i in train_idx]
        val_apps = [app_data_list[i] for i in val_idx]
        y_train_split, y_val = y_train[train_idx], y_train[val_idx]

        self.tokenizer.fit(train_texts)
        self.tfidf.fit(train_texts)

        X_seq_train = self._tokenize_batch(train_texts)
        X_seq_val = self._tokenize_batch(val_texts)

        X_tfidf_train = self._extract_tfidf_features(train_texts)
        X_tfidf_val = self._extract_tfidf_features(val_texts)

        X_stats_train = self._extract_stats(train_apps)
        X_stats_val = self._extract_stats(val_apps)

        self.stats_scaler.fit(X_stats_train)
        X_stats_train = self.stats_scaler.transform(X_stats_train).astype(np.float32)
        X_stats_val = self.stats_scaler.transform(X_stats_val).astype(np.float32)

        # Convert to tensors
        X_seq_train_t = torch.LongTensor(X_seq_train).to(self.device)
        X_tfidf_train_t = torch.FloatTensor(X_tfidf_train).to(self.device)
        X_stats_train_t = torch.FloatTensor(X_stats_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train_split).unsqueeze(1).to(self.device)

        X_seq_val_t = torch.LongTensor(X_seq_val).to(self.device)
        X_tfidf_val_t = torch.FloatTensor(X_tfidf_val).to(self.device)
        X_stats_val_t = torch.FloatTensor(X_stats_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        actual_vocab_size = len(self.tokenizer.vocabulary_) + 1
        self.model = HybridSyscallCNN(
            vocab_size=actual_vocab_size,
            embed_dim=self.embed_dim,
            num_filters=self.num_filters,
            tfidf_dim=X_tfidf_train.shape[1],
            stats_dim=X_stats_train.shape[1]
        ).to(self.device)

        pos_count = y_train_split.sum()
        neg_count = len(y_train_split) - pos_count
        pos_weight = torch.tensor([neg_count / (pos_count + 1e-6)],
                                  dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-3
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )

        early_stop_patience = 10
        best_val_loss = float('inf')
        patience_counter = 0

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
        """Predict binary labels."""
        return (self.predict_proba(app_data_list) > 0.5).astype(int)

    def save_checkpoint(self, library_name: str, fold: int):
        """Save model checkpoint with all components."""
        checkpoint_path = self.save_dir / f"{library_name}_fold{fold}.pt"

        checkpoint = {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'tokenizer': self.tokenizer,
            'tfidf': self.tfidf,
            'stats_scaler': self.stats_scaler,
            'vocab_size': self.vocab_size,
            'max_len': self.max_len,
            'tfidf_dim': self.tfidf_dim,
            'embed_dim': self.embed_dim,
            'num_filters': self.num_filters
        }

        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, library_name: str, fold: int):
        """Load model checkpoint."""
        checkpoint_path = self.save_dir / f"{library_name}_fold{fold}.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.tokenizer = checkpoint['tokenizer']
        self.tfidf = checkpoint['tfidf']
        self.stats_scaler = checkpoint['stats_scaler']

        actual_vocab_size = len(self.tokenizer.vocabulary_) + 1
        self.model = HybridSyscallCNN(
            vocab_size=actual_vocab_size,
            embed_dim=checkpoint['embed_dim'],
            num_filters=checkpoint['num_filters'],
            tfidf_dim=checkpoint['tfidf_dim'],
            stats_dim=10
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        return self
