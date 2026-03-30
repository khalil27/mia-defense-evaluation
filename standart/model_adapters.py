import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class _TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


class DistilBertKerasLike:
    def __init__(self, model_name="distilbert-base-uncased", num_labels=2, max_len=64):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_len = max_len
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.to(self.device)
        self.lr = 2e-5
        self.weight_decay = 0.0

    def compile(self, optimizer=None, loss=None, metrics=None):
        # Keep keras-like compatibility; training args are set via fit kwargs.
        return self

    def _to_text(self, X_seq):
        X = np.asarray(X_seq)
        if X.ndim == 3:
            X = X[:, 0, :]
        texts = []
        for row in X:
            tokens = [f"f{i}_{float(v):.4f}" for i, v in enumerate(row)]
            texts.append(" ".join(tokens))
        return texts

    def fit(self, X, y, epochs=3, batch_size=8, validation_split=0.0, verbose=0, **kwargs):
        texts = self._to_text(X)
        y = np.asarray(y).astype(int)
        n = len(texts)
        split = int(n * (1 - float(validation_split))) if validation_split and n > 1 else n

        tr_texts, tr_y = texts[:split], y[:split]
        train_ds = _TextDataset(tr_texts, tr_y, self.tokenizer, max_len=self.max_len)
        train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True)

        self.model.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for _ in range(int(epochs)):
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out = self.model(**batch)
                loss = out.loss
                opt.zero_grad()
                loss.backward()
                opt.step()

        class _Hist:
            history = {"loss": [0.0] * max(1, int(epochs))}

        return _Hist()

    @torch.no_grad()
    def predict(self, X, verbose=0):
        texts = self._to_text(X)
        ds = _TextDataset(texts, np.zeros(len(texts), dtype=int), self.tokenizer, max_len=self.max_len)
        loader = DataLoader(ds, batch_size=16, shuffle=False)

        self.model.eval()
        probs = []
        for batch in loader:
            labels = batch.pop("labels")
            _ = labels
            batch = {k: v.to(self.device) for k, v in batch.items()}
            logits = self.model(**batch).logits
            p = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.append(p)
        return np.concatenate(probs, axis=0)


def make_standard_model(dropout=0.15, l2v=0.0, **kwargs):
    _ = (dropout, l2v, kwargs)
    return DistilBertKerasLike()
