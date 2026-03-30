import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class VulnerableMLPKerasLike:
    def __init__(self, input_dim, hidden1=256, hidden2=128):
        self.device = torch.device("cpu")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        ).to(self.device)
        self.lr = 1e-3

    def compile(self, optimizer=None, loss=None, metrics=None):
        return self

    def _to_features(self, X_seq):
        X = np.asarray(X_seq, dtype=np.float32)
        if X.ndim == 3:
            X = X[:, 0, :]
        return X

    def fit(self, X, y, epochs=100, batch_size=32, validation_split=0.0, verbose=0, **kwargs):
        Xf = self._to_features(X)
        y = np.asarray(y, dtype=np.float32)

        ds = TensorDataset(torch.from_numpy(Xf), torch.from_numpy(y).unsqueeze(1))
        loader = DataLoader(ds, batch_size=int(batch_size), shuffle=True)

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        self.net.train()
        for _ in range(int(epochs)):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits = self.net(xb)
                loss = criterion(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

        class _Hist:
            history = {"loss": [0.0] * max(1, int(epochs))}

        return _Hist()

    @torch.no_grad()
    def predict(self, X, verbose=0):
        Xf = self._to_features(X)
        self.net.eval()
        logits = self.net(torch.from_numpy(Xf).to(self.device)).cpu().numpy().ravel()
        probs = 1.0 / (1.0 + np.exp(-logits))
        return probs


def make_vulnerable_model(input_dim, dropout=0.0, l2v=0.0, **kwargs):
    _ = (dropout, l2v, kwargs)
    return VulnerableMLPKerasLike(input_dim=input_dim)
