"""
Baseline Models for Academic Benchmarking:
  1. Simple Moving Average (SMA)
  2. Exponential Smoothing (EMA)
  3. Ridge Linear Autoregressive Regressor
  4. Random Forest Regressor
  5. Deep Gated Recurrent Unit (GRU)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from typing import Optional


class SimpleMovingAverageBaseline:
    """Predicts future bandwidth as the arithmetic mean of recent observations."""
    def __init__(self, target_idx: int = 0, horizon: int = 6):
        self.target_idx = target_idx
        self.horizon = horizon

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass  # Non-parametric

    def predict(self, X: np.ndarray) -> np.ndarray:
        # X: (N, seq_len, features)
        # Average the last 6 steps of target
        recent_target = X[:, -6:, self.target_idx]
        mean_val = np.mean(recent_target, axis=1, keepdims=True)
        # Broadcast across forecast horizon
        return np.repeat(mean_val, self.horizon, axis=1)


class ExponentialSmoothingBaseline:
    """Predicts future bandwidth using exponential decay weighting of lookback observations."""
    def __init__(self, alpha: float = 0.35, target_idx: int = 0, horizon: int = 6):
        self.alpha = alpha
        self.target_idx = target_idx
        self.horizon = horizon

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        N, seq_len, _ = X.shape
        targets = X[:, :, self.target_idx]  # (N, seq_len)
        
        # Exponential weights
        weights = (1 - self.alpha) ** np.arange(seq_len)[::-1]
        weights /= weights.sum()
        
        weighted_val = np.dot(targets, weights)[:, np.newaxis]
        return np.repeat(weighted_val, self.horizon, axis=1)


class RidgeRegressionBaseline:
    """Multi-output Ridge linear regression over flattened lookback representations."""
    def __init__(self, alpha: float = 1.0, horizon: int = 6):
        self.model = Ridge(alpha=alpha)
        self.horizon = horizon

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, seq_len, feats = X.shape
        X_flat = X.reshape(N, seq_len * feats)
        self.model.fit(X_flat, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        N, seq_len, feats = X.shape
        X_flat = X.reshape(N, seq_len * feats)
        preds = self.model.predict(X_flat)
        return np.clip(preds, 0.0, 1.0)


class RandomForestBaseline:
    """Random Forest regressor on flattened feature sequences."""
    def __init__(self, n_estimators: int = 40, max_depth: int = 12, horizon: int = 6, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=random_state
        )
        self.horizon = horizon

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, seq_len, feats = X.shape
        X_flat = X.reshape(N, seq_len * feats)
        self.model.fit(X_flat, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        N, seq_len, feats = X.shape
        X_flat = X.reshape(N, seq_len * feats)
        preds = self.model.predict(X_flat)
        return np.clip(preds, 0.0, 1.0)


class GRUBaselineModel(nn.Module):
    """Deep Gated Recurrent Unit (GRU) model for recurrent architecture comparison."""
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, output_size: int = 6, dropout: float = 0.2):
        super(GRUBaselineModel, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last_out = out[:, -1, :]
        return self.head(last_out)


class GRUTrainerWrapper:
    """Fast trainer for the GRU comparison baseline."""
    def __init__(self, input_size: int, horizon: int = 6, epochs: int = 20, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GRUBaselineModel(input_size=input_size, output_size=horizon).to(self.device)
        self.epochs = epochs

    def fit(self, X: np.ndarray, y: np.ndarray):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.004)
        tensor_x = torch.from_numpy(X).float().to(self.device)
        tensor_y = torch.from_numpy(y).float().to(self.device)
        
        dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        self.model.train()
        for _ in range(self.epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                pred = self.model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(X).float().to(self.device)
            preds = self.model(tensor_x)
            return np.clip(preds.cpu().numpy(), 0.0, 1.0)
