"""
PyTorch Deep LSTM Network for Multi-Step Network Traffic Forecasting.
Includes standard MSE loss and an Asymmetric SLA-Aware Loss function.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class SLAPenaltyLoss(nn.Module):
    """
    Asymmetric Loss Function for Network Capacity Planning.
    Penalizes under-forecasting (y > y_pred) more severely than over-forecasting,
    because under-predicting network traffic causes buffer drops, latency spikes,
    and Service Level Agreement (SLA) violations.
    """
    def __init__(self, under_predict_penalty: float = 2.5):
        super(SLAPenaltyLoss, self).__init__()
        self.alpha = under_predict_penalty
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        error = y_true - y_pred
        squared_err = error ** 2
        # Apply heavy weight when actual > predicted (traffic exceeds headroom)
        weights = torch.where(error > 0, self.alpha, 1.0)
        return torch.mean(weights * squared_err)


class BandwidthLSTM(nn.Module):
    """
    Multi-Layer LSTM with Dropout and Multi-Step Linear Forecasting Head.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 6,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super(BandwidthLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        fc_in = hidden_size * self.num_directions
        self.head = nn.Sequential(
            nn.Linear(fc_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, (hn, cn) = self.lstm(x)
        # We take the representation from the last time step
        last_step_out = lstm_out[:, -1, :]  # shape: (batch_size, hidden_size * num_directions)
        out = self.head(last_step_out)       # shape: (batch_size, output_size)
        return out


class LSTMTrainer:
    """
    High-level training, validation, and inference wrapper for BandwidthLSTM.
    """
    def __init__(
        self,
        model: BandwidthLSTM,
        device: Optional[torch.device] = None,
        learning_rate: float = 0.003,
        use_sla_loss: bool = True,
        under_predict_penalty: float = 2.5
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = SLAPenaltyLoss(under_predict_penalty) if use_sla_loss else nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    def train_epoch(self, train_loader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            self.optimizer.zero_grad()
            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * len(batch_x)
        return total_loss / len(train_loader.dataset)

    def validate(self, val_loader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)
                total_loss += loss.item() * len(batch_x)
        return total_loss / len(val_loader.dataset)

    def fit(self, train_loader, val_loader, epochs: int = 30, early_stop_patience: int = 8) -> Dict[str, List[float]]:
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.scheduler.step(val_loss)
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                break

        if best_weights is not None:
            self.model.load_state_dict(best_weights)
            
        return self.history

    def predict(self, x_numpy: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(x_numpy).float().to(self.device)
            preds = self.model(tensor_x)
            return preds.cpu().numpy()

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "config": {
                "input_size": self.model.input_size,
                "hidden_size": self.model.hidden_size,
                "num_layers": self.model.num_layers,
                "output_size": self.model.output_size,
                "bidirectional": self.model.bidirectional
            },
            "history": self.history
        }, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: Optional[torch.device] = None) -> "LSTMTrainer":
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=dev)
        cfg = checkpoint["config"]
        model = BandwidthLSTM(
            input_size=cfg["input_size"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            output_size=cfg["output_size"],
            bidirectional=cfg.get("bidirectional", False)
        )
        model.load_state_dict(checkpoint["model_state"])
        trainer = cls(model, device=dev)
        trainer.history = checkpoint.get("history", {"train_loss": [], "val_loss": []})
        return trainer
