"""
Time-Series Preprocessing, Feature Engineering, and Sliding Window Generator
for Network Bandwidth Forecasting.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
import torch
from torch.utils.data import TensorDataset, DataLoader


class NetworkTrafficPreprocessor:
    """
    Handles feature engineering, scaling, sequence generation,
    and chronological train/validation/test partitioning.
    """
    def __init__(self, lookback_window: int = 24, forecast_horizon: int = 6):
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        
        # Scaling statistics
        self.feature_mins = None
        self.feature_maxs = None
        self.target_min = None
        self.target_max = None
        self.feature_columns: List[str] = []
        self.target_col = "throughput_gbps"

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives cyclic temporal representations and short-term rolling statistics.
        Prevents lookahead bias by using backward rolling windows only.
        """
        data = df.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            
        # 1. Cyclical time encodings
        hours = data["timestamp"].dt.hour + data["timestamp"].dt.minute / 60.0
        dows = data["timestamp"].dt.weekday
        
        data["sin_hour"] = np.sin(2 * np.pi * hours / 24.0)
        data["cos_hour"] = np.cos(2 * np.pi * hours / 24.0)
        data["sin_dow"] = np.sin(2 * np.pi * dows / 7.0)
        data["cos_dow"] = np.cos(2 * np.pi * dows / 7.0)
        data["is_weekend"] = (dows >= 5).astype(float)
        
        # 2. Backward Rolling statistics (handling initial NaNs via bfill)
        data["rolling_mean_6"] = data[self.target_col].rolling(window=6, min_periods=1).mean()
        data["rolling_std_12"] = data[self.target_col].rolling(window=12, min_periods=1).std().fillna(0.0)
        
        # Selected input features
        self.feature_columns = [
            self.target_col,
            "sin_hour",
            "cos_hour",
            "sin_dow",
            "cos_dow",
            "is_weekend",
            "rolling_mean_6",
            "rolling_std_12",
            "packet_rate_mpps",
            "latency_ms"
        ]
        
        return data

    def fit(self, train_df: pd.DataFrame):
        """Fits MinMax scalers on the training set only (preventing test set leakage)."""
        df_feat = self.engineer_features(train_df)
        
        # Target scaling parameters
        self.target_min = float(df_feat[self.target_col].min())
        self.target_max = float(df_feat[self.target_col].max())
        if self.target_max == self.target_min:
            self.target_max = self.target_min + 1e-5
            
        # Feature scaling parameters
        self.feature_mins = {}
        self.feature_maxs = {}
        for col in self.feature_columns:
            cmin = float(df_feat[col].min())
            cmax = float(df_feat[col].max())
            if cmax == cmin:
                cmax = cmin + 1e-5
            self.feature_mins[col] = cmin
            self.feature_maxs[col] = cmax

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Scales features using the fitted parameters into range [0, 1]."""
        df_feat = self.engineer_features(df)
        scaled = np.zeros((len(df_feat), len(self.feature_columns)), dtype=np.float32)
        
        for idx, col in enumerate(self.feature_columns):
            cmin = self.feature_mins[col]
            cmax = self.feature_maxs[col]
            val = df_feat[col].values.astype(np.float32)
            scaled[:, idx] = np.clip((val - cmin) / (cmax - cmin), 0.0, 1.0)
            
        return scaled

    def inverse_transform_target(self, scaled_target: np.ndarray) -> np.ndarray:
        """Inverses target predictions back to raw throughput (Gbps)."""
        return scaled_target * (self.target_max - self.target_min) + self.target_min

    def scale_target(self, raw_target: np.ndarray) -> np.ndarray:
        """Scales raw target values into [0, 1]."""
        return np.clip((raw_target - self.target_min) / (self.target_max - self.target_min), 0.0, 1.0)

    def create_sliding_windows(
        self, scaled_features: np.ndarray, raw_targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates 3D input sequences X of shape (N, lookback, features)
        and target forecasts Y of shape (N, horizon).
        """
        N = len(scaled_features)
        total_window = self.lookback_window + self.forecast_horizon
        
        if N < total_window:
            raise ValueError(f"Sequence length {N} is shorter than window {total_window}")
            
        num_samples = N - total_window + 1
        X = np.zeros((num_samples, self.lookback_window, scaled_features.shape[1]), dtype=np.float32)
        Y = np.zeros((num_samples, self.forecast_horizon), dtype=np.float32)
        
        scaled_targets = self.scale_target(raw_targets)
        
        for i in range(num_samples):
            X[i] = scaled_features[i : i + self.lookback_window]
            Y[i] = scaled_targets[i + self.lookback_window : i + total_window]
            
        return X, Y

    def prepare_train_val_test_splits(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Splits data chronologically, fits scaler on train, transforms splits,
        and constructs PyTorch DataLoaders.
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_raw = df.iloc[:train_end].copy()
        val_raw = df.iloc[train_end:val_end].copy()
        test_raw = df.iloc[val_end:].copy()
        
        # Fit scaler on train only
        self.fit(train_raw)
        
        # Transform all splits
        train_scaled = self.transform(train_raw)
        val_scaled = self.transform(val_raw)
        test_scaled = self.transform(test_raw)
        
        train_targets = train_raw[self.target_col].values
        val_targets = val_raw[self.target_col].values
        test_targets = test_raw[self.target_col].values
        
        # Create sliding windows
        X_train, Y_train = self.create_sliding_windows(train_scaled, train_targets)
        X_val, Y_val = self.create_sliding_windows(val_scaled, val_targets)
        X_test, Y_test = self.create_sliding_windows(test_scaled, test_targets)
        
        # Convert to PyTorch Tensors
        t_X_train = torch.from_numpy(X_train)
        t_Y_train = torch.from_numpy(Y_train)
        t_X_val = torch.from_numpy(X_val)
        t_Y_val = torch.from_numpy(Y_val)
        t_X_test = torch.from_numpy(X_test)
        t_Y_test = torch.from_numpy(Y_test)
        
        train_loader = DataLoader(TensorDataset(t_X_train, t_Y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(t_X_val, t_Y_val), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(t_X_test, t_Y_test), batch_size=batch_size, shuffle=False)
        
        # Keep track of timestamps for test set plotting
        test_timestamps = test_raw["timestamp"].values[self.lookback_window : len(test_raw) - self.forecast_horizon + 1]
        
        return {
            "loaders": {
                "train": train_loader,
                "val": val_loader,
                "test": test_loader
            },
            "arrays": {
                "X_train": X_train, "Y_train": Y_train,
                "X_val": X_val, "Y_val": Y_val,
                "X_test": X_test, "Y_test": Y_test,
                "test_raw_targets": test_targets[self.lookback_window : len(test_raw) - self.forecast_horizon + 1],
                "test_timestamps": test_timestamps
            },
            "metadata": {
                "num_features": len(self.feature_columns),
                "feature_names": self.feature_columns,
                "target_min": self.target_min,
                "target_max": self.target_max,
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test)
            }
        }
