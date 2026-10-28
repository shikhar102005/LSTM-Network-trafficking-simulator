"""
Evaluation Suite for Time-Series Bandwidth Forecasting.
Calculates both standard statistical regression metrics and specialized
Computer Network Service Level Agreement (SLA) & Capacity Planning indicators.
"""

import time
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, List


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, capacity_gbps: float = 100.0) -> Dict[str, float]:
    """
    Computes comprehensive metric suite on unscaled (actual Gbps) data.
    y_true: array of shape (N, horizon) or (N,)
    y_pred: array of shape (N, horizon) or (N,)
    """
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    
    # Clip negative values
    y_p = np.maximum(y_p, 0.0)
    
    # 1. Standard Regression Metrics
    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_t, y_p))
    r2 = float(r2_score(y_t, y_p))
    
    # MAPE (with safe epsilon denominator)
    epsilon = 1e-3
    mape = float(np.mean(np.abs((y_t - y_p) / np.maximum(y_t, epsilon))) * 100.0)
    
    # 2. Computer Network Domain Metrics
    # (a) Peak Traffic Forecast Error (PTFE): Error during top 5% traffic periods
    peak_threshold = np.percentile(y_t, 95)
    peak_mask = y_t >= peak_threshold
    if np.any(peak_mask):
        peak_mae = float(np.mean(np.abs(y_t[peak_mask] - y_p[peak_mask])))
        peak_mape = float(np.mean(np.abs((y_t[peak_mask] - y_p[peak_mask]) / y_t[peak_mask])) * 100.0)
    else:
        peak_mae = mae
        peak_mape = mape
        
    # (b) SLA Under-Prediction Ratio (UPR):
    # What % of time did the model under-predict demand? Under-prediction can trigger packet drops.
    under_prediction_mask = y_t > y_p
    under_prediction_rate = float(np.mean(under_prediction_mask) * 100.0)
    
    # (c) Mean Under-Prediction Severity (Gbps deficit when under-predicting)
    if np.any(under_prediction_mask):
        mean_deficit_gbps = float(np.mean(y_t[under_prediction_mask] - y_p[under_prediction_mask]))
    else:
        mean_deficit_gbps = 0.0
        
    # (d) Maximum Absolute Error (Worst-case shock)
    max_error = float(np.max(np.abs(y_t - y_p)))

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mape_pct": round(mape, 2),
        "r2_score": round(r2, 4),
        "peak_traffic_mae": round(peak_mae, 4),
        "peak_traffic_mape": round(peak_mape, 2),
        "under_prediction_rate_pct": round(under_prediction_rate, 2),
        "mean_deficit_gbps": round(mean_deficit_gbps, 4),
        "max_error_gbps": round(max_error, 4)
    }


def benchmark_all_models(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    Y_test: np.ndarray,
    preprocessor,
    capacity_gbps: float = 100.0
) -> Dict[str, Dict[str, Any]]:
    """
    Runs inference across all models, inverse scales predictions,
    measures inference latency, and records comparative benchmark results.
    """
    results = {}
    
    # Real unscaled test targets
    y_true_real = preprocessor.inverse_transform_target(Y_test)
    
    for model_name, model in models_dict.items():
        # Measure inference latency
        start_time = time.perf_counter()
        preds_scaled = model.predict(X_test)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        
        # Inverse transform to Gbps
        preds_real = preprocessor.inverse_transform_target(preds_scaled)
        
        # Calculate metrics
        metrics = calculate_metrics(y_true_real, preds_real, capacity_gbps=capacity_gbps)
        metrics["inference_latency_ms"] = round(elapsed_ms / len(X_test), 3)  # latency per sample
        
        results[model_name] = {
            "metrics": metrics,
            "sample_predictions": preds_real[:30].tolist(),  # First 30 steps for dashboard visualization
        }
        
    return results
