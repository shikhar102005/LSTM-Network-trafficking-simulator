"""
End-to-End Automated Test Suite for LSTM Network Traffic Forecasting Project.
Tests Data Ingestion, Preprocessing, PyTorch Neural Networks, Baseline Models,
Evaluation Metrics, Bandwidth Advisor, and FastAPI Web Endpoints.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient

from pipeline.data_generator import generate_network_telemetry, NETWORK_PROFILES
from pipeline.preprocessor import NetworkTrafficPreprocessor
from pipeline.evaluation import calculate_metrics
from pipeline.bandwidth_advisor import BandwidthProvisioningAdvisor
from models.lstm_model import BandwidthLSTM, LSTMTrainer, SLAPenaltyLoss
from models.baseline_models import (
    SimpleMovingAverageBaseline,
    ExponentialSmoothingBaseline,
    RidgeRegressionBaseline,
    RandomForestBaseline
)
from app import app


@pytest.fixture
def sample_df():
    """Generates a small test dataset."""
    return generate_network_telemetry("isp_backbone", days=14, freq_minutes=60, seed=123)


def test_data_generation(sample_df):
    """Verifies generated telemetry schema, shape, non-negativity, and continuity."""
    assert len(sample_df) == 14 * 24
    expected_cols = [
        "timestamp", "throughput_gbps", "link_capacity_gbps", "utilization_pct",
        "packet_rate_mpps", "latency_ms", "packet_drop_pct", "active_flows"
    ]
    for col in expected_cols:
        assert col in sample_df.columns
        
    assert sample_df["throughput_gbps"].min() > 0.0
    assert sample_df["throughput_gbps"].max() <= 100.0
    assert not sample_df.isnull().any().any()


def test_preprocessor_pipeline(sample_df):
    """Tests feature engineering, MinMax scaling, sequence windowing, and tensor conversions."""
    lookback = 12
    horizon = 4
    prep = NetworkTrafficPreprocessor(lookback_window=lookback, forecast_horizon=horizon)
    
    bundle = prep.prepare_train_val_test_splits(sample_df, train_ratio=0.7, val_ratio=0.15, batch_size=16)
    
    assert "loaders" in bundle
    assert "arrays" in bundle
    assert "metadata" in bundle
    
    meta = bundle["metadata"]
    assert meta["num_features"] == 10
    assert meta["train_samples"] > 0
    
    # Check sliding window dimensions
    X_train = bundle["arrays"]["X_train"]
    Y_train = bundle["arrays"]["Y_train"]
    assert X_train.ndim == 3
    assert X_train.shape[1] == lookback
    assert X_train.shape[2] == meta["num_features"]
    assert Y_train.ndim == 2
    assert Y_train.shape[1] == horizon
    
    # Test inverse scaling accuracy
    raw_val = np.array([25.0, 50.0, 75.0])
    scaled = prep.scale_target(raw_val)
    recovered = prep.inverse_transform_target(scaled)
    np.testing.assert_allclose(raw_val, recovered, rtol=1e-4)


def test_sla_penalty_loss():
    """Verifies that the asymmetric SLA loss penalizes under-forecasting more than over-forecasting."""
    loss_fn = SLAPenaltyLoss(under_predict_penalty=3.0)
    
    y_true = torch.tensor([50.0])
    
    # Case A: Model under-predicts by 10 Gbps (demand is 50, prediction is 40)
    y_under = torch.tensor([40.0])
    loss_under = loss_fn(y_under, y_true).item()
    
    # Case B: Model over-predicts by 10 Gbps (demand is 50, prediction is 60)
    y_over = torch.tensor([60.0])
    loss_over = loss_fn(y_over, y_true).item()
    
    # Under-prediction must incur 3x higher loss due to SLA breach weighting
    assert loss_under == pytest.approx(3.0 * loss_over)


def test_lstm_forward_and_trainer(sample_df):
    """Tests PyTorch LSTM forward pass and mini training loop."""
    prep = NetworkTrafficPreprocessor(lookback_window=12, forecast_horizon=4)
    bundle = prep.prepare_train_val_test_splits(sample_df, train_ratio=0.7, val_ratio=0.15, batch_size=16)
    
    num_features = bundle["metadata"]["num_features"]
    model = BandwidthLSTM(input_size=num_features, hidden_size=32, num_layers=1, output_size=4)
    
    dummy_x = torch.randn(8, 12, num_features)
    out = model(dummy_x)
    assert out.shape == (8, 4)
    
    trainer = LSTMTrainer(model, learning_rate=0.01)
    history = trainer.fit(bundle["loaders"]["train"], bundle["loaders"]["val"], epochs=2)
    assert len(history["train_loss"]) == 2
    
    # Test batch prediction
    preds = trainer.predict(bundle["arrays"]["X_test"][:5])
    assert preds.shape == (5, 4)


def test_baseline_models():
    """Ensures all baseline models fit and output valid prediction horizons."""
    N, seq_len, feats = 20, 12, 10
    horizon = 4
    X = np.random.rand(N, seq_len, feats).astype(np.float32)
    y = np.random.rand(N, horizon).astype(np.float32)
    
    # 1. Moving Average
    sma = SimpleMovingAverageBaseline(target_idx=0, horizon=horizon)
    sma_preds = sma.predict(X)
    assert sma_preds.shape == (N, horizon)
    
    # 2. Exponential Smoothing
    ema = ExponentialSmoothingBaseline(horizon=horizon)
    ema_preds = ema.predict(X)
    assert ema_preds.shape == (N, horizon)
    
    # 3. Ridge Regression
    ridge = RidgeRegressionBaseline(horizon=horizon)
    ridge.fit(X, y)
    ridge_preds = ridge.predict(X)
    assert ridge_preds.shape == (N, horizon)
    
    # 4. Random Forest
    rf = RandomForestBaseline(n_estimators=5, max_depth=4, horizon=horizon)
    rf.fit(X, y)
    rf_preds = rf.predict(X)
    assert rf_preds.shape == (N, horizon)


def test_evaluation_metrics():
    """Tests accuracy calculation of RMSE, MAE, MAPE, R2, and SLA Under-prediction rate."""
    y_true = np.array([50.0, 60.0, 70.0, 80.0])
    y_pred = np.array([48.0, 62.0, 65.0, 85.0])
    
    metrics = calculate_metrics(y_true, y_pred, capacity_gbps=100.0)
    
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "mape_pct" in metrics
    assert "r2_score" in metrics
    assert "peak_traffic_mae" in metrics
    assert "under_prediction_rate_pct" in metrics
    
    assert metrics["rmse"] > 0.0
    assert metrics["mae"] > 0.0
    assert 0.0 <= metrics["under_prediction_rate_pct"] <= 100.0


def test_bandwidth_advisor():
    """Tests dynamic provisioning envelope and cost savings calculations."""
    advisor = BandwidthProvisioningAdvisor(link_capacity_gbps=100.0, default_headroom_pct=0.15)
    forecast = np.array([45.0, 50.0, 55.0, 60.0, 65.0, 70.0])
    current_traffic = 42.0
    
    plan = advisor.analyze_forecast(forecast, current_traffic_gbps=current_traffic, model_rmse=2.5)
    
    assert plan["health_status"] in ["OPTIMAL_NORMAL", "WARNING_HEADROOM_TIGHT", "CRITICAL_CONGESTION"]
    assert "bandwidth_savings_pct" in plan
    assert plan["bandwidth_savings_pct"] >= 0.0
    assert len(plan["steps_plan"]) == len(forecast)
    assert plan["max_forecast_gbps"] == 70.0


def test_fastapi_endpoints():
    """Tests REST API endpoints using FastAPI TestClient."""
    client = TestClient(app)
    
    # 1. Health check
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"
    
    # 2. Datasets listing
    res = client.get("/api/datasets")
    assert res.status_code == 200
    datasets = res.json()["datasets"]
    assert len(datasets) == 4
    
    # 3. Telemetry query
    res = client.get("/api/telemetry?dataset=isp_backbone&points=30")
    assert res.status_code == 200
    telemetry = res.json()
    assert telemetry["points_count"] == 30
    
    # 4. Forecast endpoint
    res = client.post("/api/forecast", json={
        "dataset": "isp_backbone",
        "horizon": 4,
        "headroom_pct": 0.15
    })
    assert res.status_code == 200
    forecast_data = res.json()
    assert len(forecast_data["forecast"]["values_gbps"]) == 4
    assert "advisor" in forecast_data
    
    # 5. Benchmarks endpoint
    res = client.get("/api/benchmarks?dataset=isp_backbone")
    assert res.status_code == 200
    assert "benchmarks" in res.json()
    
    # 6. Surge simulation endpoint
    res = client.post("/api/simulate-surge", json={
        "dataset": "isp_backbone",
        "surge_magnitude": 30.0,
        "surge_duration_steps": 2,
        "surge_type": "flash_crowd"
    })
    assert res.status_code == 200
    surge_data = res.json()
    assert "forecast_gbps" in surge_data
    assert surge_data["injected_magnitude_gbps"] == 30.0
