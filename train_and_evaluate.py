"""
Standalone Training and Academic Evaluation Pipeline
Trains LSTM and Benchmarks Against Baselines across ISP, Cloud, and Data Center networks.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pipeline.data_generator import ensure_sample_datasets_exist, load_dataset, NETWORK_PROFILES
from pipeline.preprocessor import NetworkTrafficPreprocessor
from pipeline.evaluation import benchmark_all_models, calculate_metrics
from pipeline.bandwidth_advisor import BandwidthProvisioningAdvisor
from models.lstm_model import BandwidthLSTM, LSTMTrainer
from models.baseline_models import (
    SimpleMovingAverageBaseline,
    ExponentialSmoothingBaseline,
    RidgeRegressionBaseline,
    RandomForestBaseline,
    GRUTrainerWrapper
)

SAVED_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "saved")
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "artifacts")


def run_pipeline_for_dataset(profile_key: str, epochs: int = 25, horizon: int = 6, lookback: int = 24):
    print(f"\n======================================================================")
    print(f"  RUNNING PIPELINE: {NETWORK_PROFILES[profile_key]['name'].upper()}")
    print(f"======================================================================")
    
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    # 1. Load Data
    df = load_dataset(profile_key)
    capacity = NETWORK_PROFILES[profile_key]["link_capacity_gbps"]
    print(f"[*] Loaded {len(df)} samples. Link capacity: {capacity} Gbps")
    
    # 2. Preprocessing & Temporal Splitting
    preprocessor = NetworkTrafficPreprocessor(lookback_window=lookback, forecast_horizon=horizon)
    data_bundle = preprocessor.prepare_train_val_test_splits(df, train_ratio=0.70, val_ratio=0.15, batch_size=32)
    
    loaders = data_bundle["loaders"]
    arrays = data_bundle["arrays"]
    meta = data_bundle["metadata"]
    
    print(f"[*] Train sequences: {meta['train_samples']} | Val: {meta['val_samples']} | Test: {meta['test_samples']}")
    print(f"[*] Features ({meta['num_features']}): {meta['feature_names']}")
    
    # 3. Train Primary Model: Stacked SLA-Aware LSTM
    print(f"[*] Initializing and Training BandwidthLSTM (lookback={lookback}, horizon={horizon})...")
    lstm = BandwidthLSTM(
        input_size=meta["num_features"],
        hidden_size=64,
        num_layers=2,
        output_size=horizon,
        dropout=0.2,
        bidirectional=False
    )
    trainer = LSTMTrainer(lstm, learning_rate=0.003, use_sla_loss=True, under_predict_penalty=2.5)
    history = trainer.fit(loaders["train"], loaders["val"], epochs=epochs, early_stop_patience=6)
    
    checkpoint_path = os.path.join(SAVED_MODELS_DIR, f"lstm_{profile_key}.pt")
    trainer.save_checkpoint(checkpoint_path)
    print(f"[+] Saved trained LSTM checkpoint to {checkpoint_path}")
    
    # 4. Fit Baseline Models
    print(f"[*] Training baseline benchmark models...")
    sma = SimpleMovingAverageBaseline(target_idx=0, horizon=horizon)
    ema = ExponentialSmoothingBaseline(alpha=0.35, target_idx=0, horizon=horizon)
    ridge = RidgeRegressionBaseline(alpha=1.0, horizon=horizon)
    ridge.fit(arrays["X_train"], arrays["Y_train"])
    
    rf = RandomForestBaseline(n_estimators=30, max_depth=10, horizon=horizon)
    rf.fit(arrays["X_train"], arrays["Y_train"])
    
    gru = GRUTrainerWrapper(input_size=meta["num_features"], horizon=horizon, epochs=15)
    gru.fit(arrays["X_train"], arrays["Y_train"])
    
    # 5. Comprehensive Academic Benchmark
    models_dict = {
        "LSTM (Proposed)": trainer,
        "GRU Recurrent": gru,
        "Random Forest": rf,
        "Ridge Linear Reg": ridge,
        "Exponential Smoothing": ema,
        "Moving Average (SMA)": sma
    }
    
    benchmarks = benchmark_all_models(
        models_dict=models_dict,
        X_test=arrays["X_test"],
        Y_test=arrays["Y_test"],
        preprocessor=preprocessor,
        capacity_gbps=capacity
    )
    
    # 6. Format and Print Benchmark Table
    print("\n----------------------------------------------------------------------------------------------------------")
    print(f"{'Model Architecture':<24} | {'RMSE (Gbps)':<11} | {'MAE (Gbps)':<10} | {'MAPE (%)':<9} | {'R2 Score':<8} | {'Peak MAE':<9} | {'Under-Pred %':<12}")
    print("----------------------------------------------------------------------------------------------------------")
    for m_name, res in benchmarks.items():
        m = res["metrics"]
        print(f"{m_name:<24} | {m['rmse']:<11.3f} | {m['mae']:<10.3f} | {m['mape_pct']:<9.2f} | {m['r2_score']:<8.3f} | {m['peak_traffic_mae']:<9.3f} | {m['under_prediction_rate_pct']:<12.1f}")
    print("----------------------------------------------------------------------------------------------------------\n")
    
    # 7. Bandwidth Provisioning Analysis
    y_test_real = preprocessor.inverse_transform_target(arrays["Y_test"])
    lstm_preds_scaled = trainer.predict(arrays["X_test"])
    lstm_preds_real = preprocessor.inverse_transform_target(lstm_preds_scaled)
    
    advisor = BandwidthProvisioningAdvisor(link_capacity_gbps=capacity, default_headroom_pct=0.15)
    latest_forecast = lstm_preds_real[-1]
    curr_traffic = float(y_test_real[-1, 0])
    plan = advisor.analyze_forecast(latest_forecast, current_traffic_gbps=curr_traffic, model_rmse=benchmarks["LSTM (Proposed)"]["metrics"]["rmse"])
    
    print(f"[+] Bandwidth Provisioning Advisor Analysis:")
    print(f"    Health Status: {plan['health_status']}")
    print(f"    Cost Savings vs Static Peak Provisioning: {plan['bandwidth_savings_pct']:.1f}%")
    print(f"    Recommendation: {plan['action_recommendation']}")
    
    # 8. Generate and Save Visualizations
    generate_academic_plots(
        profile_key=profile_key,
        history=history,
        benchmarks=benchmarks,
        y_test_real=y_test_real,
        lstm_preds_real=lstm_preds_real,
        capacity=capacity
    )
    
    # Save benchmark metrics to JSON
    json_path = os.path.join(SAVED_MODELS_DIR, f"benchmarks_{profile_key}.json")
    with open(json_path, "w") as f:
        # Convert non-serializable objects
        clean_benchmarks = {}
        for k, v in benchmarks.items():
            clean_benchmarks[k] = {
                "metrics": v["metrics"]
            }
        json.dump(clean_benchmarks, f, indent=2)
        
    return benchmarks


def generate_academic_plots(profile_key: str, history: dict, benchmarks: dict, y_test_real: np.ndarray, lstm_preds_real: np.ndarray, capacity: float):
    """Generates publication-quality evaluation figures."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor('#0b0f19')
    for ax in axes.ravel():
        ax.set_facecolor('#131a2a')
        ax.tick_params(colors='#8b9bb4')
        ax.xaxis.label.set_color('#8b9bb4')
        ax.yaxis.label.set_color('#8b9bb4')
        ax.title.set_color('#e2e8f0')
        for spine in ax.spines.values():
            spine.set_color('#2a364f')
            
    # Plot 1: Actual vs LSTM Forecast trace (first 96 hours of test set, step +1)
    ax1 = axes[0, 0]
    plot_len = min(96, len(y_test_real))
    actual_trace = y_test_real[:plot_len, 0]
    pred_trace = lstm_preds_real[:plot_len, 0]
    time_steps = np.arange(plot_len)
    
    ax1.plot(time_steps, actual_trace, label='Actual Telemetry (Gbps)', color='#00f2fe', linewidth=2.0)
    ax1.plot(time_steps, pred_trace, label='LSTM Forecast (Gbps)', color='#ff007f', linewidth=2.0, linestyle='--')
    ax1.axhline(y=capacity, color='#ff4d6d', linestyle=':', label=f'Link Capacity ({capacity:.0f}G)')
    ax1.fill_between(time_steps, pred_trace * 0.92, pred_trace * 1.08, color='#ff007f', alpha=0.15, label='95% Headroom Bound')
    ax1.set_title(f"Time-Series Telemetry vs Forecast ({profile_key.replace('_', ' ').title()})")
    ax1.set_xlabel("Time Step (Hours)")
    ax1.set_ylabel("Throughput (Gbps)")
    ax1.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.2, color='#64748b')

    # Plot 2: Training & Validation Loss Curve
    ax2 = axes[0, 1]
    epochs_range = np.arange(1, len(history["train_loss"]) + 1)
    ax2.plot(epochs_range, history["train_loss"], label='Train Loss (SLA Penalized)', color='#38ef7d', linewidth=2)
    ax2.plot(epochs_range, history["val_loss"], label='Validation Loss', color='#ffd200', linewidth=2)
    ax2.set_title("LSTM Convergence & Learning Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0')
    ax2.grid(True, linestyle='--', alpha=0.2, color='#64748b')

    # Plot 3: Model Benchmark Comparison (RMSE & MAE)
    ax3 = axes[1, 0]
    models = list(benchmarks.keys())
    rmses = [benchmarks[m]["metrics"]["rmse"] for m in models]
    maes = [benchmarks[m]["metrics"]["mae"] for m in models]
    x_pos = np.arange(len(models))
    width = 0.35
    
    ax3.bar(x_pos - width/2, rmses, width, label='RMSE (Gbps)', color='#4facfe')
    ax3.bar(x_pos + width/2, maes, width, label='MAE (Gbps)', color='#00f2fe', alpha=0.7)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([m.split()[0] for m in models], rotation=25, ha='right')
    ax3.set_title("Comparative Error Benchmarks Across Models")
    ax3.set_ylabel("Error (Gbps) - Lower is Better")
    ax3.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0')
    ax3.grid(True, linestyle='--', alpha=0.2, color='#64748b')

    # Plot 4: Prediction Error Residual Distribution
    ax4 = axes[1, 1]
    residuals = actual_trace - pred_trace
    ax4.hist(residuals, bins=25, color='#a855f7', edgecolor='#c084fc', alpha=0.75, density=True)
    ax4.axvline(0, color='#ffffff', linestyle='--', linewidth=1.2)
    ax4.set_title("Forecast Residual Distribution (Actual - Predicted)")
    ax4.set_xlabel("Error Residual (Gbps)")
    ax4.set_ylabel("Density")
    ax4.grid(True, linestyle='--', alpha=0.2, color='#64748b')

    plt.tight_layout()
    output_png = os.path.join(ARTIFACTS_DIR, f"eval_plot_{profile_key}.png")
    plt.savefig(output_png, dpi=180, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[+] Saved publication figure to {output_png}")


def main():
    ensure_sample_datasets_exist()
    
    # Run across default primary dataset (isp_backbone)
    primary_dataset = "isp_backbone"
    run_pipeline_for_dataset(primary_dataset, epochs=25, horizon=6, lookback=24)
    
    # Also train and save weights for the other two profiles so the web app switches instantly!
    for other_dataset in ["cloud_egress", "datacenter_fabric"]:
        run_pipeline_for_dataset(other_dataset, epochs=20, horizon=6, lookback=24)
        
    print("\n[SUCCESS] All datasets trained, evaluated, and checkpoints created successfully!")


if __name__ == "__main__":
    main()
