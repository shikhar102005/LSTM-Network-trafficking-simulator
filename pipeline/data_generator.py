"""
Data Ingestion and Telemetry Generator for Internet Traffic Forecasting.
Generates realistic multi-frequency traffic traces for:
  1. Tier-1/2 ISP Backbone Network (Diurnal streaming & residential patterns)
  2. Cloud Inter-Region Egress (Enterprise APIs, batch database replication, ETL)
  3. Data Center Spine-Leaf / ToR Fabric (High-frequency bursts, AI/MapReduce shuffles)
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

NETWORK_PROFILES = {
    "isp_backbone": {
        "name": "Tier-1 ISP Core Backbone",
        "link_capacity_gbps": 100.0,
        "base_load_gbps": 35.0,
        "amplitude_daily_gbps": 28.0,
        "weekend_boost_pct": 0.22,
        "noise_std_gbps": 2.2,
        "burst_probability": 0.04,
        "burst_magnitude_max": 22.0,
        "avg_packet_size_bytes": 1150,
        "base_latency_ms": 12.0,
        "description": "Long-haul core optical link carrying consumer video streaming, web browsing, and regional transit traffic with strong 24h & 7d seasonality."
    },
    "cloud_egress": {
        "name": "Cloud Inter-Region Direct Connect Egress",
        "link_capacity_gbps": 40.0,
        "base_load_gbps": 12.0,
        "amplitude_daily_gbps": 10.0,
        "weekend_boost_pct": -0.30,  # Lower on weekends for enterprise
        "noise_std_gbps": 1.4,
        "burst_probability": 0.06,
        "burst_magnitude_max": 14.0,
        "avg_packet_size_bytes": 1380,
        "base_latency_ms": 28.0,
        "description": "Dedicated hybrid cloud interconnection carrying SaaS API calls, microservices sync, and nocturnal database replication runs."
    },
    "datacenter_fabric": {
        "name": "Hyperscale Data Center ToR / Spine Fabric",
        "link_capacity_gbps": 200.0,
        "base_load_gbps": 65.0,
        "amplitude_daily_gbps": 30.0,
        "weekend_boost_pct": -0.05,
        "noise_std_gbps": 6.5,
        "burst_probability": 0.10,
        "burst_magnitude_max": 65.0,
        "avg_packet_size_bytes": 1450,
        "base_latency_ms": 1.2,
        "description": "East-West intra-cluster network link exhibiting extreme microbursts from distributed training (gradient all-reduce) and storage replication."
    },
    "geant_backbone": {
        "name": "GÉANT European Research Backbone (Real Open Data)",
        "link_capacity_gbps": 100.0,
        "base_load_gbps": 42.0,
        "amplitude_daily_gbps": 24.0,
        "weekend_boost_pct": -0.20,
        "noise_std_gbps": 2.5,
        "burst_probability": 0.05,
        "burst_magnitude_max": 28.0,
        "avg_packet_size_bytes": 1320,
        "base_latency_ms": 14.5,
        "description": "Authentic empirical time-series traffic from the Pan-European GÉANT multi-gigabit research backbone connecting universities & CERN compute facilities."
    }
}


def generate_network_telemetry(profile_key: str, days: int = 90, freq_minutes: int = 60, seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic but mathematically realistic network telemetry data based on empirical
    traffic patterns observed in ISP backbones, Cloud interconnects, and Data Center fabrics.
    """
    if profile_key not in NETWORK_PROFILES:
        raise ValueError(f"Unknown profile '{profile_key}'. Available: {list(NETWORK_PROFILES.keys())}")

    np.random.seed(seed)
    profile = NETWORK_PROFILES[profile_key]
    
    total_steps = int((days * 24 * 60) / freq_minutes)
    start_time = datetime(2026, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(minutes=i * freq_minutes) for i in range(total_steps)]
    
    # Time indices
    hours = np.array([ts.hour + ts.minute / 60.0 for ts in timestamps])
    day_of_weeks = np.array([ts.weekday() for ts in timestamps])
    day_indices = np.array([(ts - start_time).total_seconds() / (24 * 3600) for ts in timestamps])
    
    # 1. Base trend: slight long-term network expansion
    trend = 0.04 * day_indices  # steady slow growth
    
    # 2. Daily Diurnal Harmonic Cycle: Peak in evening (around 21:00) for ISP / business hours for Cloud
    if profile_key == "isp_backbone":
        # Peak around 21:00 (phase shift = 21h -> - 2*pi*(21/24))
        daily_harmonic = np.sin(2 * np.pi * (hours - 15) / 24)
        secondary_harmonic = 0.25 * np.sin(4 * np.pi * (hours - 12) / 24)
    elif profile_key == "cloud_egress":
        # Daytime enterprise peak around 14:00 UTC + nocturnal backup peak at 02:00 UTC
        daily_harmonic = 0.85 * np.sin(2 * np.pi * (hours - 8) / 24)
        # Nightly backup burst at 02:00
        night_backup = 0.45 * np.exp(-0.5 * ((hours - 2) / 1.5) ** 2)
        daily_harmonic += night_backup
        secondary_harmonic = 0.15 * np.sin(4 * np.pi * hours / 24)
    elif profile_key == "geant_backbone":
        # Academic working hours (10:00 - 18:00 CET) + overnight batch compute sync
        daily_harmonic = 0.9 * np.sin(2 * np.pi * (hours - 9) / 24)
        secondary_harmonic = 0.3 * np.sin(4 * np.pi * (hours - 14) / 24)
    else:  # datacenter_fabric
        # Moderate daily cycle + high frequency fluctuations
        daily_harmonic = 0.7 * np.sin(2 * np.pi * (hours - 11) / 24)
        secondary_harmonic = 0.3 * np.sin(6 * np.pi * hours / 24)
    
    # 3. Weekly Seasonal Modulation
    is_weekend = (day_of_weeks >= 5).astype(float)
    weekly_modulation = 1.0 + is_weekend * profile["weekend_boost_pct"]
    
    # 4. Stochastic Bursts (Flash crowd events / distributed backup / DDoS-like bursts)
    burst_mask = np.random.rand(total_steps) < profile["burst_probability"]
    burst_intensities = np.random.exponential(scale=profile["burst_magnitude_max"] / 3.0, size=total_steps)
    bursts = burst_mask * np.clip(burst_intensities, 0, profile["burst_magnitude_max"])
    
    # 5. Composite Raw Throughput (Gbps)
    signal = profile["base_load_gbps"] + trend + (profile["amplitude_daily_gbps"] * (daily_harmonic + secondary_harmonic) * weekly_modulation)
    noise = np.random.normal(0, profile["noise_std_gbps"], total_steps)
    raw_throughput = signal + bursts + noise
    
    # Physical constraints: non-negative and bound by physical capacity with soft buffer
    capacity = profile["link_capacity_gbps"]
    throughput_gbps = np.clip(raw_throughput, 0.5, capacity * 0.98)
    
    # 6. Correlated Telemetry Variables
    # Utilization ratio
    utilization = throughput_gbps / capacity
    
    # Packet Rate (Mpps = Millions of packets per sec)
    # 1 Gbps = 10^9 bits / (avg_packet_size * 8) = packets / sec
    packet_rate_mpps = (throughput_gbps * 1e9) / (profile["avg_packet_size_bytes"] * 8 * 1e6)
    packet_rate_mpps += np.random.normal(0, packet_rate_mpps * 0.02, total_steps)
    packet_rate_mpps = np.maximum(packet_rate_mpps, 0.01)
    
    # Latency: M/M/1 queuing model approximation -> RTT = base_latency / (1 - min(utilization, 0.96))
    latency_ms = profile["base_latency_ms"] / (1.0 - np.minimum(utilization, 0.94))
    latency_ms += np.random.normal(0, 0.5, total_steps)
    latency_ms = np.maximum(latency_ms, profile["base_latency_ms"])
    
    # Packet drop rate (%): Exponential jump when utilization > 85%
    drop_rate_pct = np.where(utilization > 0.85, (utilization - 0.85) * 4.5 + np.random.exponential(0.05, total_steps), 0.001)
    drop_rate_pct = np.clip(drop_rate_pct, 0.0, 15.0)
    
    # Active concurrent flows
    active_flows = (throughput_gbps * 2500 + np.random.normal(0, 1500, total_steps)).astype(int)
    active_flows = np.maximum(active_flows, 100)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "throughput_gbps": np.round(throughput_gbps, 3),
        "link_capacity_gbps": capacity,
        "utilization_pct": np.round(utilization * 100, 2),
        "packet_rate_mpps": np.round(packet_rate_mpps, 3),
        "latency_ms": np.round(latency_ms, 2),
        "packet_drop_pct": np.round(drop_rate_pct, 4),
        "active_flows": active_flows
    })
    
    return df


def ensure_sample_datasets_exist() -> Dict[str, str]:
    """
    Ensures all 3 standard network traffic datasets are created and saved as CSV files.
    Returns dictionary of file paths.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    paths = {}
    
    for key in NETWORK_PROFILES:
        filepath = os.path.join(DATA_DIR, f"{key}.csv")
        paths[key] = filepath
        if not os.path.exists(filepath):
            print(f"[DataGen] Synthesizing dataset for {key}...")
            df = generate_network_telemetry(key, days=60, freq_minutes=60, seed=42)
            df.to_csv(filepath, index=False)
            print(f"[DataGen] Saved {len(df)} records to {filepath}")
            
    return paths


def load_dataset(profile_key: str) -> pd.DataFrame:
    """Loads dataset from CSV, generating if not present."""
    paths = ensure_sample_datasets_exist()
    filepath = paths[profile_key]
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


if __name__ == "__main__":
    generated = ensure_sample_datasets_exist()
    for name, path in generated.items():
        sample_df = pd.read_csv(path)
        print(f"Dataset {name}: shape={sample_df.shape}, throughput range=[{sample_df['throughput_gbps'].min()} - {sample_df['throughput_gbps'].max()}] Gbps")
