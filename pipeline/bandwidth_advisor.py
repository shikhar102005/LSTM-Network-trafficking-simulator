"""
Intelligent Bandwidth Advisor & Capacity Orchestration Engine.
Translates raw LSTM forecasts into dynamic SDN provisioning actions,
headroom sizing, SLA violation prevention, and bandwidth cost savings.
"""

import numpy as np
from typing import Dict, Any, List


class BandwidthProvisioningAdvisor:
    """
    Intelligent dynamic bandwidth allocation and SLA risk manager.
    """
    def __init__(self, link_capacity_gbps: float = 100.0, default_headroom_pct: float = 0.15):
        self.link_capacity_gbps = link_capacity_gbps
        self.default_headroom_pct = default_headroom_pct

    def analyze_forecast(
        self,
        forecast_gbps: np.ndarray,
        current_traffic_gbps: float,
        headroom_pct: float = None,
        model_rmse: float = 2.0
    ) -> Dict[str, Any]:
        """
        Analyzes multi-step forecast to generate dynamic provisioning plan,
        safety headroom envelope, and SLA alert status.
        """
        headroom = headroom_pct if headroom_pct is not None else self.default_headroom_pct
        forecast_array = np.asarray(forecast_gbps).ravel()
        
        # 1. Dynamic Provisioning Envelope:
        # Recommended bandwidth allocation = Forecast * (1 + headroom) + 1.96 * RMSE (95% statistical safety bound)
        safety_margin = 1.96 * model_rmse
        recommended_allocation = np.minimum(
            self.link_capacity_gbps,
            forecast_array * (1.0 + headroom) + safety_margin
        )
        
        # Prediction Intervals (Confidence bounds for chart visualization)
        lower_bound = np.maximum(0.0, forecast_array - safety_margin)
        upper_bound = np.minimum(self.link_capacity_gbps, forecast_array + safety_margin)
        
        # 2. Maximum forecasted load & utilization
        max_forecast = float(np.max(forecast_array))
        max_utilization_pct = float((max_forecast / self.link_capacity_gbps) * 100.0)
        
        # 3. Static vs Dynamic Cost Efficiency
        # Static provisioning typically allocates 95% of peak link capacity 24/7
        static_provisioning = self.link_capacity_gbps * 0.95
        avg_dynamic_provisioning = float(np.mean(recommended_allocation))
        bandwidth_savings_pct = float(
            max(0.0, (static_provisioning - avg_dynamic_provisioning) / static_provisioning * 100.0)
        )
        
        # 4. SLA Risk and Network Health Status
        # Congestion warning if forecast exceeds 80% link capacity
        # Critical alert if forecast exceeds 90%
        if max_utilization_pct >= 90.0:
            health_status = "CRITICAL_CONGESTION"
            status_color = "#ff4d6d"
            action = f"EMERGENCY SCALE-UP: Predicted peak {max_forecast:.1f} Gbps ({max_utilization_pct:.1f}% link capacity). Proactively reroute traffic via secondary MPLS link or expand elastic cloud bandwidth."
        elif max_utilization_pct >= 75.0:
            health_status = "WARNING_HEADROOM_TIGHT"
            status_color = "#ffaa00"
            action = f"PROACTIVE EXPANSION: Demand expected to reach {max_forecast:.1f} Gbps within horizon. Pre-allocate +{float(max_forecast * 0.2):.1f} Gbps buffer on SDN controller."
        else:
            health_status = "OPTIMAL_NORMAL"
            status_color = "#00f5a0"
            action = f"STABLE: Dynamic headroom adequate. Estimated {bandwidth_savings_pct:.1f}% bandwidth cost savings compared to static 24/7 peak over-provisioning."

        # Hourly provisioning breakdown
        steps_plan = []
        for step_idx, (f_val, rec_val) in enumerate(zip(forecast_array, recommended_allocation)):
            steps_plan.append({
                "step_hour": step_idx + 1,
                "forecast_gbps": round(float(f_val), 2),
                "lower_bound_gbps": round(float(lower_bound[step_idx]), 2),
                "upper_bound_gbps": round(float(upper_bound[step_idx]), 2),
                "recommended_allocation_gbps": round(float(rec_val), 2),
                "utilization_pct": round(float(f_val / self.link_capacity_gbps * 100.0), 1)
            })

        return {
            "health_status": health_status,
            "status_color": status_color,
            "action_recommendation": action,
            "current_traffic_gbps": round(current_traffic_gbps, 2),
            "link_capacity_gbps": self.link_capacity_gbps,
            "max_forecast_gbps": round(max_forecast, 2),
            "max_utilization_pct": round(max_utilization_pct, 1),
            "avg_recommended_allocation_gbps": round(avg_dynamic_provisioning, 2),
            "static_baseline_gbps": round(static_provisioning, 2),
            "bandwidth_savings_pct": round(bandwidth_savings_pct, 1),
            "headroom_pct_applied": round(headroom * 100, 1),
            "steps_plan": steps_plan
        }
