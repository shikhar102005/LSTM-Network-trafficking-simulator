/**
 * NetForecaster Neo-Brutalist Dashboard Controller
 * Handles live actual telemetry vs multi-step LSTM forecast comparisons,
 * test-set ground-truth overlays, and dynamic SDN headroom allocations.
 */

document.addEventListener("DOMContentLoaded", () => {
    // Application State
    const state = {
        dataset: "isp_backbone",
        horizon: 6,
        headroomPct: 0.15,
        mode: "test_eval", // 'test_eval' (actual ground truth overlay) or 'future_projection'
        testOffset: 0,
        datasetsMeta: {},
        currentForecast: null,
        currentTelemetry: null,
        benchmarks: null
    };

    // Chart Instances
    let mainChart = null;
    let benchmarkChart = null;

    // DOM Elements
    const elements = {
        pillGeant: document.getElementById("pillGeant"),
        pillIsp: document.getElementById("pillIsp"),
        pillCloud: document.getElementById("pillCloud"),
        pillDc: document.getElementById("pillDc"),
        modeTestEval: document.getElementById("modeTestEval"),
        modeFuture: document.getElementById("modeFuture"),
        kpiCurrentTraffic: document.getElementById("kpiCurrentTraffic"),
        kpiUtilizationBadge: document.getElementById("kpiUtilizationBadge"),
        kpiForecastValue: document.getElementById("kpiForecastValue"),
        kpiDeltaBadge: document.getElementById("kpiDeltaBadge"),
        kpiSlaStatus: document.getElementById("kpiSlaStatus"),
        kpiUnderPredBadge: document.getElementById("kpiUnderPredBadge"),
        kpiCostSavings: document.getElementById("kpiCostSavings"),
        horizonRange: document.getElementById("horizonRange"),
        horizonVal: document.getElementById("horizonVal"),
        headroomRange: document.getElementById("headroomRange"),
        headroomVal: document.getElementById("headroomVal"),
        offsetRange: document.getElementById("offsetRange"),
        offsetVal: document.getElementById("offsetVal"),
        applyParamsBtn: document.getElementById("applyParamsBtn"),
        runInferenceBtn: document.getElementById("runInferenceBtn"),
        slotViewBtn: document.getElementById("slotViewBtn"),
        surgeBtn: document.getElementById("surgeBtn"),
        resetTelemetryBtn: document.getElementById("resetTelemetryBtn"),
        legendActualText: document.getElementById("legendActualText"),
        verifyModeTag: document.getElementById("verifyModeTag"),
        stepsChipsGrid: document.getElementById("stepsChipsGrid"),
        benchmarkTableBody: document.getElementById("benchmarkTableBody"),
        surgeModal: document.getElementById("surgeModal"),
        cancelModalBtn: document.getElementById("cancelModalBtn"),
        executeSurgeBtn: document.getElementById("executeSurgeBtn"),
        surgeTypeSelect: document.getElementById("surgeTypeSelect"),
        surgeModalHighlight: document.getElementById("surgeModalHighlight"),
        theoryModal: document.getElementById("theoryModal"),
        toggleTheoryBtn: document.getElementById("toggleTheoryBtn"),
        closeTheoryBtn: document.getElementById("closeTheoryBtn")
    };

    // Chart.js Clean High-Contrast Neo-Brutalist Theme Defaults
    Chart.defaults.color = '#000000';
    Chart.defaults.font.family = "'Outfit', -apple-system, sans-serif";
    Chart.defaults.font.weight = '600';
    Chart.defaults.borderColor = 'rgba(0, 0, 0, 0.08)';

    // Initialize Application
    async function init() {
        setupEventListeners();
        await loadDatasetsMeta();
        await refreshAllData();
    }

    // Load available datasets metadata
    async function loadDatasetsMeta() {
        try {
            const res = await fetch("/api/datasets");
            const data = await res.json();
            data.datasets.forEach(ds => {
                state.datasetsMeta[ds.key] = ds;
            });
        } catch (err) {
            console.error("Failed to load datasets metadata:", err);
        }
    }

    // Refresh all telemetry, forecast, and benchmark views
    async function refreshAllData() {
        try {
            elements.runInferenceBtn.disabled = true;
            elements.runInferenceBtn.textContent = 'Computing...';

            await Promise.all([
                fetchTelemetry(),
                fetchForecast(),
                fetchBenchmarks()
            ]);
        } catch (err) {
            console.error("Error refreshing dashboard:", err);
        } finally {
            elements.runInferenceBtn.disabled = false;
            elements.runInferenceBtn.textContent = 'Run Predictor';
        }
    }

    // Fetch Telemetry Stream
    async function fetchTelemetry() {
        const res = await fetch(`/api/telemetry?dataset=${state.dataset}&points=72`);
        const data = await res.json();
        state.currentTelemetry = data;
    }

    // Fetch Forecast with Actual Ground Truth
    async function fetchForecast() {
        const res = await fetch("/api/forecast", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                dataset: state.dataset,
                horizon: state.horizon,
                headroom_pct: state.headroomPct,
                mode: state.mode,
                test_offset: state.testOffset
            })
        });
        const data = await res.json();
        state.currentForecast = data;

        updateKpisAndAdvisor(data);
        renderMainForecastChart(data);
        renderStepVerifications(data);
    }

    // Fetch Academic Benchmarks
    async function fetchBenchmarks() {
        const res = await fetch(`/api/benchmarks?dataset=${state.dataset}`);
        const data = await res.json();
        state.benchmarks = data.benchmarks;

        renderBenchmarkChart(data.benchmarks);
        renderBenchmarkTable(data.benchmarks);
    }

    // Update KPI Metric Cards
    function updateKpisAndAdvisor(data) {
        const adv = data.advisor;
        const currentTraffic = adv.current_traffic_gbps;
        const capacity = adv.link_capacity_gbps;
        const util = (currentTraffic / capacity * 100).toFixed(1);

        // 1. Current Telemetry KPI
        elements.kpiCurrentTraffic.textContent = currentTraffic.toFixed(1);
        elements.kpiUtilizationBadge.textContent = `Util: ${util}% of ${capacity} Gbps`;

        // 2. 1-Hour Ahead Forecast KPI
        const nextStepForecast = data.forecast.values_gbps[0] || currentTraffic;
        elements.kpiForecastValue.textContent = nextStepForecast.toFixed(1);
        const delta = nextStepForecast - currentTraffic;
        const deltaPct = ((delta / currentTraffic) * 100).toFixed(1);

        if (delta >= 0) {
            elements.kpiDeltaBadge.textContent = `Trend: +${delta.toFixed(1)} Gbps (+${deltaPct}%)`;
            elements.kpiDeltaBadge.className = 'kpi-badge badge-green';
        } else {
            elements.kpiDeltaBadge.textContent = `Trend: ${delta.toFixed(1)} Gbps (${deltaPct}%)`;
            elements.kpiDeltaBadge.className = 'kpi-badge badge-blue';
        }

        // 3. SLA Congestion Risk
        elements.kpiSlaStatus.textContent = adv.health_status.replace(/_/g, " ");
        if (adv.health_status.includes("CRITICAL")) {
            elements.kpiSlaStatus.style.color = "#dc2626";
            elements.kpiUnderPredBadge.className = "kpi-badge badge-coral";
            elements.kpiUnderPredBadge.textContent = "SLA Breach Risk: High";
        } else if (adv.health_status.includes("WARNING")) {
            elements.kpiSlaStatus.style.color = "#d97706";
            elements.kpiUnderPredBadge.className = "kpi-badge badge-yellow";
            elements.kpiUnderPredBadge.textContent = "Tight Headroom Buffer";
        } else {
            elements.kpiSlaStatus.style.color = "#15803d";
            elements.kpiUnderPredBadge.className = "kpi-badge badge-green";
            elements.kpiUnderPredBadge.textContent = "Headroom Buffer Intact";
        }

        // 4. Dynamic Cost Savings
        elements.kpiCostSavings.textContent = `${adv.bandwidth_savings_pct.toFixed(1)}%`;
    }

    // Render Main Forecast Chart with ACTUAL GROUND TRUTH DATA
    function renderMainForecastChart(data) {
        const ctx = document.getElementById("mainForecastChart").getContext("2d");

        const histLabels = data.history.timestamps;
        const histValues = data.history.throughput_gbps;
        const futureLabels = data.forecast.timestamps;
        const futureValues = data.forecast.values_gbps;
        const plan = data.advisor.steps_plan || [];
        const capacity = data.capacity_gbps;
        const hasActualFuture = data.actual_ground_truth && data.actual_ground_truth.values_gbps;

        // Combined timestamps across history and forecast
        const allLabels = [...histLabels, ...futureLabels];

        // 1. ACTUAL Telemetry: In test_eval mode, extends through the entire sequence!
        let actualSeries;
        if (hasActualFuture) {
            actualSeries = [...histValues, ...data.actual_ground_truth.values_gbps];
            elements.legendActualText.textContent = "Actual Telemetry (Ground Truth)";
        } else {
            actualSeries = [...histValues, ...new Array(futureLabels.length).fill(null)];
            elements.legendActualText.textContent = "Observed Telemetry";
        }

        // 2. LSTM Forecast: Connects with the last historical point and plots predicted curve
        const forecastSeries = new Array(histValues.length - 1).fill(null);
        forecastSeries.push(histValues[histValues.length - 1]);
        forecastSeries.push(...futureValues);

        // 3. Dynamic SDN Headroom Allocation & Bounds
        const headroomSeries = new Array(histValues.length).fill(null);
        const lowerBoundSeries = new Array(histValues.length).fill(null);
        const upperBoundSeries = new Array(histValues.length).fill(null);

        plan.forEach(step => {
            headroomSeries.push(step.recommended_allocation_gbps);
            lowerBoundSeries.push(step.lower_bound_gbps);
            upperBoundSeries.push(step.upper_bound_gbps);
        });

        // 4. Capacity threshold line
        const capacitySeries = new Array(allLabels.length).fill(capacity);

        if (mainChart) {
            mainChart.destroy();
        }

        mainChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: allLabels,
                datasets: [
                    {
                        label: 'Actual Observed Data',
                        data: actualSeries,
                        borderColor: '#0284c7', // Deep Cyan / Sky Blue
                        backgroundColor: 'rgba(2, 132, 199, 0.08)',
                        borderWidth: 2.8,
                        tension: 0.25,
                        pointRadius: hasActualFuture ? (ctx) => (ctx.dataIndex >= histValues.length ? 4 : 0) : 0,
                        pointBackgroundColor: '#0284c7',
                        pointBorderColor: '#ffffff',
                        pointBorderWidth: 1.5,
                        fill: false
                    },
                    {
                        label: 'LSTM Forecast',
                        data: forecastSeries,
                        borderColor: '#e11d48', // Vibrant Coral / Pink
                        borderWidth: 3.0,
                        borderDash: [6, 4],
                        tension: 0.25,
                        pointRadius: 5,
                        pointBackgroundColor: '#e11d48',
                        pointBorderColor: '#000000',
                        pointBorderWidth: 2,
                        fill: false
                    },
                    {
                        label: 'SDN Headroom Capacity',
                        data: headroomSeries,
                        borderColor: '#10b981', // Mint Green
                        borderWidth: 2.2,
                        stepped: true,
                        pointRadius: 3,
                        pointBackgroundColor: '#10b981',
                        fill: false
                    },
                    {
                        label: '95% Confidence Upper',
                        data: upperBoundSeries,
                        borderColor: 'transparent',
                        backgroundColor: 'rgba(225, 29, 72, 0.12)',
                        pointRadius: 0,
                        fill: '+1'
                    },
                    {
                        label: '95% Confidence Lower',
                        data: lowerBoundSeries,
                        borderColor: 'transparent',
                        pointRadius: 0,
                        fill: false
                    },
                    {
                        label: 'Link Capacity Limit',
                        data: capacitySeries,
                        borderColor: '#f43f5e',
                        borderWidth: 2,
                        borderDash: [4, 4],
                        pointRadius: 0,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#ffffff',
                        titleColor: '#000000',
                        bodyColor: '#1f2937',
                        borderColor: '#000000',
                        borderWidth: 2.5,
                        padding: 12,
                        boxPadding: 6,
                        usePointStyle: true,
                        callbacks: {
                            label: function(context) {
                                if (context.raw === null || context.raw === undefined) return null;
                                return `${context.dataset.label}: ${context.raw} Gbps`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(0, 0, 0, 0.05)' },
                        ticks: {
                            color: '#000000',
                            font: { weight: '700', size: 10 },
                            maxTicksLimit: 14
                        }
                    },
                    y: {
                        grid: { color: 'rgba(0, 0, 0, 0.06)' },
                        title: {
                            display: true,
                            text: 'Bandwidth Throughput (Gbps)',
                            color: '#000000',
                            font: { size: 12, weight: '800' }
                        },
                        ticks: {
                            color: '#000000',
                            font: { weight: '700' }
                        },
                        min: 0,
                        suggestedMax: capacity * 1.05
                    }
                }
            }
        });
    }

    // Render Step-by-Step Accuracy Verification Cards
    function renderStepVerifications(data) {
        elements.stepsChipsGrid.innerHTML = "";
        
        if (data.mode === "test_eval" && data.step_evaluations && data.step_evaluations.length > 0) {
            elements.verifyModeTag.textContent = "Ground Truth Test Verification";
            elements.verifyModeTag.style.backgroundColor = "var(--pastel-yellow)";

            data.step_evaluations.forEach(step => {
                const isAccurate = step.status === "ACCURATE";
                const card = document.createElement("div");
                card.className = "step-chip";
                card.style.borderLeft = isAccurate ? "4px solid #10b981" : "4px solid #f87171";

                card.innerHTML = `
                    <div class="step-chip-head">
                        <span>Step ${step.step} (${step.timestamp})</span>
                        <span style="color: ${isAccurate ? '#15803d' : '#b91c1c'};">${isAccurate ? '✅ Accurate' : '⚠️ Alert'}</span>
                    </div>
                    <div class="step-chip-body">
                        <span>Actual: <strong>${step.actual_gbps}</strong> G</span>
                        <span>LSTM: <strong>${step.forecast_gbps}</strong> G</span>
                        <span>Δ ${step.error_gbps > 0 ? '+' : ''}${step.error_gbps} G (${step.error_pct}%)</span>
                    </div>
                `;
                elements.stepsChipsGrid.appendChild(card);
            });
        } else {
            elements.verifyModeTag.textContent = "Future Horizon Mode (Projection)";
            elements.verifyModeTag.style.backgroundColor = "var(--pastel-blue)";

            data.forecast.values_gbps.forEach((val, idx) => {
                const card = document.createElement("div");
                card.className = "step-chip";
                card.innerHTML = `
                    <div class="step-chip-head">
                        <span>Step ${idx + 1} (${data.forecast.timestamps[idx]})</span>
                        <span style="color: #0284c7;">Projected</span>
                    </div>
                    <div class="step-chip-body">
                        <span>Predicted: <strong>${val}</strong> Gbps</span>
                        <span>Headroom: <strong>${data.advisor.steps_plan[idx]?.recommended_allocation_gbps || '--'}</strong> G</span>
                    </div>
                `;
                elements.stepsChipsGrid.appendChild(card);
            });
        }
    }

    // Render Academic Benchmark Chart
    function renderBenchmarkChart(benchmarks) {
        const ctx = document.getElementById("benchmarkChart").getContext("2d");

        const models = Object.keys(benchmarks);
        const rmseValues = models.map(m => benchmarks[m].metrics.rmse);
        const maeValues = models.map(m => benchmarks[m].metrics.mae);

        const bgColorsRmse = models.map(m => m.includes("LSTM") ? '#fde047' : '#93c5fd');
        const bgColorsMae = models.map(m => m.includes("LSTM") ? '#fca5a5' : '#bbf7d0');

        if (benchmarkChart) {
            benchmarkChart.destroy();
        }

        benchmarkChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: models.map(m => m.split(' ')[0]),
                datasets: [
                    {
                        label: 'RMSE (Gbps)',
                        data: rmseValues,
                        backgroundColor: bgColorsRmse,
                        borderColor: '#000000',
                        borderWidth: 2,
                        borderRadius: 6
                    },
                    {
                        label: 'MAE (Gbps)',
                        data: maeValues,
                        backgroundColor: bgColorsMae,
                        borderColor: '#000000',
                        borderWidth: 2,
                        borderRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: '#000000',
                            font: { weight: '700', size: 11 }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { color: '#000000', font: { weight: '800' } }
                    },
                    y: {
                        grid: { color: 'rgba(0,0,0,0.06)' },
                        ticks: { color: '#000000', font: { weight: '700' } },
                        title: { display: true, text: 'Error (Gbps)', color: '#000000', font: { weight: '800' } }
                    }
                }
            }
        });
    }

    // Render Academic Benchmark Table
    function renderBenchmarkTable(benchmarks) {
        elements.benchmarkTableBody.innerHTML = "";

        for (const [modelName, details] of Object.entries(benchmarks)) {
            const m = details.metrics;
            const tr = document.createElement("tr");
            if (modelName.includes("LSTM")) {
                tr.classList.add("highlight-tr");
            }

            tr.innerHTML = `
                <td><strong>${modelName}</strong></td>
                <td><span>${m.rmse.toFixed(3)}</span></td>
                <td><span>${m.mae.toFixed(3)}</span></td>
                <td><span>${m.mape_pct.toFixed(1)}%</span></td>
                <td><span>${m.r2_score.toFixed(3)}</span></td>
                <td><span>${m.peak_traffic_mae.toFixed(3)}</span></td>
                <td><span style="color: ${m.under_prediction_rate_pct > 50 ? '#b91c1c' : '#15803d'}; font-weight: 800;">${m.under_prediction_rate_pct.toFixed(1)}%</span></td>
                <td><span>${m.inference_latency_ms ? m.inference_latency_ms + ' ms' : '< 1 ms'}</span></td>
            `;
            elements.benchmarkTableBody.appendChild(tr);
        }
    }

    // Setup Event Listeners
    function setupEventListeners() {
        // Dataset profile pills
        const datasetPills = [
            { el: elements.pillGeant, key: "geant_backbone" },
            { el: elements.pillIsp, key: "isp_backbone" },
            { el: elements.pillCloud, key: "cloud_egress" },
            { el: elements.pillDc, key: "datacenter_fabric" }
        ];

        datasetPills.forEach(({ el, key }) => {
            if (!el) return;
            el.addEventListener("click", () => {
                datasetPills.forEach(p => p.el.classList.remove("active"));
                el.classList.add("active");
                state.dataset = key;
                refreshAllData();
            });
        });

        // Mode toggling
        elements.modeTestEval.addEventListener("click", () => {
            elements.modeTestEval.classList.add("active");
            elements.modeFuture.classList.remove("active");
            state.mode = "test_eval";
            fetchForecast();
        });

        elements.modeFuture.addEventListener("click", () => {
            elements.modeFuture.classList.add("active");
            elements.modeTestEval.classList.remove("active");
            state.mode = "future_projection";
            fetchForecast();
        });

        // Quick Horizon buttons
        document.querySelectorAll("[data-h]").forEach(btn => {
            btn.addEventListener("click", () => {
                document.querySelectorAll("[data-h]").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                state.horizon = parseInt(btn.dataset.h);
                elements.horizonRange.value = state.horizon;
                elements.horizonVal.textContent = `${state.horizon} Hours`;
                fetchForecast();
            });
        });

        // Range Sliders
        elements.horizonRange.addEventListener("input", (e) => {
            state.horizon = parseInt(e.target.value);
            elements.horizonVal.textContent = `${state.horizon} Hours`;
        });

        elements.headroomRange.addEventListener("input", (e) => {
            state.headroomPct = parseInt(e.target.value) / 100.0;
            elements.headroomVal.textContent = `${e.target.value}%`;
        });

        elements.offsetRange.addEventListener("input", (e) => {
            state.testOffset = parseInt(e.target.value);
            elements.offsetVal.textContent = state.testOffset === 0 ? "Latest Window" : `-${state.testOffset}h Ago`;
        });

        // Apply Params Button
        elements.applyParamsBtn.addEventListener("click", () => {
            fetchForecast();
        });

        // Run Predictor Button
        elements.runInferenceBtn.addEventListener("click", () => {
            refreshAllData();
        });

        // Surge Simulation Modal (Image 3 design)
        elements.surgeBtn.addEventListener("click", () => {
            elements.surgeModal.classList.remove("hidden");
        });

        elements.cancelModalBtn.addEventListener("click", () => {
            elements.surgeModal.classList.add("hidden");
        });

        elements.surgeTypeSelect.addEventListener("change", (e) => {
            const val = e.target.value;
            if (val === "flash_crowd") {
                elements.surgeModalHighlight.textContent = '"Flash-Crowd Surge (+35 Gbps)"';
            } else if (val === "distributed_sync") {
                elements.surgeModalHighlight.textContent = '"Distributed DB Sync (+55 Gbps)"';
            } else if (val === "ddos_attack") {
                elements.surgeModalHighlight.textContent = '"Volumetric DDoS (+80 Gbps)"';
            }
        });

        elements.executeSurgeBtn.addEventListener("click", async () => {
            const type = elements.surgeTypeSelect.value;
            let mag = 35.0;
            if (type === "distributed_sync") mag = 55.0;
            if (type === "ddos_attack") mag = 80.0;

            elements.surgeModal.classList.add("hidden");
            elements.executeSurgeBtn.disabled = true;

            try {
                const res = await fetch("/api/simulate-surge", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        dataset: state.dataset,
                        surge_magnitude: mag,
                        surge_duration_steps: 3,
                        surge_type: type
                    })
                });

                const data = await res.json();
                state.currentForecast = data;

                // IMMEDIATELY render the surge on the chart and update all KPIs & verifications!
                renderMainForecastChart(data);
                updateKpisAndAdvisor(data);
                renderStepVerifications(data);

                // Flash alert styling
                elements.kpiSlaStatus.style.transform = "scale(1.1)";
                setTimeout(() => {
                    elements.kpiSlaStatus.style.transform = "scale(1)";
                }, 400);

            } catch (err) {
                console.error("Surge execution failed:", err);
            } finally {
                elements.executeSurgeBtn.disabled = false;
            }
        });

        // Reset Telemetry Button
        if (elements.resetTelemetryBtn) {
            elements.resetTelemetryBtn.addEventListener("click", async () => {
                elements.resetTelemetryBtn.disabled = true;
                elements.resetTelemetryBtn.textContent = "Resetting...";

                try {
                    const res = await fetch(`/api/reset-telemetry?dataset=${state.dataset}`, {
                        method: "POST"
                    });
                    const data = await res.json();
                    state.currentForecast = data;

                    renderMainForecastChart(data);
                    updateKpisAndAdvisor(data);
                    renderStepVerifications(data);
                } catch (err) {
                    console.error("Reset telemetry failed:", err);
                } finally {
                    elements.resetTelemetryBtn.disabled = false;
                    elements.resetTelemetryBtn.textContent = "🔄 Reset Telemetry";
                }
            });
        }

        // Theory Modal
        elements.toggleTheoryBtn.addEventListener("click", () => {
            elements.theoryModal.classList.remove("hidden");
        });

        elements.closeTheoryBtn.addEventListener("click", () => {
            elements.theoryModal.classList.add("hidden");
        });
    }

    // Start
    init();
});
