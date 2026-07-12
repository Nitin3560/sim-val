# sim-val

**Quantitative Sim-to-Real Sensor Fidelity Analyzer for UAV Autonomy Stacks**

SimVal measures the sensor fidelity gap between Gazebo and Isaac Sim RTX for identical UAV scenarios. It compares camera rendering with PSNR, SSIM, and LPIPS, evaluates LiDAR scan fidelity where paired LiDAR data exists, and reports how those simulator differences affect TwinGuard's EKF trust and authority outputs.

Built to validate the simulation assumptions behind [TwinGuard](https://github.com/Nitin3560/TwinGuard-Swarm-Gazebo).

## Findings

Real values will be filled after the Isaac Sim GPU run and Gazebo SITL recordings are complete.

| Scenario | PSNR (dB) | SSIM | LPIPS | Authority Delta | Risk |
|---|---:|---:|---:|---:|---|
| hover_3m_nominal | — | — | — | — | pending GPU |
| circle_5m_nominal | — | — | — | — | pending GPU |
| hover_3m_camera_occlusion | — | — | — | — | pending GPU |

> **Key finding:** pending real Gazebo and Isaac Sim paired runs.

After reports are generated, plots are written to:

```text
outputs/plots/ssim_timeline_<scenario>.png
outputs/plots/metrics_comparison.png
outputs/plots/authority_comparison.png
```

## Why This Exists

TwinGuard's autonomy integrity thresholds are calibrated in simulation. Gazebo is useful and widely used, but its default sensor models are not the same as Isaac Sim RTX camera and LiDAR rendering. SimVal quantifies that simulator gap before relying on Gazebo-calibrated thresholds as if they transfer cleanly.

The output is a structured gap report:

```text
outputs/reports/gap_report_<scenario>.json
```

Each report includes trajectory alignment quality, camera fidelity metrics, optional LiDAR fidelity metrics, optional EKF propagation deltas, and a calibration recommendation.

## Architecture

```text
Gazebo / PX4 / ROS 2              Isaac Sim RTX
        |                              |
        v                              v
 extract_gazebo.py              extract_isaac.py
        |                              |
        +-------------+----------------+
                      v
              align_trajectories.py
                      |
        +-------------+--------------+
        v                            v
 camera_metrics.py             lidar_metrics.py
 PSNR / SSIM / LPIPS           DCD-inspired / density gap
        |                            |
        +-------------+--------------+
                      v
              run_pipeline.py
                      |
                      v
          gap_report_<scenario>.json
```

## Metrics

**Camera fidelity:** PSNR, SSIM, and LPIPS between aligned Gazebo and Isaac Sim RGB frames.

**LiDAR fidelity:** normalized bidirectional Chamfer distance with explicit point-density penalty. This is a DCD-inspired metric, not the exact Wu et al. formulation.

**Trajectory alignment:** position-norm cross-correlation and resampling so frames are compared at the same point in the mission.

**EKF propagation:** authority-scale and trust-state deltas after replaying Isaac Sim sensor outputs through the TwinGuard integrity pipeline.

## Current Scope

Gazebo recording uses `gz_x500_depth`, which provides RGB and depth but no ray-cast LiDAR. For this model, the report correctly writes:

```json
"lidar": null
```

LiDAR comparison requires a Gazebo model with a LiDAR sensor or an additional paired LiDAR recording.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest -v
```

Expected current suite:

```text
99 tests passing
```

Build Isaac USD scenes locally:

```bash
python -c "from isaac.build_scene import build_all_scenes; print(build_all_scenes())"
```

Run the comparison pipeline after Gazebo and Isaac data exist:

```bash
python -m pipeline.run_pipeline \
  --scenario hover_3m_nominal \
  --gazebo-bag outputs/gazebo/hover_3m_nominal \
  --isaac-dir outputs/isaac/hover_3m_nominal \
  --output-dir outputs/reports
```

Generate plots:

```bash
python -c "from report.plot_metrics import generate_all_plots; print(generate_all_plots())"
```

## Isaac Sim GPU Run

Run on an NVIDIA GPU instance with Isaac Sim installed:

```bash
/isaac-sim/python.sh isaac/run_scenario.py \
  config/scenario_hover_3m.yaml \
  outputs/isaac/hover_3m_nominal
```

Repeat for each scenario in `config/`.

## Repository Layout

```text
config/      Scenario YAMLs
gazebo/      ROS 2 bag extractor
isaac/       USD builder, Isaac runner, Isaac extractor
metrics/     Camera and LiDAR fidelity metrics
pipeline/    Alignment, EKF propagation, gap-report pipeline
report/      Plot generation
schema/      SensorRecord schema
tests/       Unit and integration tests
```

## Related

- [TwinGuard-Swarm-Gazebo](https://github.com/Nitin3560/TwinGuard-Swarm-Gazebo)
