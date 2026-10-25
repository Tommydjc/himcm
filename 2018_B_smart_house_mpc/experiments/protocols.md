# Experiment Protocols — 2018 HiMCM Problem B (AeroTherm-MPC)

## Dual-track rule (do not mix tables)
| Track | Specs | Purpose |
| --- | --- | --- |
| **Teaching (T)** | `building_specs_teaching.json`, later `tou_tariffs_v2.json` | 24h Bang-Bang vs prediction-matrix MPC (paper main text) |
| **Legacy (L)** | `building_specs.json`, `tou_tariffs.json` | Week-slice multi-physics / Nest–Ecobee / Streamlit Tab B |

Phase status:
- **Phase 1 DONE** — `building_rc.py` (2R2C RK4 Δt=60s, COP=3.8−0.04(T−25)).
- **Phase 2 DONE** — `comfort_tariffs/models.py` + `tou_tariffs_v2.json`
  (PMV penalty outside |PMV|>0.5; TOU 0.12 / 0.22 / **0.48**).
- **Phase 3 DONE** — `baseline_thermostat/bang_bang.py` + `weather_summer_day_1min.csv`
  → `results/baseline_bang_bang_dynamics.csv` (1440 min). Measured peak-cost share
  ~43% (do **not** invent ≥60%); short-cycle starts = 42.
- **Phase 4 DONE** — `state_space_2r2c.py` + `prediction_matrices.py`
  (ZOH Ts=900s; \(\mathbf{X}=\mathcal{A}x_0+\mathcal{B}U+\mathcal{E}D\);
  mat vs sim max|err| ≈ 7e-15).
- **Phase 5 DONE** — `mpc_controller.py` rolling MPC (H=24, Ts=15min) vs Bang-Bang.
  Locked: `mpc_optimal_dynamics.csv`, `bang_vs_mpc_summary.json`,
  `fig_bang_vs_mpc_day.png`. Measured: cost −4.2%, peak energy −15.8%,
  compressor starts 42→3; prepeak ū=0.40 vs peak ū=0.17.
- **Phase 6 DONE** — `eval_harness/run_logger.py` JSONL + Streamlit 3 tabs
  (Teaching / Legacy / Harness) + paper refresh from `bang_vs_mpc_summary.json`.
  Nest−64.6% relegated to Appendix (Legacy 36h). Launch:
  `PYTHONPATH=src streamlit run src/ui/interactive_app.py --server.port 8507
  --server.fileWatcherType none`.
- **Prompt-5 Harness DONE** — `eval_harness/benchmark_harness.py`
  (scenarios A–E; M1–M4; `harness_benchmark_results.csv`;
  `fig_harness_radar_benchmark.png`, `fig_temperature_tracking_comparison.png`).
- **Prompt-6 Cognitive MAS DONE** — `agent_inhabitant/cognitive_thermostat.py`
  (Ollama qwen2.5:7b + Mock; Pydantic consensus → `MPCComfortConfig`;
  `results/agent_thermostat_dialogue.json`).

## Locked design rationale
1. Multi-physics ODEs: temperature (2R2C) + humidity + allergen/PM dynamics.
2. Thermal inertia as virtual storage for TOU pre-cool / peak shedding.
3. Irregular arrival = geofence ETA ⊕ calendar prior → dynamic lead time.
4. Multi-occupant conflict = asymmetric Nash / MAS bargain (not mean setpoint).
5. Deliverable for Req 3 = 1-page AP-style News Release.

## Scenarios
| ID | Description | Weather | Occupancy |
| --- | --- | --- | --- |
| S0 | Free-float hot/humid | T_out=32°C, RH=75% | empty |
| S1 | Summer week slice | `weather_tmy3_summer.csv` hours 0–47 | 3 inhabitants + irregular returns |
| S2 | Heatwave | S1 + ΔT=+4°C | same |
| S3 | Tariff surge | S1 rates ×1.8 | same |
| S4 | Sensor drift | evaluate MPC traj with +1.2°C bias | same |
| S5 | Multi-zone | 5-zone Laplacian + VAV | zone-mapped occupancy |

## Controllers (shared physics)
- programmable, remote_app, nest_like, ecobee_like, aerotherm_mpc

## KPIs
- energy_kWh, cost_USD, peak_energy_kWh (14:00–19:00)
- discomfort degree-hours, allergen-over hours (C_pol>35), RH>60 hours

## Reproduction
```bash
cd 2018_B_smart_house_mpc
# Phase 1 teaching thermal core
PYTHONPATH=src /opt/anaconda3/bin/python -m thermal_physics.building_rc
PYTHONPATH=src /opt/anaconda3/bin/python -m pytest tests/test_building_rc.py -q
# Phase 2 comfort + TOU v2
PYTHONPATH=src /opt/anaconda3/bin/python -m comfort_tariffs.models
PYTHONPATH=src /opt/anaconda3/bin/python -m pytest tests/test_comfort_tou_v2.py -q
# Phase 3 Bang-Bang summer day
PYTHONPATH=src /opt/anaconda3/bin/python -m baseline_thermostat.bang_bang
PYTHONPATH=src /opt/anaconda3/bin/python -m pytest tests/test_bang_bang.py -q
# Phase 4 prediction matrices (MPC_02)
PYTHONPATH=src /opt/anaconda3/bin/python -m control_mpc.prediction_matrices
PYTHONPATH=src /opt/anaconda3/bin/python -m pytest tests/test_prediction_matrices.py -q
# Phase 5 rolling MPC vs Bang-Bang
PYTHONPATH=src /opt/anaconda3/bin/python -m control_mpc.mpc_controller
PYTHONPATH=src /opt/anaconda3/bin/python -m pytest tests/test_mpc_controller.py -q
# Phase 6 JSONL logger + Streamlit Teaching Tab
PYTHONPATH=src /opt/anaconda3/bin/python -m eval_harness.run_logger
PYTHONPATH=src /opt/anaconda3/bin/python -m pytest tests/test_run_logger.py -q
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src streamlit run src/ui/interactive_app.py \
  --server.port 8507 --server.fileWatcherType none
# Prompt-5 black-box stress harness (Bang vs MPC, scenarios A–E)
PYTHONPATH=src /opt/anaconda3/bin/python -m eval_harness.benchmark_harness
PYTHONPATH=src /opt/anaconda3/bin/python -m pytest tests/test_benchmark_harness.py -q
# Prompt-6 cognitive MAS (Ollama or Mock) → MPC comfort band
PYTHONPATH=src /opt/anaconda3/bin/python -m agent_inhabitant.cognitive_thermostat
PYTHONPATH=src /opt/anaconda3/bin/python -m pytest tests/test_cognitive_thermostat.py -q
# Legacy week pipeline
PYTHONPATH=src /opt/anaconda3/bin/python -m scripts.run_full_pipeline
PYTHONPATH=src /opt/anaconda3/bin/python -m pytest tests/test_physics_and_mpc.py -q
```

## Evidence rule
Paper tables must cite `results/*.csv` / locked `*.json` only. No invented percentages.

## Paper compile (from `paper/`)
```bash
cd paper
pdflatex himcm_paper.tex
pdflatex summary_sheet.tex
pdflatex homeowner_advisory_letter.tex
pdflatex ai_disclosure.tex
```
Primary KPIs: `bang_vs_mpc_summary.json`, `harness_benchmark_results.csv`,
`agent_thermostat_dialogue.json`. Figures: `fig_bang_vs_mpc_day.png`,
`fig_temperature_tracking_comparison.png`, `fig_harness_radar_benchmark.png`.
