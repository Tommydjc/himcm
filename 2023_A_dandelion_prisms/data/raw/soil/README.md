# 地表土壤含水量

## 情景（`*_smi.csv`）

列：`date, smi`。`smi` ∈ [0, 1] 为无量纲 Soil Moisture Index。

月尺度取自 `climate_monthly.json`，日内保持常数。干旱扰动在实验层乘以 `smi_scale`，不改本目录情景文件。

## Open-Meteo 2022（`*_soil.csv`）

列：`date, soil_moisture`。单位 m³/m³（0–7 cm 日均 `soil_moisture_0_to_7cm_mean`）。
由 `scripts/download_real_climate.py` 与天气表同一请求拉取。这是体积含水量，不是情景 `smi`；接入仿真前须单独归一。
