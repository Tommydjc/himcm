# 风速与气温

## 情景（`*_hourly.csv`）

列：`timestamp, wind_speed_mps, wind_from_deg, air_temp_c`。

由 `src/scripts/generate_forcing.py` 根据 `data/parameters/climate_monthly.json` 展开。
日变化是确定的正弦，**不是**机场 METAR。风向角为气象来向（0° = 北，顺时针）。

## Open-Meteo 2022（`*_weather.csv`）

列：`date, wind_speed, wind_direction, temperature`。

由 `scripts/download_real_climate.py` 从 [Archive API](https://archive-api.open-meteo.com/v1/archive) 拉取：
温带普林斯顿 (40.35, −74.66)、干旱凤凰城 (33.45, −112.07)、热带迈阿密 (25.76, −80.19)；
`2022-01-01`–`2022-12-31`；`wind_speed_10m_max`（m/s）、`wind_direction_10m_dominant`（°）、`temperature_2m_mean`（℃）。
格点与检索时刻见 `data/raw/open_meteo_2022_manifest.json`。不覆盖本目录情景小时文件。
