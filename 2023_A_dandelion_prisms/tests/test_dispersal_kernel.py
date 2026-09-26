"""混合扩散核：沉降时间、离散归一、衰减前质量守恒、\(\Phi\) 边界。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.physics.dispersal_kernel import (
    OPEN_METEO_TEMPERATURE,
    OPEN_METEO_VWC,
    OPEN_METEO_WIND_FROM,
    OPEN_METEO_WIND_SPEED,
    GerminationParams,
    GridConfig,
    PappusAeroParams,
    calculate_germination_modifier,
    compute_effective_seed_flux,
    compute_fall_time,
    evaluate_bivariate_kernel,
    kernel_area_mass,
    temperature_from_record,
    volumetric_moisture_from_record,
    wind_vector_from_record,
)


class TestFallTime(unittest.TestCase):
    """\(\tau=H_0/v_t\\approx 0.8929\,\mathrm{s}\)。"""

    def test_nominal_svr_time(self) -> None:
        tau = compute_fall_time(0.25, 0.28)
        self.assertAlmostEqual(tau, 0.25 / 0.28, places=12)
        self.assertAlmostEqual(tau, 0.8928571428571428, places=10)

    def test_rejects_nonpositive(self) -> None:
        with self.assertRaises(ValueError):
            compute_fall_time(0.0, 0.28)


class TestKernelNormalization(unittest.TestCase):
    """\(\sum K\\Delta x\\Delta y=1\)，无 Python 逐格循环。"""

    def test_unit_mass_on_hectare(self) -> None:
        cfg = GridConfig()
        xs = cfg.x_min + (np.arange(100) + 0.5) * cfg.dx
        ys = cfg.y_min + (np.arange(100) + 0.5) * cfg.dy
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
        kernel = evaluate_bivariate_kernel(grid_x, grid_y, wind_vector=(2.0, -0.5))
        mass = float(np.sum(kernel) * cfg.dx * cfg.dy)
        np.testing.assert_allclose(mass, 1.0, atol=1.0e-12)
        self.assertTrue(np.all(kernel >= -1.0e-15))

    def test_calm_wind_still_normalizes(self) -> None:
        cfg = GridConfig()
        xs = np.linspace(-20.0, 20.0, 81)
        ys = np.linspace(-20.0, 20.0, 81)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
        kernel = evaluate_bivariate_kernel(grid_x, grid_y, wind_vector=(0.0, 0.0))
        dx = float(xs[1] - xs[0])
        dy = float(ys[1] - ys[0])
        np.testing.assert_allclose(np.sum(kernel) * dx * dy, 1.0, atol=1.0e-12)


class TestGerminationPhi(unittest.TestCase):
    """\(\Phi\\in[0,1]\)；渍水与过冷被抑制。"""

    def test_scalar_range(self) -> None:
        phi = calculate_germination_modifier(18.0, 0.20)
        self.assertIsInstance(phi, float)
        self.assertGreater(float(phi), 0.2)
        self.assertLessEqual(float(phi), 1.0)

    def test_waterlog_zero(self) -> None:
        phi = calculate_germination_modifier(20.0, 0.50)
        self.assertEqual(phi, 0.0)

    def test_vectorized_matches_scalar(self) -> None:
        temps = np.array([5.0, 18.0, 40.0])
        theta = np.array([0.20, 0.20, 0.20])
        out = calculate_germination_modifier(temps, theta)
        self.assertIsInstance(out, np.ndarray)
        assert isinstance(out, np.ndarray)
        for t, th, got in zip(temps, theta, out):
            self.assertAlmostEqual(float(got), float(calculate_germination_modifier(float(t), float(th))), places=12)


class TestEffectiveFlux(unittest.TestCase):
    """衰减前粒数守恒；Open-Meteo 列名可解析。"""

    def test_pre_attenuation_conservation(self) -> None:
        cfg = GridConfig()
        aero = PappusAeroParams()
        germ = GerminationParams()
        n_seeds = 180
        grid_x, grid_y, effective = compute_effective_seed_flux(
            n_seeds,
            (3.0, 0.0),
            temperature=18.0,
            soil_moisture=0.20,
            grid_cfg=cfg,
            aero=aero,
            germ=germ,
        )
        kernel = evaluate_bivariate_kernel(grid_x, grid_y, (3.0, 0.0), params=aero)
        landing = n_seeds * kernel
        np.testing.assert_allclose(kernel_area_mass(landing, cfg), float(n_seeds), atol=1.0e-8)
        phi = float(calculate_germination_modifier(18.0, 0.20, germ))
        np.testing.assert_allclose(effective, landing * phi, atol=1.0e-12)

    def test_open_meteo_row_bindings(self) -> None:
        weather_path = ROOT / "data" / "raw" / "weather" / "temperate_weather.csv"
        soil_path = ROOT / "data" / "raw" / "soil" / "temperate_soil.csv"
        if not weather_path.is_file() or not soil_path.is_file():
            self.skipTest("Open-Meteo CSVs not downloaded")
        weather = pd.read_csv(weather_path)
        soil = pd.read_csv(soil_path)
        self.assertIn(OPEN_METEO_WIND_SPEED, weather.columns)
        self.assertIn(OPEN_METEO_WIND_FROM, weather.columns)
        self.assertIn(OPEN_METEO_TEMPERATURE, weather.columns)
        self.assertIn(OPEN_METEO_VWC, soil.columns)
        row = {
            OPEN_METEO_WIND_SPEED: float(weather.iloc[150][OPEN_METEO_WIND_SPEED]),
            OPEN_METEO_WIND_FROM: float(weather.iloc[150][OPEN_METEO_WIND_FROM]),
            OPEN_METEO_TEMPERATURE: float(weather.iloc[150][OPEN_METEO_TEMPERATURE]),
            OPEN_METEO_VWC: float(soil.iloc[150][OPEN_METEO_VWC]),
        }
        wind = wind_vector_from_record(row)
        temp = temperature_from_record(row)
        theta = volumetric_moisture_from_record(row)
        _x, _y, field = compute_effective_seed_flux(180, wind, temp, theta, GridConfig())
        self.assertEqual(field.shape, (100, 100))
        self.assertTrue(np.all(field >= -1.0e-15))


if __name__ == "__main__":
    unittest.main()
