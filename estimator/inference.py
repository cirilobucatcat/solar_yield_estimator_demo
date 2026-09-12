"""
Day 5: ML model integration.

Turns a user's (city, month, system_size_kw, panel_tilt_deg) into a
predicted daily yield in kWh and monthly PHP savings, using the Day 3
XGBoost model plus the Day 4 CityClimateProfile/CityUtilityProfile
lookup tables.

The trained model only knows instantaneous weather -> capacity_factor;
it has no notion of "a full day". So this module:

  1. Looks up the city's monthly avg peak-sun-hours (~kWh/m^2/day) and
     avg ambient temperature (Day 4 data).
  2. Simulates a 15-minute-resolution irradiance curve for a single
     representative day: a half-sine over an assumed 6am-6pm daylight
     window. The Philippines sits close enough to the equator that day
     length barely swings across the year, so a fixed window is a
     reasonable approximation rather than a real per-city sunrise/sunset
     calculation.
  3. Derives module temperature at each interval from ambient temp +
     irradiance via the standard NOCT (Nominal Operating Cell
     Temperature) formula, since the model was trained on measured
     module temperature but no per-city module-temp data exists for a
     new location:
         T_module = T_ambient + (NOCT - 20) / 800 * irradiance_W/m^2
  4. Runs the Day 3 model across all ~48 intervals to get a predicted
     capacity_factor curve, multiplies by system_size_kw to get a power
     curve in kW, and applies a physics-inspired tilt correction (cosine
     falloff from the city's known/estimated optimal tilt) since tilt
     was never a trained feature (Day 2/3 finding: zero variance in the
     training data -- see CityUtilityProfile docstring for where the
     optimal-tilt numbers come from).
  5. Integrates the power curve (kW * 0.25h per step) into a daily kWh
     total, and converts to a monthly PHP savings figure using the
     city's utility rate.
"""

import json
import math
from dataclasses import dataclass

import joblib
import pandas as pd
from django.conf import settings

from .models import CityClimateProfile, CityUtilityProfile

DAYLIGHT_START_HOUR = 6.0
DAYLIGHT_END_HOUR = 18.0
INTERVAL_HOURS = 0.25  # 15 minutes, matches the training data's grain
NOCT_C = 45.0  # typical crystalline-silicon datasheet value
DAYS_IN_MONTH = {  # non-leap approximation -- fine for an "average month" savings figure
    1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31,
}

_model = None
_feature_columns = None


class InferenceUnavailable(Exception):
    """Raised when required model or lookup data isn't available."""


def _load_model():
    global _model, _feature_columns
    if _model is None:
        _model = joblib.load(settings.ML_MODEL_PATH)
        meta = json.loads(settings.ML_FEATURE_COLUMNS_PATH.read_text())
        _feature_columns = meta["feature_columns"]
    return _model, _feature_columns


@dataclass
class YieldEstimate:
    predicted_daily_yield_kwh: float
    predicted_avg_capacity_factor: float
    estimated_monthly_php_savings: float
    electricity_rate_php_per_kwh: float
    tilt_efficiency_factor: float
    hourly_profile: list  # [{"hour": float, "power_kw": float}, ...] for charting
    avg_peak_sun_hours: float
    avg_ambient_temp_c: float
    optimal_tilt_deg: float


def _simulate_irradiance_curve(avg_peak_sun_hours: float):
    """
    Half-sine irradiance profile over [DAYLIGHT_START_HOUR, DAYLIGHT_END_HOUR]
    whose integral equals avg_peak_sun_hours (kWh/m^2/day).
    Integral of a half sine of amplitude I_peak over duration D is
    I_peak * D * (2/pi), so peak = avg_peak_sun_hours * pi / (2 * D).
    """
    daylight_duration = DAYLIGHT_END_HOUR - DAYLIGHT_START_HOUR
    peak_irradiance_kw_m2 = avg_peak_sun_hours * math.pi / (2 * daylight_duration)

    curve, t = [], DAYLIGHT_START_HOUR
    while t < DAYLIGHT_END_HOUR:
        fraction = (t - DAYLIGHT_START_HOUR) / daylight_duration
        irradiance = max(peak_irradiance_kw_m2 * math.sin(math.pi * fraction), 0.0)
        curve.append((t, irradiance))
        t += INTERVAL_HOURS
    return curve


def _module_temperature(ambient_c: float, irradiance_kw_m2: float) -> float:
    irradiance_w_m2 = irradiance_kw_m2 * 1000
    return ambient_c + (NOCT_C - 20) / 800 * irradiance_w_m2


def _tilt_efficiency_factor(panel_tilt_deg, optimal_tilt_deg: float) -> float:
    """
    Simplified, physics-inspired cosine falloff around the city's
    known/estimated optimal tilt -- NOT a rigorous solar-geometry
    calculation (that would need latitude, declination, and hour angle
    per interval). Floored at 0.4 so an extreme tilt doesn't zero out
    output entirely; real panels still pick up diffuse irradiance.
    """
    deviation_rad = math.radians(float(panel_tilt_deg) - optimal_tilt_deg)
    return max(math.cos(deviation_rad), 0.4)


def estimate_daily_yield(system_size_kw, panel_tilt_deg, city: str, month: int,
                          electricity_rate_php_per_kwh) -> YieldEstimate:
    model, feature_columns = _load_model()

    climate = CityClimateProfile.objects.filter(city=city, month=month).first()
    if climate is None:
        raise InferenceUnavailable(
            f"No climate data for city={city!r}, month={month!r}. Run "
            f"'python manage.py load_city_data' if you haven't yet."
        )

    utility = CityUtilityProfile.objects.filter(city=city).first()
    optimal_tilt = utility.optimal_tilt_deg if utility else 10.0

    system_size_kw = float(system_size_kw)
    tilt_factor = _tilt_efficiency_factor(panel_tilt_deg, optimal_tilt)
    irradiance_curve = _simulate_irradiance_curve(climate.avg_peak_sun_hours)

    rows = [
        {
            "irradiation": irradiance,
            "ambient_temperature": climate.avg_ambient_temp_c,
            "module_temperature": _module_temperature(climate.avg_ambient_temp_c, irradiance),
            "hour": int(hour),
            "month": month,
        }
        for hour, irradiance in irradiance_curve
    ]
    feature_df = pd.DataFrame(rows)[feature_columns]
    capacity_factors = [max(float(cf), 0.0) for cf in model.predict(feature_df)]

    hourly_profile, total_kwh = [], 0.0
    for (hour, _), cf in zip(irradiance_curve, capacity_factors):
        power_kw = cf * system_size_kw * tilt_factor
        total_kwh += power_kw * INTERVAL_HOURS
        hourly_profile.append({"hour": round(hour, 2), "power_kw": round(power_kw, 3)})

    avg_cf = sum(capacity_factors) / len(capacity_factors) if capacity_factors else 0.0
    rate = float(electricity_rate_php_per_kwh) if electricity_rate_php_per_kwh else 0.0
    monthly_savings = total_kwh * DAYS_IN_MONTH[month] * rate

    return YieldEstimate(
        predicted_daily_yield_kwh=round(total_kwh, 2),
        predicted_avg_capacity_factor=round(avg_cf, 4),
        estimated_monthly_php_savings=round(monthly_savings, 2),
        electricity_rate_php_per_kwh=rate,
        tilt_efficiency_factor=round(tilt_factor, 4),
        hourly_profile=hourly_profile,
        avg_peak_sun_hours=climate.avg_peak_sun_hours,
        avg_ambient_temp_c=climate.avg_ambient_temp_c,
        optimal_tilt_deg=optimal_tilt,
    )


def estimate_annual_outlook(system_size_kw, panel_tilt_deg, city: str,
                             electricity_rate_php_per_kwh):
    """
    Day 6: runs estimate_daily_yield for all 12 months (reusing the
    already-cached model) to drive the monthly outlook chart and an
    annual kWh/PHP summary.
    """
    from .forms import MONTH_CHOICES
    month_labels = dict(MONTH_CHOICES)

    breakdown = []
    for month in range(1, 13):
        try:
            est = estimate_daily_yield(
                system_size_kw=system_size_kw,
                panel_tilt_deg=panel_tilt_deg,
                city=city,
                month=month,
                electricity_rate_php_per_kwh=electricity_rate_php_per_kwh,
            )
        except InferenceUnavailable:
            continue
        breakdown.append({
            "month": month,
            "month_label": month_labels[month],
            "daily_kwh": est.predicted_daily_yield_kwh,
            "monthly_php": est.estimated_monthly_php_savings,
        })

    annual_kwh = sum(row["daily_kwh"] * DAYS_IN_MONTH[row["month"]] for row in breakdown)
    annual_php = sum(row["monthly_php"] for row in breakdown)
    return breakdown, round(annual_kwh, 1), round(annual_php, 2)
