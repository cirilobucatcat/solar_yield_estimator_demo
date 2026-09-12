import math

import pytest

from estimator.inference import (
    estimate_daily_yield,
    estimate_annual_outlook,
    InferenceUnavailable,
    _tilt_efficiency_factor,
    _module_temperature,
    _simulate_irradiance_curve,
    DAYLIGHT_START_HOUR,
    DAYLIGHT_END_HOUR,
)


class TestTiltEfficiencyFactor:
    def test_tilt_at_optimal_gives_full_factor(self):
        assert _tilt_efficiency_factor(10.0, 10.0) == pytest.approx(1.0)

    def test_tilt_far_from_optimal_hits_floor(self):
        # 90 degrees off should be well past where cosine would go negative
        assert _tilt_efficiency_factor(100.0, 10.0) == pytest.approx(0.4)

    def test_symmetric_around_optimal(self):
        above = _tilt_efficiency_factor(25.0, 10.0)
        below = _tilt_efficiency_factor(-5.0, 10.0)
        assert above == pytest.approx(below)

    def test_never_below_floor(self):
        for tilt in range(0, 181, 10):
            assert _tilt_efficiency_factor(tilt, 10.0) >= 0.4


class TestModuleTemperature:
    def test_zero_irradiance_equals_ambient(self):
        assert _module_temperature(ambient_c=25.0, irradiance_kw_m2=0.0) == pytest.approx(25.0)

    def test_higher_irradiance_gives_higher_module_temp(self):
        low = _module_temperature(ambient_c=25.0, irradiance_kw_m2=0.3)
        high = _module_temperature(ambient_c=25.0, irradiance_kw_m2=1.0)
        assert high > low > 25.0


class TestSimulateIrradianceCurve:
    def test_integral_matches_target_peak_sun_hours(self):
        target = 5.5
        curve = _simulate_irradiance_curve(target)
        integral = sum(irr * 0.25 for _, irr in curve)  # kW/m^2 * 0.25h steps
        assert integral == pytest.approx(target, rel=0.02)

    def test_curve_is_zero_at_edges_and_positive_at_midday(self):
        curve = _simulate_irradiance_curve(5.5)
        first_hour, first_val = curve[0]
        assert first_val == pytest.approx(0.0, abs=0.05)
        midday = [irr for hour, irr in curve if abs(hour - 12.0) < 0.2][0]
        assert midday > 0

    def test_covers_expected_daylight_window(self):
        curve = _simulate_irradiance_curve(5.5)
        hours = [h for h, _ in curve]
        assert min(hours) == pytest.approx(DAYLIGHT_START_HOUR)
        assert max(hours) < DAYLIGHT_END_HOUR


@pytest.mark.django_db
class TestEstimateDailyYield:
    def test_happy_path_returns_positive_yield(self, city_data):
        est = estimate_daily_yield(
            system_size_kw=5.0, panel_tilt_deg=10.0,
            city="manila", month=4, electricity_rate_php_per_kwh=14.78,
        )
        assert est.predicted_daily_yield_kwh > 0
        assert 0 <= est.predicted_avg_capacity_factor <= 1.0
        assert est.estimated_monthly_php_savings > 0
        assert len(est.hourly_profile) > 0

    def test_larger_system_yields_proportionally_more(self, city_data):
        small = estimate_daily_yield(
            system_size_kw=1.0, panel_tilt_deg=10.0,
            city="manila", month=4, electricity_rate_php_per_kwh=14.78,
        )
        large = estimate_daily_yield(
            system_size_kw=10.0, panel_tilt_deg=10.0,
            city="manila", month=4, electricity_rate_php_per_kwh=14.78,
        )
        assert large.predicted_daily_yield_kwh == pytest.approx(
            small.predicted_daily_yield_kwh * 10, rel=0.01
        )

    def test_missing_climate_data_raises_inference_unavailable(self, db):
        # No city_data fixture used here -> lookup tables are empty.
        with pytest.raises(InferenceUnavailable):
            estimate_daily_yield(
                system_size_kw=5.0, panel_tilt_deg=10.0,
                city="manila", month=4, electricity_rate_php_per_kwh=14.78,
            )

    def test_tilt_at_optimal_beats_tilt_far_off(self, city_data):
        # Manila's optimal tilt (Day 4 data) is 7.5 degrees.
        at_optimal = estimate_daily_yield(
            system_size_kw=5.0, panel_tilt_deg=7.5,
            city="manila", month=4, electricity_rate_php_per_kwh=14.78,
        )
        far_off = estimate_daily_yield(
            system_size_kw=5.0, panel_tilt_deg=90.0,
            city="manila", month=4, electricity_rate_php_per_kwh=14.78,
        )
        assert at_optimal.predicted_daily_yield_kwh > far_off.predicted_daily_yield_kwh

    def test_zero_rate_gives_zero_savings_but_still_predicts_yield(self, city_data):
        est = estimate_daily_yield(
            system_size_kw=5.0, panel_tilt_deg=10.0,
            city="manila", month=4, electricity_rate_php_per_kwh=None,
        )
        assert est.predicted_daily_yield_kwh > 0
        assert est.estimated_monthly_php_savings == 0


@pytest.mark.django_db
class TestEstimateAnnualOutlook:
    def test_returns_twelve_months(self, city_data):
        breakdown, annual_kwh, annual_php = estimate_annual_outlook(
            system_size_kw=5.0, panel_tilt_deg=10.0,
            city="manila", electricity_rate_php_per_kwh=14.78,
        )
        assert len(breakdown) == 12
        assert {row["month"] for row in breakdown} == set(range(1, 13))

    def test_annual_totals_are_positive_and_consistent(self, city_data):
        breakdown, annual_kwh, annual_php = estimate_annual_outlook(
            system_size_kw=5.0, panel_tilt_deg=10.0,
            city="manila", electricity_rate_php_per_kwh=14.78,
        )
        assert annual_kwh > 0
        assert annual_php > 0
        assert annual_php == pytest.approx(sum(r["monthly_php"] for r in breakdown), rel=0.01)
