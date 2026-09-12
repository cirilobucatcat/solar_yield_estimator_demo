import pytest
from django.db.utils import IntegrityError

from estimator.models import CityClimateProfile, CityUtilityProfile, EstimationRecord


@pytest.mark.django_db
class TestCityClimateProfile:
    def test_str_includes_city_and_month(self):
        row = CityClimateProfile.objects.create(
            city="manila", month=4, avg_peak_sun_hours=6.31, avg_ambient_temp_c=29.5,
        )
        assert "Manila" in str(row)
        assert "April" in str(row)

    def test_city_month_unique_together(self):
        CityClimateProfile.objects.create(
            city="manila", month=4, avg_peak_sun_hours=6.31, avg_ambient_temp_c=29.5,
        )
        with pytest.raises(IntegrityError):
            CityClimateProfile.objects.create(
                city="manila", month=4, avg_peak_sun_hours=5.0, avg_ambient_temp_c=28.0,
            )

    def test_is_estimated_defaults_false(self):
        row = CityClimateProfile.objects.create(
            city="manila", month=4, avg_peak_sun_hours=6.31, avg_ambient_temp_c=29.5,
        )
        assert row.is_estimated is False


@pytest.mark.django_db
class TestCityUtilityProfile:
    def test_city_is_unique(self):
        CityUtilityProfile.objects.create(
            city="manila", utility_name="Meralco",
            electricity_rate_php_per_kwh=14.78, optimal_tilt_deg=7.5,
        )
        with pytest.raises(IntegrityError):
            CityUtilityProfile.objects.create(
                city="manila", utility_name="Someone Else",
                electricity_rate_php_per_kwh=10.0, optimal_tilt_deg=10.0,
            )


@pytest.mark.django_db
class TestEstimationRecord:
    def test_create_minimal_record(self):
        record = EstimationRecord.objects.create(
            system_size_kw=5.0, panel_tilt_deg=10.0, city="iloilo_city",
        )
        assert record.predicted_daily_yield_kwh is None
        assert record.pk is not None

    def test_ordering_is_most_recent_first(self):
        first = EstimationRecord.objects.create(
            system_size_kw=1.0, panel_tilt_deg=10.0, city="manila",
        )
        second = EstimationRecord.objects.create(
            system_size_kw=2.0, panel_tilt_deg=10.0, city="manila",
        )
        records = list(EstimationRecord.objects.all())
        assert records[0].pk == second.pk
        assert records[1].pk == first.pk
