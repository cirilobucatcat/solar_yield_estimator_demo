import pytest
from django.urls import reverse

from estimator.models import EstimationRecord


@pytest.mark.django_db
class TestIndexView:
    def test_get_renders_empty_form(self, client):
        response = client.get(reverse("estimator:index"))
        assert response.status_code == 200
        assert b"Estimate your daily solar yield" in response.content

    def test_valid_post_creates_record_and_shows_prediction(self, client, city_data):
        response = client.post(reverse("estimator:index"), data={
            "system_size_kw": "5.0",
            "panel_tilt_deg": "10.0",
            "city": "iloilo_city",
            "month": "9",
            "electricity_rate_override": "",
        })
        assert response.status_code == 200
        assert EstimationRecord.objects.count() == 1
        record = EstimationRecord.objects.first()
        assert record.city == "iloilo_city"
        assert record.predicted_daily_yield_kwh is not None
        assert record.predicted_daily_yield_kwh > 0
        assert float(record.electricity_rate_php_per_kwh) == pytest.approx(14.45)  # Iloilo/MORE Power default

    def test_rate_override_is_used_instead_of_default(self, client, city_data):
        client.post(reverse("estimator:index"), data={
            "system_size_kw": "5.0",
            "panel_tilt_deg": "10.0",
            "city": "iloilo_city",
            "month": "9",
            "electricity_rate_override": "9.99",
        })
        record = EstimationRecord.objects.first()
        assert float(record.electricity_rate_php_per_kwh) == pytest.approx(9.99)

    def test_invalid_post_does_not_create_record(self, client, city_data):
        response = client.post(reverse("estimator:index"), data={
            "system_size_kw": "-5.0",  # invalid
            "panel_tilt_deg": "10.0",
            "city": "iloilo_city",
            "month": "9",
            "electricity_rate_override": "",
        })
        assert response.status_code == 200
        assert EstimationRecord.objects.count() == 0

    def test_missing_climate_data_shows_friendly_error_not_500(self, client, db):
        # No city_data fixture -> CityClimateProfile table is empty.
        response = client.post(reverse("estimator:index"), data={
            "system_size_kw": "5.0",
            "panel_tilt_deg": "10.0",
            "city": "iloilo_city",
            "month": "9",
            "electricity_rate_override": "",
        })
        assert response.status_code == 200
        assert EstimationRecord.objects.count() == 0
        messages = list(response.context["messages"])
        assert any("load_city_data" in str(m) for m in messages)

    def test_recent_estimates_appear_after_submission(self, client, city_data):
        for _ in range(3):
            client.post(reverse("estimator:index"), data={
                "system_size_kw": "5.0", "panel_tilt_deg": "10.0",
                "city": "manila", "month": "6", "electricity_rate_override": "",
            })
        response = client.get(reverse("estimator:index"))
        assert len(response.context["recent_estimates"]) == 3
