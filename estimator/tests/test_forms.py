import datetime

import pytest

from estimator.forms import SolarInputForm


def valid_data(**overrides):
    data = {
        "system_size_kw": "5.0",
        "panel_tilt_deg": "10.0",
        "city": "manila",
        "month": "6",
        "electricity_rate_override": "",
    }
    data.update(overrides)
    return data


class TestSolarInputForm:
    def test_valid_data_passes(self):
        form = SolarInputForm(data=valid_data())
        assert form.is_valid(), form.errors

    @pytest.mark.parametrize("size", ["0.5", "1000"])
    def test_system_size_at_form_boundaries_is_valid(self, size):
        form = SolarInputForm(data=valid_data(system_size_kw=size))
        assert form.is_valid(), form.errors

    @pytest.mark.parametrize("size", ["0.4", "1000.1", "-5"])
    def test_system_size_outside_boundaries_is_invalid(self, size):
        form = SolarInputForm(data=valid_data(system_size_kw=size))
        assert not form.is_valid()
        assert "system_size_kw" in form.errors

    @pytest.mark.parametrize("tilt", ["0", "90"])
    def test_tilt_at_boundaries_is_valid(self, tilt):
        form = SolarInputForm(data=valid_data(panel_tilt_deg=tilt))
        assert form.is_valid(), form.errors

    @pytest.mark.parametrize("tilt", ["-1", "91"])
    def test_tilt_outside_boundaries_is_invalid(self, tilt):
        form = SolarInputForm(data=valid_data(panel_tilt_deg=tilt))
        assert not form.is_valid()
        assert "panel_tilt_deg" in form.errors

    def test_invalid_city_choice_rejected(self):
        form = SolarInputForm(data=valid_data(city="atlantis"))
        assert not form.is_valid()
        assert "city" in form.errors

    def test_invalid_month_rejected(self):
        form = SolarInputForm(data=valid_data(month="13"))
        assert not form.is_valid()
        assert "month" in form.errors

    def test_electricity_rate_override_is_optional(self):
        form = SolarInputForm(data=valid_data(electricity_rate_override=""))
        assert form.is_valid(), form.errors
        assert form.cleaned_data["electricity_rate_override"] is None

    def test_electricity_rate_override_used_when_given(self):
        form = SolarInputForm(data=valid_data(electricity_rate_override="12.50"))
        assert form.is_valid(), form.errors
        assert float(form.cleaned_data["electricity_rate_override"]) == 12.50

    def test_month_default_is_current_month(self):
        form = SolarInputForm()
        # `initial` on a TypedChoiceField is a callable here; Django resolves
        # it lazily, so call it the same way the form does.
        initial = form.fields["month"].initial
        resolved = initial() if callable(initial) else initial
        assert resolved == datetime.date.today().month
