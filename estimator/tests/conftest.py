import pytest
from django.core.management import call_command


@pytest.fixture
def city_data(db):
    """Seeds CityClimateProfile + CityUtilityProfile, same as `manage.py load_city_data`."""
    call_command("load_city_data")
