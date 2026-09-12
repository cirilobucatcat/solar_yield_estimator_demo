"""
Seeds CityClimateProfile (monthly solar/temperature lookup) and
CityUtilityProfile (rate + tilt per city) so Day 5 inference has
something to read from.

Run with:  python manage.py load_city_data

Sourcing, honestly stated:
- Monthly peak-sun-hours (PSH, ~= kWh/m^2/day) for Manila, Cebu, Davao,
  and Iloilo, March-November, are taken from a Philippine solar ROI
  reference citing NASA POWER satellite reanalysis (pinas.solar,
  accessed Sep 2026). December-February aren't published there, so
  they're linearly interpolated between November and March.
- Cagayan de Oro, General Santos, Zamboanga only had a rainy-season and
  dry-season PSH *range* (no per-month figures) from the same source.
  Their monthly curve is Iloilo's shape rescaled to fit each city's own
  range -- a reasonable approximation, not a direct measurement.
- Bacolod and Quezon City had no independent PSH data at all: Bacolod
  reuses Iloilo's curve (same general region/climate type), Quezon City
  reuses Manila's (same metro area).
- Ambient temperatures are generic lowland-tropical (~26-30 C) vs.
  Baguio's highland climate (~15-20 C) -- common-knowledge approximations,
  not sourced per-city, since no city-level monthly temperature data was
  found.
- Electricity rates and optimal tilt: see CityUtilityProfile.
Every estimated (as opposed to directly sourced) value is flagged
is_estimated=True / rate_is_estimated=True so it's never confused with
the real thing later.
"""

from django.core.management.base import BaseCommand

from estimator.models import CityClimateProfile, CityUtilityProfile

# --- Sourced monthly PSH, March(3) through November(11) ---
SOURCED_PSH = {
    "manila": {3: 6.01, 4: 6.31, 5: 5.61, 6: 4.72, 7: 4.52, 8: 4.48, 9: 4.69, 10: 4.83, 11: 4.51},
    "cebu_city": {3: 6.42, 4: 6.61, 5: 5.90, 6: 5.18, 7: 4.95, 8: 5.03, 9: 5.31, 10: 5.48, 11: 5.28},
    "davao_city": {3: 6.15, 4: 6.30, 5: 5.68, 6: 5.01, 7: 4.88, 8: 4.95, 9: 5.18, 10: 5.35, 11: 5.22},
    "iloilo_city": {3: 6.38, 4: 6.55, 5: 5.85, 6: 5.12, 7: 4.89, 8: 4.97, 9: 5.24, 10: 5.41, 11: 5.15},
}
# Cities reusing another city's exact curve (same metro/region).
COPIES_CURVE_OF = {
    "quezon_city": "manila",
    "bacolod_city": "iloilo_city",
}
# Cities with only a (rainy_min, rainy_max, dry_min, dry_max) range --
# Iloilo's shape gets linearly rescaled to fit.
RANGE_ONLY = {
    "cagayan_de_oro": {"rainy": (4.82, 5.30), "dry": (5.62, 6.28)},
    "general_santos": {"rainy": (4.90, 5.38), "dry": (5.73, 6.42)},
    "zamboanga_city": {"rainy": (5.02, 5.42), "dry": (5.78, 6.48)},
}
# Baguio's mountain-fog climate doesn't resemble the lowland cities, so
# it gets its own explicit (still approximate) monthly shape instead of
# being rescaled from Iloilo.
BAGUIO_PSH = {3: 5.4, 4: 5.6, 5: 5.0, 6: 4.2, 7: 3.85, 8: 3.9, 9: 4.1, 10: 4.3, 11: 4.0}

LOWLAND_TEMP_C = {1: 26.5, 2: 27.0, 3: 28.5, 4: 29.5, 5: 29.5, 6: 28.5,
                  7: 28.0, 8: 28.0, 9: 28.0, 10: 27.5, 11: 27.0, 12: 26.5}
HIGHLAND_TEMP_C = {1: 16.0, 2: 16.5, 3: 18.0, 4: 19.5, 5: 20.0, 6: 19.0,
                    7: 18.0, 8: 18.0, 9: 18.0, 10: 17.5, 11: 17.0, 12: 16.0}

UTILITY_DATA = {
    # city: (utility_name, rate_php_per_kwh, optimal_tilt_deg, rate_is_estimated)
    "manila": ("Meralco", 14.78, 7.5, False),
    "quezon_city": ("Meralco", 14.78, 7.5, False),
    "cebu_city": ("VECO", 14.96, 25.0, False),
    "davao_city": ("DLPC", 13.09, 7.5, False),
    "iloilo_city": ("MORE Power", 14.45, 10.0, False),
    "bacolod_city": ("CENECO", 14.28, 10.0, False),
    "baguio_city": ("BENECO", 12.29, 10.0, False),
    "cagayan_de_oro": ("CEPALCO", 12.50, 10.0, True),
    "zamboanga_city": ("ZAMCELCO", 12.50, 10.0, True),
    "general_santos": ("SOCOTECO II", 12.50, 10.0, True),
}


def rescale_to_range(curve: dict, new_min: float, new_max: float) -> dict:
    old_min, old_max = min(curve.values()), max(curve.values())
    span = old_max - old_min
    return {
        month: new_min + (value - old_min) / span * (new_max - new_min)
        for month, value in curve.items()
    }


def fill_dec_jan_feb(curve: dict) -> dict:
    """Linearly interpolate the 3 unsourced months between Nov and Mar."""
    nov, mar = curve[11], curve[3]
    step = (mar - nov) / 4  # Nov -> Dec -> Jan -> Feb -> Mar (4 steps)
    full = dict(curve)
    full[12] = round(nov + step, 2)
    full[1] = round(nov + 2 * step, 2)
    full[2] = round(nov + 3 * step, 2)
    return full


def build_all_city_curves():
    curves, estimated_months = {}, {}

    for city, months in SOURCED_PSH.items():
        full = fill_dec_jan_feb(months)
        curves[city] = full
        estimated_months[city] = {1, 2, 12}  # only the interpolated ones

    for city, source_city in COPIES_CURVE_OF.items():
        curves[city] = dict(curves[source_city])
        estimated_months[city] = set(range(1, 13))  # whole thing is borrowed

    iloilo_curve = SOURCED_PSH["iloilo_city"]
    for city, ranges in RANGE_ONLY.items():
        rescaled = rescale_to_range(iloilo_curve, ranges["rainy"][0], ranges["dry"][1])
        curves[city] = fill_dec_jan_feb(rescaled)
        estimated_months[city] = set(range(1, 13))

    curves["baguio_city"] = fill_dec_jan_feb(BAGUIO_PSH)
    estimated_months["baguio_city"] = {1, 2, 12}

    return curves, estimated_months


class Command(BaseCommand):
    help = "Seed CityClimateProfile and CityUtilityProfile with monthly solar/weather/rate data."

    def handle(self, *args, **options):
        curves, estimated_months = build_all_city_curves()

        climate_count = 0
        for city, monthly_psh in curves.items():
            is_highland = city == "baguio_city"
            temp_table = HIGHLAND_TEMP_C if is_highland else LOWLAND_TEMP_C
            for month, psh in monthly_psh.items():
                CityClimateProfile.objects.update_or_create(
                    city=city, month=month,
                    defaults={
                        "avg_peak_sun_hours": round(psh, 2),
                        "avg_ambient_temp_c": temp_table[month],
                        "is_estimated": month in estimated_months[city],
                    },
                )
                climate_count += 1

        utility_count = 0
        for city, (utility_name, rate, tilt, rate_estimated) in UTILITY_DATA.items():
            CityUtilityProfile.objects.update_or_create(
                city=city,
                defaults={
                    "utility_name": utility_name,
                    "electricity_rate_php_per_kwh": rate,
                    "optimal_tilt_deg": tilt,
                    "rate_is_estimated": rate_estimated,
                },
            )
            utility_count += 1

        self.stdout.write(self.style.SUCCESS(
            f"Loaded {climate_count} CityClimateProfile rows and "
            f"{utility_count} CityUtilityProfile rows."
        ))
