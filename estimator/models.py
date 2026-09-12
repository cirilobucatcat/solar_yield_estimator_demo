from django.core.validators import MinValueValidator, MaxValueValidator
from django.db import models

from .forms import PH_CITY_CHOICES, MONTH_CHOICES


class CityClimateProfile(models.Model):
    """
    Monthly solar/weather lookup used at inference time (Day 5) to turn a
    user's chosen city into the features the trained model actually needs
    (irradiation, ambient_temperature) -- since the training data (2 plants
    in India) has no city information at all.

    avg_peak_sun_hours is equivalent to kWh/m^2/day, i.e. the same unit as
    the model's "irradiation" feature once expressed as a daily total.

    Sourced from NASA POWER-derived monthly averages as published in a
    Philippine solar ROI reference (pinas.solar, accessed Sep 2026) for
    March-November; December-February are linear interpolations between
    November and March since that source didn't publish those months.
    is_estimated=True flags every interpolated/approximated row so it's
    obvious which numbers are sourced vs. filled in.
    """

    city = models.CharField(max_length=64, choices=PH_CITY_CHOICES)
    month = models.PositiveSmallIntegerField(choices=MONTH_CHOICES)
    avg_peak_sun_hours = models.FloatField(
        help_text="Equivalent daily solar irradiation, kWh/m^2/day."
    )
    avg_ambient_temp_c = models.FloatField(
        help_text="Typical ambient temperature for the month, °C."
    )
    is_estimated = models.BooleanField(
        default=False,
        help_text="True if this row is interpolated/approximated rather than directly sourced.",
    )

    class Meta:
        unique_together = ("city", "month")
        ordering = ["city", "month"]

    def __str__(self):
        return f"{self.get_city_display()} / {self.get_month_display()}: {self.avg_peak_sun_hours} kWh/m²/day"


class CityUtilityProfile(models.Model):
    """
    Per-city attributes that don't vary month to month: which utility
    serves the area, its current per-kWh rate (for the PHP savings
    calculation), and a recommended panel tilt.

    optimal_tilt_deg for Manila/Davao (~7.5°) and Cebu (~25°) comes from
    Palconit et al., "Optimal tilt angle for solar PV in the Philippines"
    (ISPRS Archives, 2024) -- low-latitude tropical sites often favor a
    flatter tilt than the naive "tilt = latitude" rule of thumb. Cities
    without dedicated research use a generic 10° near-equator estimate.

    electricity_rate_php_per_kwh values are illustrative snapshots (Sep
    2026) and WILL drift -- ERC-regulated rates are revised monthly.
    Treat these as defaults the form lets a user override, not ground truth.
    """

    city = models.CharField(max_length=64, choices=PH_CITY_CHOICES, unique=True)
    utility_name = models.CharField(max_length=64)
    electricity_rate_php_per_kwh = models.DecimalField(max_digits=6, decimal_places=2)
    optimal_tilt_deg = models.FloatField(default=10.0)
    rate_is_estimated = models.BooleanField(
        default=False,
        help_text="True if no city-specific rate was found and a national-average default was used.",
    )

    class Meta:
        ordering = ["city"]

    def __str__(self):
        return f"{self.get_city_display()} ({self.utility_name}): ₱{self.electricity_rate_php_per_kwh}/kWh"


class EstimationRecord(models.Model):
    """
    Stores each solar yield estimation a user runs, so results can be
    listed/revisited and later used to sanity-check the model in production.
    """

    # --- User inputs (Django Form -> here) ---
    system_size_kw = models.DecimalField(
        max_digits=6, decimal_places=2,
        validators=[MinValueValidator(0.5), MaxValueValidator(1000)],
        help_text="Installed capacity of the solar system, in kW.",
    )
    panel_tilt_deg = models.DecimalField(
        max_digits=4, decimal_places=1,
        validators=[MinValueValidator(0), MaxValueValidator(90)],
        help_text="Panel tilt angle in degrees from horizontal.",
    )
    city = models.CharField(
        max_length=64,
        choices=PH_CITY_CHOICES,
        help_text="Philippine city/province used to look up irradiation & weather features.",
    )

    # --- Assumptions used for this specific estimate (Day 5 fills these in) ---
    # Kept on the record itself, not just looked up live, so a past estimate
    # stays reproducible even after CityClimateProfile/CityUtilityProfile
    # rows are later updated (e.g. rates change monthly).
    month_used = models.PositiveSmallIntegerField(choices=MONTH_CHOICES, null=True, blank=True)
    electricity_rate_php_per_kwh = models.DecimalField(
        max_digits=6, decimal_places=2, null=True, blank=True,
        help_text="Rate actually used for this estimate's PHP savings figure.",
    )
    predicted_capacity_factor = models.FloatField(
        null=True, blank=True,
        help_text="Raw model output before scaling by system_size_kw -- useful for QA/Day 7.",
    )

    # --- Model outputs (filled in once inference runs, Day 5) ---
    predicted_daily_yield_kwh = models.FloatField(null=True, blank=True)
    estimated_monthly_php_savings = models.FloatField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.city} | {self.system_size_kw}kW @ {self.panel_tilt_deg}° ({self.created_at:%Y-%m-%d})"
