from django.contrib import admin

from .models import EstimationRecord, CityClimateProfile, CityUtilityProfile


@admin.register(EstimationRecord)
class EstimationRecordAdmin(admin.ModelAdmin):
    list_display = (
        "city",
        "system_size_kw",
        "panel_tilt_deg",
        "predicted_daily_yield_kwh",
        "created_at",
    )
    list_filter = ("city",)
    ordering = ("-created_at",)


@admin.register(CityClimateProfile)
class CityClimateProfileAdmin(admin.ModelAdmin):
    list_display = ("city", "month", "avg_peak_sun_hours", "avg_ambient_temp_c", "is_estimated")
    list_filter = ("city", "is_estimated")
    ordering = ("city", "month")


@admin.register(CityUtilityProfile)
class CityUtilityProfileAdmin(admin.ModelAdmin):
    list_display = ("city", "utility_name", "electricity_rate_php_per_kwh",
                     "optimal_tilt_deg", "rate_is_estimated")
    list_filter = ("rate_is_estimated",)
    ordering = ("city",)
