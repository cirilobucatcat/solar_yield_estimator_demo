import json

from django.contrib import messages
from django.shortcuts import render

from .forms import SolarInputForm
from .models import EstimationRecord, CityUtilityProfile
from .inference import estimate_daily_yield, estimate_annual_outlook, InferenceUnavailable


def index(request):
    """
    Renders the input form and, once submitted, runs the trained model
    (via estimator.inference) to produce a real daily kWh + PHP savings
    estimate, a 12-month outlook, and a breakdown of the assumptions used.
    """
    result = None
    hourly_profile = None
    breakdown = None
    monthly_outlook = None
    annual_kwh = annual_php = None

    if request.method == "POST":
        form = SolarInputForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data

            rate = data.get("electricity_rate_override")
            if rate is None:
                utility = CityUtilityProfile.objects.filter(city=data["city"]).first()
                rate = utility.electricity_rate_php_per_kwh if utility else None

            try:
                estimate = estimate_daily_yield(
                    system_size_kw=data["system_size_kw"],
                    panel_tilt_deg=data["panel_tilt_deg"],
                    city=data["city"],
                    month=data["month"],
                    electricity_rate_php_per_kwh=rate,
                )
                result = EstimationRecord.objects.create(
                    system_size_kw=data["system_size_kw"],
                    panel_tilt_deg=data["panel_tilt_deg"],
                    city=data["city"],
                    month_used=data["month"],
                    electricity_rate_php_per_kwh=estimate.electricity_rate_php_per_kwh,
                    predicted_capacity_factor=estimate.predicted_avg_capacity_factor,
                    predicted_daily_yield_kwh=estimate.predicted_daily_yield_kwh,
                    estimated_monthly_php_savings=estimate.estimated_monthly_php_savings,
                )
                hourly_profile = estimate.hourly_profile
                breakdown = estimate

                monthly_outlook, annual_kwh, annual_php = estimate_annual_outlook(
                    system_size_kw=data["system_size_kw"],
                    panel_tilt_deg=data["panel_tilt_deg"],
                    city=data["city"],
                    electricity_rate_php_per_kwh=estimate.electricity_rate_php_per_kwh,
                )
            except InferenceUnavailable as exc:
                messages.error(request, str(exc))
            except FileNotFoundError:
                messages.error(
                    request,
                    "Model file not found. Run ml_pipeline/train.py (Day 3) first to "
                    "generate ml_models/xgb_solar_yield_model.pkl.",
                )
    else:
        form = SolarInputForm()

    recent_estimates = EstimationRecord.objects.all()[:5]

    return render(request, "estimator/index.html", {
        "form": form,
        "result": result,
        "breakdown": breakdown,
        "hourly_profile_json": json.dumps(hourly_profile) if hourly_profile else "null",
        "monthly_outlook_json": json.dumps(monthly_outlook) if monthly_outlook else "null",
        "annual_kwh": annual_kwh,
        "annual_php": annual_php,
        "recent_estimates": recent_estimates,
    })
