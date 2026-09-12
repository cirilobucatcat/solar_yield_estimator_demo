"""
Day 7: sweep the full input space through estimate_daily_yield() and
flag anything that looks physically wrong -- negative output, capacity
factor outside [0, 1.05], implausible specific yield, etc. -- rather
than just spot-checking a handful of cases by hand.

Run with: python manage.py qa_sweep
"""

from django.core.management.base import BaseCommand

from estimator.forms import PH_CITY_CHOICES
from estimator.inference import estimate_daily_yield, InferenceUnavailable

SYSTEM_SIZES_KW = [0.5, 5.0, 1000.0]  # form min, typical, form max
TILTS_DEG = [0.0, 10.0, 45.0, 90.0]   # flat, common default, steep, vertical
RATE = 14.0  # arbitrary fixed rate -- only the kWh side is under test here

# Physically generous bounds for a sanity check, not a tight spec:
# even excellent tropical sites rarely clear ~7 kWh/kWp/day, and this
# should never go negative.
MAX_PLAUSIBLE_SPECIFIC_YIELD = 7.5


class Command(BaseCommand):
    help = "Stress-test estimate_daily_yield() across cities x months x tilts x system sizes."

    def handle(self, *args, **options):
        cities = [c for c, _ in PH_CITY_CHOICES]
        total_runs = 0
        anomalies = []

        for city in cities:
            for month in range(1, 13):
                for tilt in TILTS_DEG:
                    for size in SYSTEM_SIZES_KW:
                        total_runs += 1
                        try:
                            est = estimate_daily_yield(
                                system_size_kw=size, panel_tilt_deg=tilt,
                                city=city, month=month,
                                electricity_rate_php_per_kwh=RATE,
                            )
                        except InferenceUnavailable as exc:
                            anomalies.append(f"{city}/{month}/{tilt}°/{size}kW: "
                                              f"InferenceUnavailable: {exc}")
                            continue

                        specific_yield = est.predicted_daily_yield_kwh / size
                        problems = []
                        if est.predicted_daily_yield_kwh < 0:
                            problems.append("negative daily yield")
                        if not (0.0 <= est.predicted_avg_capacity_factor <= 1.05):
                            problems.append(f"capacity factor out of range "
                                             f"({est.predicted_avg_capacity_factor})")
                        if not (0.4 <= est.tilt_efficiency_factor <= 1.0001):
                            problems.append(f"tilt factor out of range "
                                             f"({est.tilt_efficiency_factor})")
                        if specific_yield > MAX_PLAUSIBLE_SPECIFIC_YIELD:
                            problems.append(f"implausible specific yield "
                                             f"({specific_yield:.2f} kWh/kWp/day)")
                        if est.estimated_monthly_php_savings < 0:
                            problems.append("negative PHP savings")

                        if problems:
                            anomalies.append(
                                f"{city}/month={month}/tilt={tilt}°/size={size}kW: "
                                f"{'; '.join(problems)} "
                                f"(daily_kwh={est.predicted_daily_yield_kwh}, "
                                f"cf={est.predicted_avg_capacity_factor}, "
                                f"tilt_factor={est.tilt_efficiency_factor})"
                            )

        self.stdout.write(f"Ran {total_runs} combinations "
                           f"({len(cities)} cities x 12 months x "
                           f"{len(TILTS_DEG)} tilts x {len(SYSTEM_SIZES_KW)} sizes).")
        if anomalies:
            self.stdout.write(self.style.ERROR(f"{len(anomalies)} anomalies found:"))
            for line in anomalies:
                self.stdout.write("  - " + line)
        else:
            self.stdout.write(self.style.SUCCESS("No anomalies found."))
