"""
Day 2: Dataset preprocessing & feature engineering.

Source: Kaggle "Solar Power Generation Data" (2 plants, 15-minute interval
readings, 34 days: 2020-05-15 to 2020-06-17):
    - Plant_X_Generation_Data.csv     -> DATE_TIME, PLANT_ID, SOURCE_KEY,
                                          DC_POWER, AC_POWER, DAILY_YIELD, TOTAL_YIELD
    - Plant_X_Weather_Sensor_Data.csv -> DATE_TIME, PLANT_ID, SOURCE_KEY,
                                          AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION

Important reality-check vs. the app's original input fields:
    - There is NO "tilt angle" anywhere in this data (single fixed
      installation, zero variance) -> a model can't learn a tilt effect
      from it. Tilt will be applied as a physics-based multiplier at
      inference time (Day 5), not as a trained feature.
    - There is NO "city" column -- just 2 plant sites in India. "city"
      in the Django form will map to a small lookup table of typical
      irradiation/temperature (Day 5), not a raw model feature.
    - SOURCE_KEY = individual inverter. Each plant has 22 inverters.
      DAILY_YIELD is *cumulative since midnight* per inverter (resets
      daily), not a per-row delta -- so a day's total needs max(), not sum().

This script produces two tables:
    1. data/processed/interval_features.csv
       One row per (plant, 15-min timestamp): total plant AC/DC power
       (all 22 inverters summed) + matching weather reading + time
       features. This is the PRIMARY table for Day 3 -- ~6.4k rows,
       versus only ~68 if we aggregated straight to daily, and it
       captures the actual physical relationship between weather and
       power output.
    2. data/processed/daily_summary.csv
       One row per (plant, day): total daily yield in kWh (from the
       dataset's own DAILY_YIELD field) + daily weather aggregates +
       an estimated plant capacity (kW) used later to convert model
       output into a per-kW "specific yield" that generalizes to any
       user-input system size.
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLANTS = {
    4135001: {
        "label": "plant_1",
        "gen_file": "Plant_1_Generation_Data.csv",
        "weather_file": "Plant_1_Weather_Sensor_Data.csv",
        # Plant 1's generation file uses DD-MM-YYYY; every other file uses YYYY-MM-DD.
        "gen_date_format": "%d-%m-%Y %H:%M",
    },
    4136001: {
        "label": "plant_2",
        "gen_file": "Plant_2_Generation_Data.csv",
        "weather_file": "Plant_2_Weather_Sensor_Data.csv",
        "gen_date_format": "%Y-%m-%d %H:%M:%S",
    },
}


def load_plant(plant_id: int, cfg: dict):
    gen = pd.read_csv(RAW_DIR / cfg["gen_file"])
    gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"], format=cfg["gen_date_format"])

    weather = pd.read_csv(RAW_DIR / cfg["weather_file"])
    weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"], format="%Y-%m-%d %H:%M:%S")

    return gen, weather


def build_interval_table(gen: pd.DataFrame, weather: pd.DataFrame, plant_id: int) -> pd.DataFrame:
    """One row per 15-min timestamp: whole-plant power (22 inverters summed) + weather."""
    plant_power = (
        gen.groupby("DATE_TIME", as_index=False)
        .agg(total_dc_power_kw=("DC_POWER", "sum"), total_ac_power_kw=("AC_POWER", "sum"))
    )

    weather_cols = weather[
        ["DATE_TIME", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
    ]

    merged = plant_power.merge(weather_cols, on="DATE_TIME", how="inner")
    merged["plant_id"] = plant_id

    merged["hour"] = merged["DATE_TIME"].dt.hour
    merged["minute_of_day"] = merged["DATE_TIME"].dt.hour * 60 + merged["DATE_TIME"].dt.minute
    merged["month"] = merged["DATE_TIME"].dt.month
    merged["day_of_year"] = merged["DATE_TIME"].dt.dayofyear
    merged["is_daylight"] = (merged["IRRADIATION"] > 0).astype(int)

    return merged.rename(columns={
        "AMBIENT_TEMPERATURE": "ambient_temperature",
        "MODULE_TEMPERATURE": "module_temperature",
        "IRRADIATION": "irradiation",
    })


def build_daily_summary(gen: pd.DataFrame, weather: pd.DataFrame, plant_id: int,
                         capacity_kw_estimate: float) -> pd.DataFrame:
    """One row per day: total plant yield (kWh) + daily weather aggregates."""
    gen = gen.copy()
    gen["date"] = gen["DATE_TIME"].dt.date

    # DAILY_YIELD accumulates since midnight per inverter -> the day's total
    # for that inverter is its max reading that day. Sum across inverters
    # for the whole plant's daily yield.
    per_inverter_daily = (
        gen.groupby(["date", "SOURCE_KEY"])["DAILY_YIELD"].max().reset_index()
    )
    daily_yield = per_inverter_daily.groupby("date")["DAILY_YIELD"].sum().reset_index()
    daily_yield.columns = ["date", "daily_yield_kwh"]

    weather = weather.copy()
    weather["date"] = weather["DATE_TIME"].dt.date
    # IRRADIATION is instantaneous (kW/m^2) at 15-min steps -> sum * 0.25h = kWh/m^2/day
    daily_weather = weather.groupby("date").agg(
        avg_ambient_temperature=("AMBIENT_TEMPERATURE", "mean"),
        avg_module_temperature=("MODULE_TEMPERATURE", "mean"),
        daily_irradiation_kwh_m2=("IRRADIATION", lambda s: s.sum() * 0.25),
    ).reset_index()

    daily = daily_yield.merge(daily_weather, on="date", how="inner")
    daily["plant_id"] = plant_id
    daily["capacity_kw_estimate"] = capacity_kw_estimate
    daily["specific_yield_kwh_per_kwp"] = daily["daily_yield_kwh"] / capacity_kw_estimate
    daily["month"] = pd.to_datetime(daily["date"]).dt.month
    daily["day_of_year"] = pd.to_datetime(daily["date"]).dt.dayofyear

    return daily


def main():
    interval_frames, daily_frames = [], []

    for plant_id, cfg in PLANTS.items():
        gen, weather = load_plant(plant_id, cfg)

        interval = build_interval_table(gen, weather, plant_id)
        # Peak simultaneous plant output observed = proxy for installed AC capacity (kW).
        capacity_kw_estimate = interval["total_ac_power_kw"].max()

        daily = build_daily_summary(gen, weather, plant_id, capacity_kw_estimate)

        interval_frames.append(interval)
        daily_frames.append(daily)

        print(f"{cfg['label']}: capacity_kw_estimate={capacity_kw_estimate:,.1f} kW | "
              f"{len(interval)} interval rows | {len(daily)} daily rows")

    interval_df = pd.concat(interval_frames, ignore_index=True)
    daily_df = pd.concat(daily_frames, ignore_index=True)

    interval_path = OUT_DIR / "interval_features.csv"
    daily_path = OUT_DIR / "daily_summary.csv"
    interval_df.to_csv(interval_path, index=False)
    daily_df.to_csv(daily_path, index=False)

    print(f"\nWrote {len(interval_df)} rows -> {interval_path}")
    print(f"Wrote {len(daily_df)} rows -> {daily_path}")

    print("\nCorrelation with total_ac_power_kw (interval table):")
    print(interval_df[["irradiation", "ambient_temperature", "module_temperature",
                        "total_ac_power_kw"]].corr()["total_ac_power_kw"])


if __name__ == "__main__":
    main()
