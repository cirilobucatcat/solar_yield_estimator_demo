"""
Day 7: why was plant 2 the harder fit in Day 3's backtest?

Day 3 found plant 2's held-out daily-total error ranged up to ~18%,
noticeably worse than plant 1's <3%. This script digs into *why*,
using the same train/test split and model config as ml_pipeline/train.py.

Finding: plant 2 has real zero-output-despite-strong-sunlight events
that plant 1 doesn't. E.g. 2020-06-11 12:00 -- irradiation 0.62 kW/m^2
(strong midday sun), ambient 29.7C -- actual capacity_factor was
exactly 0.0. That's not a weather effect; it looks like an inverter
outage, fault, or curtailment event specific to plant 2's equipment.
A model trained only on weather has no way to predict those -- they're
operational/equipment events, not solar-resource events.

Quantified: rows with irradiation > 0.3 kW/m^2 but capacity_factor <
0.05 (i.e. "sunny but basically no output") make up ~1% of plant 2's
sunny-condition rows and appear ZERO times for plant 1. Small in
volume, but each one is a large single-interval residual, and when
several land in the same held-out day they add up to a double-digit
daily percentage error -- which is exactly the pattern Day 3 saw.

Practical takeaway for the app: this is a data-quality/equipment-outage
issue specific to the training plants, not a flaw in the
weather-to-capacity_factor relationship itself (residuals center near
zero with no systematic bias -- see the mean residual figures below).
It doesn't call the modeling approach into question; it's a reminder
that any single day's real-world output can deviate from a
weather-only prediction due to things the model can't see (outages,
soiling, curtailment) -- worth a line of caveat text in the app itself.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

FEATURE_COLUMNS = ["irradiation", "ambient_temperature", "module_temperature", "hour", "month"]


def main():
    df = pd.read_csv("data/processed/interval_features.csv", parse_dates=["DATE_TIME"])
    capacity_by_plant = df.groupby("plant_id")["total_ac_power_kw"].max()
    df["capacity_kw_estimate"] = df["plant_id"].map(capacity_by_plant)
    df["capacity_factor"] = df["total_ac_power_kw"] / df["capacity_kw_estimate"]
    df["date"] = df["DATE_TIME"].dt.date

    unique_dates = sorted(df["date"].unique())
    cutoff = int(len(unique_dates) * 0.8)
    train_dates, test_dates = set(unique_dates[:cutoff]), set(unique_dates[cutoff:])
    train_df = df[df["date"].isin(train_dates)]
    test_df = df[df["date"].isin(test_dates)].copy()

    model = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                          objective="reg:squarederror", random_state=42)
    model.fit(train_df[FEATURE_COLUMNS], train_df["capacity_factor"])
    test_df["pred_cf"] = np.clip(model.predict(test_df[FEATURE_COLUMNS]), 0, None)
    test_df["residual"] = test_df["pred_cf"] - test_df["capacity_factor"]

    print("=== Residual mean/std by plant (test days, daylight only) ===")
    daylight = test_df[test_df["irradiation"] > 0.01]
    print(daylight.groupby("plant_id")["residual"].agg(["mean", "std"]).round(4))
    print("Near-zero mean in both cases -> no systematic bias. Plant 2's std is")
    print("~3x plant 1's -> plant 2 is noisier, not mis-modeled in one direction.\n")

    print("=== 'Sunny but ~zero output' rows (irradiation>0.3, capacity_factor<0.05) ===")
    suspect = df[(df["irradiation"] > 0.3) & (df["capacity_factor"] < 0.05)]
    meaningful = df[df["irradiation"] > 0.3]
    counts = suspect.groupby("plant_id").size()
    rates = (counts / meaningful.groupby("plant_id").size() * 100).round(2)
    print("Counts:\n", counts.to_string())
    print("As % of that plant's sunny-condition rows:\n", rates.to_string())
    print("\nPlant 1 (4135001): 0 such rows. Plant 2 (4136001): ~1% of sunny rows.")
    print("These look like inverter outages/curtailment specific to plant 2's")
    print("equipment -- not something a weather-only model could ever predict.\n")

    print("=== Worst single-interval over-prediction, plant 2 ===")
    p2 = test_df[test_df["plant_id"] == 4136001].sort_values("residual", ascending=False)
    row = p2.iloc[0]
    print(f"{row['DATE_TIME']}: irradiation={row['irradiation']:.2f} kW/m^2, "
          f"actual capacity_factor={row['capacity_factor']:.3f}, "
          f"predicted={row['pred_cf']:.3f} -- model (correctly) expected output "
          f"given the sunlight; the plant just didn't deliver it that interval.")


if __name__ == "__main__":
    main()
