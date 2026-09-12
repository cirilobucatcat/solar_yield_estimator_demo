"""
Day 3: Model training (XGBoost) & serialization.

Trains on data/processed/interval_features.csv (from Day 2).

Key design choice: predict CAPACITY FACTOR, not raw kW.
---------------------------------------------------------
The two training plants are ~29 MW and ~26 MW -- nothing like a
residential/commercial PH installation. If we trained on raw
total_ac_power_kw, the model would learn power levels tied to those
two specific plants, which is useless for a user typing in "5 kW".

Instead, each row's power is normalized by its own plant's estimated
capacity:
    capacity_factor = total_ac_power_kw / capacity_kw_estimate   (~0 to ~1)

The model learns "what fraction of installed capacity does a system
deliver given this weather", which is (approximately) scale-invariant
physics and transfers to any system size. At inference (Day 5):

    predicted_power_kw(t)  = capacity_factor_pred(t) * user_system_size_kw
    predicted_daily_kwh    = sum over a day's intervals of predicted_power_kw(t) * 0.25h

This also means plant_id is deliberately EXCLUDED from the feature
list -- it's a site identity the app could never supply for a new PH
city, and normalizing already removes the need for it.

Validation strategy: chronological split (not random). The last ~20%
of days become the test set, so we're checking generalization to
unseen days rather than unseen 15-minute slices of days the model
already partly saw.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "ml_models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLUMNS = [
    "irradiation",
    "ambient_temperature",
    "module_temperature",
    "hour",
    "month",
]
TARGET_COLUMN = "capacity_factor"
TEST_FRACTION_OF_DAYS = 0.2


def load_and_prepare() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "interval_features.csv", parse_dates=["DATE_TIME"])

    capacity_by_plant = df.groupby("plant_id")["total_ac_power_kw"].max()
    df["capacity_kw_estimate"] = df["plant_id"].map(capacity_by_plant)
    df["capacity_factor"] = df["total_ac_power_kw"] / df["capacity_kw_estimate"]
    df["date"] = df["DATE_TIME"].dt.date

    return df


def chronological_split(df: pd.DataFrame):
    unique_dates = sorted(df["date"].unique())
    cutoff_idx = int(len(unique_dates) * (1 - TEST_FRACTION_OF_DAYS))
    train_dates = set(unique_dates[:cutoff_idx])
    test_dates = set(unique_dates[cutoff_idx:])

    train_df = df[df["date"].isin(train_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()
    print(f"Train: {train_df['date'].min()} -> {train_df['date'].max()} "
          f"({len(train_dates)} days, {len(train_df)} rows)")
    print(f"Test:  {test_df['date'].min()} -> {test_df['date'].max()} "
          f"({len(test_dates)} days, {len(test_df)} rows)")
    return train_df, test_df


def train_model(train_df: pd.DataFrame) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])
    return model


def evaluate_interval_level(model: XGBRegressor, test_df: pd.DataFrame):
    preds = model.predict(test_df[FEATURE_COLUMNS])
    preds = np.clip(preds, 0, None)  # capacity factor can't be negative

    mae = mean_absolute_error(test_df[TARGET_COLUMN], preds)
    rmse = mean_squared_error(test_df[TARGET_COLUMN], preds) ** 0.5
    r2 = r2_score(test_df[TARGET_COLUMN], preds)
    print(f"\n[Interval-level, capacity_factor] MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")
    return preds


def evaluate_daily_backtest(test_df: pd.DataFrame, preds: np.ndarray):
    """
    The number that actually matters for the app: integrate predicted
    capacity_factor back into predicted daily kWh per plant, and compare
    against the real daily_summary.csv totals from Day 2.
    """
    test_df = test_df.copy()
    test_df["predicted_capacity_factor"] = preds
    test_df["predicted_power_kw"] = test_df["predicted_capacity_factor"] * test_df["capacity_kw_estimate"]

    predicted_daily = (
        test_df.groupby(["plant_id", "date"])["predicted_power_kw"]
        .sum().mul(0.25)  # kW at 15-min steps -> kWh
        .reset_index(name="predicted_daily_yield_kwh")
    )

    actual_daily = pd.read_csv(PROCESSED_DIR / "daily_summary.csv")
    actual_daily["date"] = pd.to_datetime(actual_daily["date"]).dt.date

    merged = predicted_daily.merge(actual_daily[["plant_id", "date", "daily_yield_kwh"]],
                                    on=["plant_id", "date"], how="inner")
    merged["abs_pct_error"] = (
        (merged["predicted_daily_yield_kwh"] - merged["daily_yield_kwh"]).abs()
        / merged["daily_yield_kwh"]
    )

    print("\n[Daily backtest: predicted vs. actual total kWh on held-out days]")
    print(merged[["plant_id", "date", "predicted_daily_yield_kwh", "daily_yield_kwh", "abs_pct_error"]]
          .round({"predicted_daily_yield_kwh": 0, "daily_yield_kwh": 0, "abs_pct_error": 3})
          .to_string(index=False))
    print(f"\nMean absolute % error on daily totals (held-out days): "
          f"{merged['abs_pct_error'].mean() * 100:.1f}%")


def main():
    df = load_and_prepare()
    train_df, test_df = chronological_split(df)

    model = train_model(train_df)
    preds = evaluate_interval_level(model, test_df)
    evaluate_daily_backtest(test_df, preds)

    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_.round(4).tolist()))
    print(f"\nFeature importances: {importances}")

    # --- Serialize for Day 5 ---
    model_path = MODEL_DIR / "xgb_solar_yield_model.pkl"
    columns_path = MODEL_DIR / "feature_columns.json"
    joblib.dump(model, model_path)
    columns_path.write_text(json.dumps({
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "note": "Model predicts capacity_factor (0-1). "
                "predicted_power_kw = capacity_factor * user_system_size_kw. "
                "No scaler needed -- tree-based model, and no feature scaling was applied.",
    }, indent=2))

    print(f"\nSaved model -> {model_path}")
    print(f"Saved feature metadata -> {columns_path}")


if __name__ == "__main__":
    main()
