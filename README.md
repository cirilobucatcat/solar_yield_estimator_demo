# SolarYield Estimator

Django + XGBoost web app that estimates a Philippine solar installation's
daily energy yield (kWh) and rough PHP savings, given system size, panel
tilt, and city.

## Roadmap status

| Day | Task | Status |
|---|---|---|
| 1 | Django & PostgreSQL environment setup | ✅ done (this delivery) |
| 2 | Dataset preprocessing & feature engineering | ⏳ next |
| 3 | Model training (XGBoost) & serialization | ⏳ pending |
| 4 | Django models and forms | ✅ done early (needed a shape for Day 1 to be testable) |
| 5 | ML model integration in views.py | ⏳ stubbed, not wired up |
| 6 | UI templates & Chart.js dashboard | ✅ basic version done, will expand |
| 7 | Testing, validation, refactoring | ⏳ pending |

## Project layout

```
solaryield_estimator/
├── config/             # Django project settings, root urls
├── estimator/           # Main app: models, forms, views, urls, admin
├── templates/estimator/ # base.html + index.html (Bootstrap + Chart.js)
├── static/               # Static assets (empty for now)
├── ml_models/            # Trained .pkl artifacts go here (Day 3 output)
├── data/raw/             # Put the Kaggle CSV(s) here (gitignored)
├── requirements.txt
├── .env.example
└── manage.py
```

## Setup

1. **Create a virtual environment and install dependencies**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   For the very first run, before Postgres is set up, leave `USE_SQLITE=True`
   in `.env` — the app will use a local `db.sqlite3` file instead. Switch it
   to `False` and fill in `DB_NAME`/`DB_USER`/`DB_PASSWORD` once your
   PostgreSQL database exists.

3. **If using PostgreSQL**, create the database and user first, e.g.:
   ```sql
   CREATE DATABASE solaryield_db;
   CREATE USER solaryield_user WITH PASSWORD 'change-me';
   GRANT ALL PRIVILEGES ON DATABASE solaryield_db TO solaryield_user;
   ```

4. **Run migrations and start the server**
   ```bash
   python manage.py migrate
   python manage.py createsuperuser   # optional, for /admin/
   python manage.py runserver
   ```
   Visit `http://127.0.0.1:8000/` — you'll see the input form and, after
   submitting, a results panel. The predicted yield will show as "not
   trained yet" until Day 3's model is dropped into `ml_models/` and Day 5
   wires up inference in `estimator/views.py`.

## What's stubbed, on purpose

- `estimator/views.py` creates an `EstimationRecord` on every submit but
  leaves `predicted_daily_yield_kwh` as `None` — there's a commented block
  showing exactly where `joblib.load(...)` and `_model.predict(...)` will go
  once the model exists.
- `PH_CITY_CHOICES` in `estimator/forms.py` is a placeholder list of cities.
  Once Day 2's feature engineering is done, swap it for whatever
  location granularity the actual weather/irradiation data supports (e.g.
  it may end up being lat/long or a smaller set of plant sites rather than
  city names — depends on what the Kaggle dataset actually contains).
- `config/settings.py` already points `ML_MODEL_PATH` / `ML_SCALER_PATH` at
  `ml_models/xgb_solar_yield_model.pkl` and `feature_scaler.pkl`, so Day 3's
  serialization step just needs to save to those exact filenames.

## Next step (Day 2)

Feature engineering depends on what's actually in the Kaggle "Solar Power
Generation Data" — column names, whether it's per-plant sensor readings or
already-aggregated daily yield, units, etc. Bring the CSV (or a link/sample
of it) and we can build the preprocessing pipeline against the real schema
instead of guessing at it.
