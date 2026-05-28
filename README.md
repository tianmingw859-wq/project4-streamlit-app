# Project 4 Streamlit Frontend

This folder contains the Streamlit web app for the Auckland Green Travel Demand Predictor.

## How to run

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Files

- `streamlit_app.py` — main app file
- `data/daily_combined.csv` — cleaned combined dataset
- `models/*.joblib` — trained Random Forest models from the modelling notebook
- `models/feature_cols.joblib` — feature order used by the models

## Demo suggestion

During the presentation, show two scenarios:

1. Dry weekday, no public holiday, low rainfall.
2. Wet weekend or public holiday, higher rainfall and wind.

Compare the predicted bus, train, ferry and cycling demand.
