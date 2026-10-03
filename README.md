https://stormsense-hlrexhaym9vh27wvsmksrk.streamlit.app/

# StormSense

Explainable storm-risk predictor built for the MoE sevafirst hackathon (Theme 01).

StormSense estimates the risk of a storm (heavy rain or strong gusts) in the next
6 hours for Mangaluru and Bengaluru, and explains every prediction with SHAP.

**Educational prototype, not an official weather warning.**

## What it does
- **Live Storm Risk:** fetches current weather from Open-Meteo, shows a Low / Medium / High
  gauge, a SHAP chart, and a plain-language reason.
- **What-If Simulator:** move the sliders and watch the risk and explanation update.
- **Dashboard:** dataset overview, model performance, feature importance, recent weather
  trend, and a log of past checks.

## How it works
- Data: hourly weather for 2022-2025 from the Open-Meteo archive.
- Storm hour: rain of at least 4 mm/h or gusts of at least 50 km/h within the next 6 hours.
- Model: XGBoost on current conditions plus trends (pressure, humidity, rain, gusts),
  trained on 2022-2024 and tested on 2025. Recall 80%, ROC-AUC 0.90.
- Pressure features are constrained so that lower pressure never lowers the risk.
- Risk levels come from a calibration check: in 2025, a Low reading was followed by a storm
  about 2% of the time, Medium about 12%, High about 53%.

## Limitations
- Two cities and four years of data; not validated for other places.
- Scores are risk scores, not calibrated probabilities.
- Temperature mostly acts as a stand-in for time of day and season.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Retrain
    python3 data.py
    python3 train.py
