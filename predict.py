import joblib
import numpy as np
import pandas as pd
import requests
import shap

from features import FEATURES, add_features

CITIES = {
    "Mangaluru": (12.91, 74.86),
    "Bengaluru": (12.97, 77.59),
}
VARS = ("surface_pressure,relative_humidity_2m,wind_speed_10m,"
        "temperature_2m,precipitation,wind_gusts_10m")

LOW_MAX, MED_MAX = 0.30, 0.60   # probability cut-offs for Low / Medium / High

_bundle = joblib.load("model.pkl")
MODEL = _bundle["model"]
EXPLAINER = shap.TreeExplainer(MODEL)

PHRASES = {
    "wind_gusts_10m":      lambda v: f"gusts are {v:.0f} km/h",
    "gust_max_6h":         lambda v: f"the strongest gust in 6 hours was {v:.0f} km/h",
    "wind_speed_10m":      lambda v: f"wind speed is {v:.0f} km/h",
    "precipitation":       lambda v: f"rainfall is {v:.1f} mm/h",
    "rain_last_6h":        lambda v: f"{v:.1f} mm of rain fell in the last 6 hours",
    "relative_humidity_2m": lambda v: f"humidity is {v:.0f}%",
    "humidity_change_3h":  lambda v: f"humidity changed {v:+.0f}% in 3 hours",
    "temperature_2m":      lambda v: f"temperature is {v:.0f}°C",
    "surface_pressure":    lambda v: f"pressure is {v:.0f} hPa",
    "pressure_change_3h":  lambda v: f"pressure changed {v:+.1f} hPa in 3 hours",
    "pressure_change_6h":  lambda v: f"pressure changed {v:+.1f} hPa in 6 hours",
    "pressure_change_12h": lambda v: f"pressure changed {v:+.1f} hPa in 12 hours",
}


def fetch_live(city):
    """Last 24h of hourly weather plus the current hour, with features added."""
    lat, lon = CITIES[city]
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat, "longitude": lon, "hourly": VARS,
            "past_hours": 24, "forecast_hours": 1,
            "timezone": "Asia/Kolkata",
        },
        timeout=20,
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"]).rename(columns={"time": "ts"})
    df["ts"] = pd.to_datetime(df["ts"])
    df["city"] = city
    df = add_features(df).dropna(subset=FEATURES)
    return df  # the last row is the current hour


def risk_level(p):
    return "Low" if p < LOW_MAX else "Medium" if p < MED_MAX else "High"


def predict_features(feats):
    """feats: dict with all 12 FEATURES. Used by Tab 1 and the What-If tab."""
    X = pd.DataFrame([feats])[FEATURES]
    p = float(MODEL.predict_proba(X)[0, 1])
    sv = EXPLAINER.shap_values(X)[0]
    contrib = pd.Series(sv, index=FEATURES)
    return {
        "probability": p,
        "level": risk_level(p),
        "contrib": contrib.sort_values(),      # for the SHAP bar chart
        "values": X.iloc[0].to_dict(),
        "reason": make_reason(p, contrib, X.iloc[0]),
    }


def make_reason(p, contrib, row):
    level = risk_level(p)
    if level == "Low":
        calm = contrib.drop(labels=[f for f in contrib.index if "pressure" in f])
        down = calm.sort_values().head(2).index
        parts = [PHRASES[f](row[f]) for f in down]
        return f"Risk is LOW because " + " and ".join(parts) + "."
    up = contrib.sort_values(ascending=False)
    up = up[up > 0].head(3).index
    parts = [PHRASES[f](row[f]) for f in up]
    if len(parts) > 1:
        text = ", ".join(parts[:-1]) + " and " + parts[-1]
    else:
        text = parts[0] if parts else "several factors add up"
    return f"Risk is {level.upper()} because {text}."

def predict_live(city):
    df = fetch_live(city)
    row = df.iloc[-1]
    out = predict_features({f: float(row[f]) for f in FEATURES})
    out["city"] = city
    out["time"] = row["ts"]
    return out


if __name__ == "__main__":
    for c in CITIES:
        res = predict_live(c)
        print(f"{c}  {res['time']}  {res['level']}  p={res['probability']:.2f}")
        print("  ", res["reason"])
        print("  ", {k: round(v, 1) for k, v in res["values"].items()})
