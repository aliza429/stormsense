import numpy as np
import pandas as pd
import streamlit as st

from features import FEATURES
from predict import predict_features, LEVEL_HIT_RATE


@st.cache_data
def data_links():
    """Relationships and ranges taken from the real training data."""
    d = pd.read_csv("storm_data.csv", usecols=FEATURES)
    w = d[d["wind_speed_10m"] > 3]
    gust_ratio = float((w["wind_gusts_10m"] / w["wind_speed_10m"]).median())
    r = d[d["precipitation"] >= 1]
    rain_ratio = float((r["rain_last_6h"] / r["precipitation"]).median())
    var6 = d["pressure_change_6h"].var()
    slopes = {c: float(d[c].cov(d["pressure_change_6h"]) / var6)
              for c in ["pressure_change_3h", "pressure_change_12h",
                        "pressure_anomaly"]}
    bounds = {c: (float(d[c].min()), float(d[c].max())) for c in FEATURES}
    return gust_ratio, rain_ratio, slopes, bounds


def render_whatif(gauge, shap_chart):
    gust_ratio, rain_ratio, slopes, bounds = data_links()
    max_rain = float(min(30, np.floor(bounds["precipitation"][1])))
    max_wind = int(min(30, np.floor(bounds["wind_speed_10m"][1])))

    left, right = st.columns([1, 2])

    with left:
        st.subheader("Adjust conditions")
        pchg = st.slider("Pressure change over 6 hours (hPa)",
                         -7.0, 5.0, 0.0, 0.5, key="wi_p")
        hum = st.slider("Humidity (%)", 20, 100, 70, key="wi_h")
        wind = st.slider("Wind speed (km/h)", 0, max_wind, min(10, max_wind),
                         key="wi_w")
        temp = st.slider("Temperature (°C)", 15, 40, 29, key="wi_t")
        rain = st.slider("Rainfall (mm/h)", 0.0, max_rain, 0.0, 0.5, key="wi_r")
        hchg = 0
    gust = wind * gust_ratio
    feats = {
        "pressure_anomaly": pchg * slopes["pressure_anomaly"],
        "relative_humidity_2m": float(hum),
        "wind_speed_10m": float(wind),
        "temperature_2m": float(temp),
        "precipitation": float(rain),
        "wind_gusts_10m": gust,
        "pressure_change_3h": pchg * slopes["pressure_change_3h"],
        "pressure_change_6h": pchg,
        "pressure_change_12h": pchg * slopes["pressure_change_12h"],
        "humidity_change_3h": float(hchg),
        "rain_last_6h": rain * rain_ratio,
        "gust_max_6h": gust,
    }
    # never feed the model values outside the range it was trained on
    feats = {k: float(np.clip(v, *bounds[k])) for k, v in feats.items()}
    res = predict_features(feats)

    with right:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(gauge(res["probability"], res["level"]),
                            width="stretch", key="wi_gauge")
        with c2:
            st.subheader("Why this prediction?")
            st.pyplot(shap_chart(res["contrib"]))
        st.info(res["reason"])
        st.caption(f"In our 2025 test data, a {res['level']} reading was followed "
                   f"by a storm within 6 hours about "
                   f"{LEVEL_HIT_RATE[res['level']]:.0%} of the time.")
        st.caption("Simulated inputs, limited to the ranges seen in our 2022-2025 "
                   "data. Gusts follow wind speed, and the other pressure "
                   "measures follow the 6-hour change, using the typical "
                   "relationships in the data.")
