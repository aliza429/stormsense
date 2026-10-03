import pandas as pd
import streamlit as st

from predict import predict_features, LEVEL_HIT_RATE


@st.cache_data
def typical_ratios():
    """Data-driven links: how gusts relate to wind, and 6h rain to current rain."""
    d = pd.read_csv("storm_data.csv", usecols=[
        "wind_speed_10m", "wind_gusts_10m", "precipitation", "rain_last_6h"])
    w = d[d["wind_speed_10m"] > 3]
    gust_ratio = float((w["wind_gusts_10m"] / w["wind_speed_10m"]).median())
    r = d[d["precipitation"] >= 1]
    rain_ratio = float((r["rain_last_6h"] / r["precipitation"]).median())
    return gust_ratio, rain_ratio


def render_whatif(gauge, shap_chart):
    gust_ratio, rain_ratio = typical_ratios()
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Adjust conditions")
        pchg = st.slider("Pressure change over 6 hours (hPa)",
                         -15.0, 5.0, 0.0, 0.5, key="wi_p")
        hum = st.slider("Humidity (%)", 20, 100, 70, key="wi_h")
        wind = st.slider("Wind speed (km/h)", 0, 30, 10, key="wi_w")
        temp = st.slider("Temperature (°C)", 15, 40, 29, key="wi_t")
        rain = st.slider("Rainfall (mm/h)", 0.0, 30.0, 0.0, 0.5, key="wi_r")
        with st.expander("More controls (optional)"):
            hchg = st.slider("Humidity change over 3 hours (%)",
                             -20, 30, 0, key="wi_hc")

    gust = wind * gust_ratio
    feats = {
        "pressure_anomaly": pchg,
        "relative_humidity_2m": float(hum),
        "wind_speed_10m": float(wind),
        "temperature_2m": float(temp),
        "precipitation": float(rain),
        "wind_gusts_10m": gust,
        "pressure_change_3h": pchg / 2,
        "pressure_change_6h": pchg,
        "pressure_change_12h": pchg * 1.5,
        "humidity_change_3h": float(hchg),
        "rain_last_6h": rain * rain_ratio,
        "gust_max_6h": gust,
    }
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
        st.caption("Simulated inputs. Gusts follow wind speed using the typical "
                   "gust-to-wind ratio in our data, and rain over the last 6 hours "
                   "follows the rainfall slider. A falling pressure is applied as "
                   "both a drop in the trend and a drop below normal.")
