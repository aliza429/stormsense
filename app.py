import os
from datetime import datetime
from dashboard import render_dashboard
from names import FRIENDLY

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from predict import CITIES, predict_live, LEVEL_HIT_RATE
st.set_page_config(page_title="StormSense", page_icon="⛈️", layout="wide")
st.title("⛈️ StormSense")
st.caption("Predicting storms, explaining why. Educational prototype, "
           "not an official weather warning.")

LOG = "log.csv"
COLORS = {"Low": "#2e9e5b", "Medium": "#f0a500", "High": "#d64545"}


def gauge(prob, level):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100),
        number={"suffix": "%"},
        title={"text": f"{level.upper()} risk"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": COLORS[level]},
            "steps": [
                  {"range": [30, 70], "color": "#fdf1d6"},
                  {"range": [70, 100], "color": "#f9dede"},
            ],
        },
    ))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def shap_chart(contrib, top=8):
    s = contrib.reindex(contrib.abs().sort_values(ascending=False).head(top).index)
    s = s.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([FRIENDLY[i] for i in s.index], s.values,
            color=["#d64545" if v > 0 else "#3b7dd8" for v in s.values])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Pushes risk down  <  0  >  pushes risk up")
    fig.tight_layout()
    return fig


def log_check(res):
    row = pd.DataFrame([{
        "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "city": res["city"],
        "risk": res["level"],
        "probability": round(res["probability"], 3),
    }])
    row.to_csv(LOG, mode="a", header=not os.path.exists(LOG), index=False)


tab1, tab2, tab3 = st.tabs(["Live Storm Risk", "What-If Simulator", "Dashboard"])

with tab1:
    city = st.selectbox("Location", list(CITIES), key="city")
    if st.button("Check storm risk", type="primary"):
        try:
            with st.spinner("Fetching current weather..."):
                st.session_state["result"] = predict_live(city)
            log_check(st.session_state["result"])
        except Exception as e:
            st.error("Couldn't fetch live weather. Check your internet "
                     f"connection and try again. ({type(e).__name__})")

    res = st.session_state.get("result")
    if res:
        st.caption(f"{res['city']} · weather as of {res['time']:%d %b %Y, %H:%M}")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(gauge(res["probability"], res["level"]),
                            width="stretch")
        with c2:
            st.subheader("Why this prediction?")
            st.pyplot(shap_chart(res["contrib"]))
        st.info(res["reason"])
        st.caption("The percentage is the model's storm risk score for the next "
                   "6 hours (heavy rain or strong gusts). It is not a calibrated "
                   "probability.")
        st.caption(f"In our 2025 test data, a {res['level']} reading was "
                   f"followed by a storm within 6 hours about "
                   f"{LEVEL_HIT_RATE[res['level']]:.0%} of the time.")

with tab2:
    st.write("Coming next.")

with tab3:
    render_dashboard()
