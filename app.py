from alerts import render_alert
from riskmap import render_riskmap
from whatif import render_whatif
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
COLORS = {"Low": "#34c77b", "Medium": "#ffb020", "High": "#ff5c5c"}
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
            {"range": [0, 30], "color": "#14352a"},
            {"range": [30, 70], "color": "#3a2f10"},
            {"range": [70, 100], "color": "#3d1a1a"},
            ],
        },
    ))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def shap_chart(contrib, top=8):
    s = contrib.reindex(contrib.abs().sort_values(ascending=False).head(top).index)
    s = s.iloc[::-1]
    bg, fg = "#0b1220", "#e8eefc"
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.barh([FRIENDLY[i] for i in s.index], s.values,
            color=["#ff6b6b" if v > 0 else "#5aa9ff" for v in s.values])
    ax.axvline(0, color="#8a96b0", lw=0.8)
    ax.tick_params(colors=fg)
    ax.set_xlabel("Pushes risk down  <  0  >  pushes risk up", color=fg)
    for sp in ax.spines.values():
        sp.set_color("#3a4660")
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


tab1, tab_map, tab2, tab3 = st.tabs(["Live Storm Risk", "Risk Map", "What-If Simulator", "Dashboard"])
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
        render_alert(res)
        st.caption("The percentage is the model's storm risk score for the next "
                   "6 hours (heavy rain or strong gusts). It is not a calibrated "
                   "probability.")
        st.caption(f"In our 2025 test data, a {res['level']} reading was "
                   f"followed by a storm within 6 hours about "
                   f"{LEVEL_HIT_RATE[res['level']]:.0%} of the time.")

with tab2:
    render_whatif(gauge, shap_chart)
with tab3:
    render_dashboard()
with tab_map:
    render_riskmap()
