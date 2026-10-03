import json
import os

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from names import FRIENDLY
from predict import CITIES, fetch_live


@st.cache_data
def load_labels():
    return pd.read_csv("storm_data.csv", usecols=["city", "storm_next_6h"])


@st.cache_data(ttl=600)
def load_recent(city):
    return fetch_live(city)


def trend_chart(df, col, title, fixed_range=None, zero=False):
    fig = go.Figure(go.Scatter(x=df.index, y=df[col], mode="lines",
                               line=dict(color="#3b7dd8")))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10),
                      title=title)
    if fixed_range:
        fig.update_yaxes(range=fixed_range)
    if zero:
        fig.update_yaxes(rangemode="tozero")
    return fig


def render_dashboard():
    metrics = json.load(open("metrics.json"))
    bundle = joblib.load("model.pkl")

    # 1) Dataset overview
    st.subheader("Dataset overview")
    labels = load_labels()
    storm = int(labels["storm_next_6h"].sum())
    total = len(labels)
    c1, c2, c3 = st.columns(3)
    c1.metric("Hourly records", f"{total:,}")
    c2.metric("Storm hours", f"{storm:,}")
    c3.metric("Non-storm hours", f"{total - storm:,}")
    st.caption(f"A storm hour = rain of at least {metrics['rain_mm']} mm/h or gusts "
               f"of at least {metrics['gust_kmh']} km/h within the next 6 hours. "
               "Mangaluru and Bengaluru, 2022-2025.")

    labels = labels.assign(storm=labels["storm_next_6h"].astype(int))
    counts = labels.groupby("city")["storm"].agg(["sum", "count"])
    fig0 = go.Figure()
    fig0.add_bar(name="Non-storm hours", x=counts.index,
                 y=counts["count"] - counts["sum"], marker_color="#9db8e0")
    fig0.add_bar(name="Storm hours", x=counts.index,
                 y=counts["sum"], marker_color="#d64545")
    fig0.update_layout(barmode="stack", height=320, yaxis_title="Hours",
                       margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig0, width="stretch")

    # 2) Model performance
    st.subheader("Model performance (tested on unseen 2025 data)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recall (storms caught)", f"{metrics['recall']:.0%}")
    m2.metric("ROC-AUC", f"{metrics['roc_auc']:.2f}")
    m3.metric("Precision", f"{metrics['precision']:.0%}")
    m4.metric("Accuracy", f"{metrics['accuracy']:.0%}")
    st.caption("Storms are rare, so accuracy alone is misleading. Recall shows how "
               "many real storms the model catches, and it tends to warn rather "
               "than miss, so some alarms are false.")

    cm = metrics["confusion_matrix"]
    fig = go.Figure(go.Heatmap(
        z=cm, x=["Predicted: no storm", "Predicted: storm"],
        y=["Actual: no storm", "Actual: storm"],
        text=[[f"{v:,}" for v in row] for row in cm],
        texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_autorange="reversed", title="Confusion matrix")
    st.plotly_chart(fig, width="stretch")

    # 3) Feature importance
    st.subheader("What the model pays attention to")
    imp = pd.Series(bundle["model"].feature_importances_,
                    index=[FRIENDLY[f] for f in bundle["features"]]).sort_values()
    fig2 = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                            marker_color="#3b7dd8"))
    fig2.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                       xaxis_title="Importance")
    st.plotly_chart(fig2, width="stretch")

    # 4) Recent weather trend (loads only when switched on)
    st.subheader("Recent weather trend (last 24 hours)")
    city = st.selectbox("City", list(CITIES), key="trend_city")
    if st.toggle("Load recent weather trend", key="trend_on"):
        try:
            df = load_recent(city).set_index("ts")
            a, b = st.columns(2)
            a.plotly_chart(trend_chart(df, "pressure_anomaly",
                                       "Pressure vs normal (hPa)"),
                           width="stretch")
            b.plotly_chart(trend_chart(df, "relative_humidity_2m", "Humidity (%)",
                                       fixed_range=[0, 100]), width="stretch")
            c, d = st.columns(2)
            c.plotly_chart(trend_chart(df, "wind_gusts_10m", "Wind gusts (km/h)",
                                       zero=True), width="stretch")
            d.plotly_chart(trend_chart(df, "precipitation", "Rainfall (mm/h)",
                                       zero=True), width="stretch")
        except Exception as e:
            st.warning(f"Couldn't load live weather right now ({type(e).__name__}).")
    else:
        st.caption("Switch on to load the last 24 hours of live weather.")

    # 5) Prediction log
    st.subheader("Prediction log")
    if os.path.exists("log.csv"):
        log = pd.read_csv("log.csv").iloc[::-1]
        st.dataframe(log, width="stretch", hide_index=True)
    elif os.path.exists("sample_log.csv"):
        log = pd.read_csv("sample_log.csv").iloc[::-1]
        st.dataframe(log, width="stretch", hide_index=True)
        st.caption("Sample entries from earlier checks. New checks appear here "
                   "as you run them.")
    else:
        st.info("No checks yet. Run one on the Live Storm Risk tab.")
