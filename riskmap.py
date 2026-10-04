import math

import plotly.graph_objects as go
import streamlit as st

from predict import CITIES, predict_live

COLORS = {"Low": "#34c77b", "Medium": "#ffb020", "High": "#ff5c5c"}
RADIUS_KM = 60


def hex_rgba(h, a):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def circle(lat, lon, radius_km, n=72):
    lats, lons = [], []
    for i in range(n + 1):
        ang = 2 * math.pi * i / n
        lats.append(lat + (radius_km / 111.0) * math.cos(ang))
        lons.append(lon + (radius_km / (111.0 * math.cos(math.radians(lat))))
                    * math.sin(ang))
    return lats, lons


def legend_html():
    items = [(COLORS[k], k) for k in COLORS] + [("#3a4660", "Not covered by the model")]
    chips = "".join(
        f"<div style='margin:6px 0;'><span style='display:inline-block;width:16px;"
        f"height:16px;background:{c};border-radius:4px;margin-right:10px;"
        f"vertical-align:middle;'></span>{name}</div>"
        for c, name in items
    )
    return (
        "<div style='background:#151f33;padding:14px 18px;border-radius:10px;"
        "margin-bottom:14px;'><div style='font-weight:700;margin-bottom:6px;'>"
        f"Risk level</div>{chips}</div>"
    )


def city_card(r):
    col = COLORS[r["level"]]
    v = r["values"]
    return (
        f"<div style='background:#151f33;border-left:6px solid {col};"
        f"padding:12px 16px;border-radius:8px;margin-bottom:12px;'>"
        f"<div style='font-weight:700;font-size:1.05rem;'>{r['city']}</div>"
        f"<div style='color:{col};font-weight:700;'>{r['level'].upper()} "
        f"· score {r['probability']:.0%}</div>"
        f"<div style='margin-top:6px;font-size:0.9rem;line-height:1.6;'>"
        f"Gusts: {v['wind_gusts_10m']:.0f} km/h<br>"
        f"Rain now: {v['precipitation']:.1f} mm/h<br>"
        f"Humidity: {v['relative_humidity_2m']:.0f}%<br>"
        f"Pressure vs normal: {v['pressure_anomaly']:+.1f} hPa</div></div>"
    )


def render_riskmap():
    st.subheader("Karnataka storm risk map")
    if st.button("Refresh map", type="primary", key="map_btn"):
        try:
            with st.spinner("Fetching current weather for both cities..."):
                st.session_state["map_results"] = [predict_live(c) for c in CITIES]
        except Exception as e:
            st.error("Couldn't fetch live weather. Check your internet "
                     f"connection and try again. ({type(e).__name__})")

    results = st.session_state.get("map_results")
    if not results:
        st.info("Click Refresh map to load the current risk for both cities.")
        return

    fig = go.Figure()
    for r in results:
        lat, lon = CITIES[r["city"]]
        clat, clon = circle(lat, lon, RADIUS_KM)
        col = COLORS[r["level"]]
        fig.add_trace(go.Scattermap(
            lat=clat, lon=clon, mode="lines", fill="toself",
            fillcolor=hex_rgba(col, 0.35), line=dict(color=col, width=2),
            hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scattermap(
        lat=[CITIES[r["city"]][0] for r in results],
        lon=[CITIES[r["city"]][1] for r in results],
        mode="markers+text",
        marker=dict(size=14, color=[COLORS[r["level"]] for r in results]),
        text=[f"{r['city']}: {r['level']}" for r in results],
        textposition="top center",
        textfont=dict(color="#e8eefc", size=14),
        hoverinfo="text", showlegend=False))
    fig.update_layout(
        map=dict(style="carto-darkmatter",
                 center=dict(lat=12.94, lon=76.2), zoom=6.3),
        height=520, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#0b1220")

    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown(legend_html(), unsafe_allow_html=True)
        for r in results:
            st.markdown(city_card(r), unsafe_allow_html=True)

    st.caption(f"Weather as of {results[0]['time']:%d %b %Y, %H:%M}. Each zone is a "
               f"{RADIUS_KM} km circle around the point where weather is measured "
               "and shows that city's current risk, not a forecast for every place "
               "inside it. Areas without colour are not covered by the model.")
