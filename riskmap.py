import plotly.graph_objects as go
import streamlit as st

from predict import CITIES, predict_live

COLORS = {"Low": "#34c77b", "Medium": "#ffb020", "High": "#ff5c5c"}


def render_riskmap():
    st.subheader("Storm risk map")
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

    lats = [CITIES[r["city"]][0] for r in results]
    lons = [CITIES[r["city"]][1] for r in results]
    fig = go.Figure(go.Scattermap(
        lat=lats, lon=lons, mode="markers+text",
        marker=dict(size=30, color=[COLORS[r["level"]] for r in results]),
        text=[f"{r['city']}: {r['level']}" for r in results],
        textposition="top center",
        textfont=dict(color="#e8eefc", size=14),
        hoverinfo="text",
    ))
    fig.update_layout(
        map=dict(style="carto-darkmatter",
                 center=dict(lat=12.94, lon=76.2), zoom=6.3),
        height=480, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#0b1220",
    )
    st.plotly_chart(fig, width="stretch")

    cols = st.columns(len(results))
    for col, r in zip(cols, results):
        v = r["values"]
        with col:
            st.markdown(f"**{r['city']}**")
            st.markdown(f"Risk: **{r['level']}** ({r['probability']:.0%})")
            st.markdown(f"Gusts: {v['wind_gusts_10m']:.0f} km/h")
            st.markdown(f"Rain now: {v['precipitation']:.1f} mm/h")
            st.markdown(f"Humidity: {v['relative_humidity_2m']:.0f}%")
            st.markdown(f"Pressure vs normal: {v['pressure_anomaly']:+.1f} hPa")
    st.caption("Only the two cities the model was trained on are shown. The map "
               "does not estimate risk anywhere in between.")
