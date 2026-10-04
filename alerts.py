import streamlit as st

STYLE = {
    "Low": ("#34c77b", "#14352a", "LOW STORM RISK"),
    "Medium": ("#ffb020", "#3a2f10", "MEDIUM STORM RISK"),
    "High": ("#ff5c5c", "#3d1a1a", "HIGH STORM RISK"),
}

ADVICE = {
    "Low": [
        "A storm is unlikely in the next 6 hours.",
        "Check again if the sky or wind changes.",
    ],
    "Medium": [
        "Stay alert, because conditions can change quickly.",
        "Bring in light outdoor items.",
        "Check the latest update before travelling.",
    ],
    "High": [
        "Avoid coastal and open areas.",
        "Secure loose objects outdoors.",
        "Avoid unnecessary travel.",
        "Keep your phone charged and follow IMD and local authority updates.",
        "Emergency number in India: 112.",
    ],
}


def hazards(v):
    found = []
    gust = max(v["wind_gusts_10m"], v["gust_max_6h"])
    if gust >= 50:
        found.append(f"Strong gusts ({gust:.0f} km/h)")
    if v["precipitation"] >= 4:
        found.append(f"Heavy rainfall ({v['precipitation']:.1f} mm/h)")
    if v["pressure_change_6h"] <= -3.8:
        found.append(f"Rapid pressure drop ({v['pressure_change_6h']:+.1f} hPa in 6 hours)")
    return found


def render_alert(res):
    fg, bg, title = STYLE[res["level"]]
    banner = (
        f"<div style='border-left:6px solid {fg};background:{bg};"
        f"padding:16px 20px;border-radius:8px;margin:8px 0 16px 0;'>"
        f"<div style='font-size:1.4rem;font-weight:700;color:{fg};'>{title}</div>"
        f"<div style='margin-top:4px;'>{res['city']} · risk score "
        f"{res['probability']:.0%} · within the next 6 hours</div></div>"
    )
    st.markdown(banner, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Conditions detected**")
        found = hazards(res["values"])
        if found:
            for h in found:
                st.markdown(f"- ⚠ {h}")
        else:
            st.markdown("- No single hazard threshold exceeded")
    with c2:
        st.markdown("**Recommended**")
        for a in ADVICE[res["level"]]:
            st.markdown(f"- {a}")

    st.info(res["reason"])
    st.caption("General safety guidance, not an official warning. Follow IMD and "
               "local authorities. Hazard thresholds: gusts of 50 km/h or more, "
               "rain of 4 mm/h or more, or a pressure drop of 3.8 hPa or more in "
               "6 hours, which is in the lowest 5% of changes in our data.")
