import requests
import pandas as pd

CITIES = {
    "Mangaluru": (12.91, 74.86),
    "Bengaluru": (12.97, 77.59),
}

VARS = ("surface_pressure,relative_humidity_2m,wind_speed_10m,"
        "temperature_2m,precipitation,wind_gusts_10m")


def fetch_history(city, start="2022-01-01", end="2025-12-31"):
    lat, lon = CITIES[city]
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "hourly": VARS,
            "timezone": "Asia/Kolkata",
        },
        timeout=60,
    )
    r.raise_for_status()

    df = pd.DataFrame(r.json()["hourly"]).rename(columns={"time": "ts"})
    df["ts"] = pd.to_datetime(df["ts"])
    df["city"] = city

    # feature: how much pressure changed in the last 6 hours
    df["pressure_change_6h"] = df["surface_pressure"].diff(6)

    return df.dropna(subset=["pressure_change_6h"])


if __name__ == "__main__":
    frames = []
    for c in CITIES:
        print("Fetching", c)
        frames.append(fetch_history(c))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv("raw_data.csv", index=False)
    print("Saved", out.shape)
