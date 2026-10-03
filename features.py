import pandas as pd

FEATURES = [
    "surface_pressure", "relative_humidity_2m", "wind_speed_10m",
    "temperature_2m", "precipitation", "wind_gusts_10m",
    "pressure_change_3h", "pressure_change_6h", "pressure_change_12h",
    "humidity_change_3h", "rain_last_6h", "gust_max_6h",
]


def add_features(df):
    """df: hourly rows for ONE city, sorted by time."""
    df = df.sort_values("ts").copy()
    df["pressure_change_3h"] = df["surface_pressure"].diff(3)
    df["pressure_change_6h"] = df["surface_pressure"].diff(6)
    df["pressure_change_12h"] = df["surface_pressure"].diff(12)
    df["humidity_change_3h"] = df["relative_humidity_2m"].diff(3)
    df["rain_last_6h"] = df["precipitation"].rolling(6).sum()
    df["gust_max_6h"] = df["wind_gusts_10m"].rolling(6).max()
    return df


def add_features_all(raw):
    return pd.concat([add_features(g) for _, g in raw.groupby("city")])
