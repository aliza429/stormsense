import pandas as pd

df = pd.read_csv("raw_data.csv", parse_dates=["ts"])


def make_labels(df, rain_mm, gust_kmh, horizon=6):
    out = []
    for city, g in df.groupby("city"):
        g = g.sort_values("ts").copy()
        storm_now = ((g["precipitation"] >= rain_mm) |
                     (g["wind_gusts_10m"] >= gust_kmh)).astype(int)
        # 1 if a storm hour occurs in the next `horizon` hours
        g["storm_next_6h"] = (storm_now[::-1]
                              .rolling(horizon, min_periods=horizon)
                              .max()[::-1].shift(-1))
        out.append(g)
    return pd.concat(out).dropna()


if __name__ == "__main__":
    for rain in (2, 4, 8):
        for gust in (40, 50, 60):
            rate = make_labels(df, rain, gust)["storm_next_6h"].mean()
            print(f"rain>={rain}mm/h or gust>={gust}km/h -> storm rate {rate:.3f}")
