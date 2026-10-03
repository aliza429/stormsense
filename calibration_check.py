import joblib
import pandas as pd

bundle = joblib.load("model.pkl")
model, F = bundle["model"], bundle["features"]
d = pd.read_csv("storm_data.csv", parse_dates=["ts"])
t = d[d["ts"] >= "2025-01-01"].copy()
t["p"] = model.predict_proba(t[F])[:, 1]
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
t["score"] = pd.cut(t["p"], bins, include_lowest=True)
print(t.groupby("score", observed=True).agg(
    hours=("p", "size"), real_storm_rate=("storm_next_6h", "mean")).round(3))
