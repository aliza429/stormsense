import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

bundle = joblib.load("model.pkl")
model, FEATURES = bundle["model"], bundle["features"]

data = pd.read_csv("storm_data.csv", parse_dates=["ts"])
test = data[data["ts"] >= "2025-01-01"]
X = test[FEATURES]

explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X)

# 1) global SHAP importance
imp = pd.Series(np.abs(sv).mean(axis=0), index=FEATURES).sort_values(ascending=False)
print("Mean |SHAP|:\n", imp, "\n")

# 2) among the 200 highest-risk hours, what pushed risk up?
proba = model.predict_proba(X)[:, 1]
top = np.argsort(proba)[-200:]
print("Avg SHAP in top-200 risk hours:\n",
      pd.Series(sv[top].mean(axis=0), index=FEATURES).sort_values(ascending=False), "\n")

# 3) what-if test: does dragging pressure change move the risk?
base = X.median().to_frame().T
print("What-if on a median hour:")
for chg in (0, -3, -6, -9, -12):
    row = base.copy()
    row["pressure_change_6h"] = chg
    row["pressure_change_3h"] = chg / 2
    row["pressure_change_12h"] = chg * 1.5
    print(f"  pressure change {chg:>3} hPa -> risk {model.predict_proba(row)[0, 1]:.3f}")

shap.summary_plot(sv, X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_global.png", dpi=120)
print("\nSaved shap_global.png")

