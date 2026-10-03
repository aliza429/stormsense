import json
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             roc_auc_score, confusion_matrix)
from labels import make_labels, df as raw
from features import FEATURES, add_features_all

RAIN_MM, GUST_KMH = 4, 50

data = make_labels(add_features_all(raw), RAIN_MM, GUST_KMH)
data.to_csv("storm_data.csv", index=False)

# time-based split: train on earlier years, test on 2025
train = data[data["ts"] < "2025-01-01"]
test = data[data["ts"] >= "2025-01-01"]
X_train, y_train = train[FEATURES], train["storm_next_6h"].astype(int)
X_test, y_test = test[FEATURES], test["storm_next_6h"].astype(int)

spw = (y_train == 0).sum() / (y_train == 1).sum()
model = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=spw, eval_metric="logloss", random_state=42,
)
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

metrics = {
    "accuracy": float(accuracy_score(y_test, pred)),
    "recall": float(recall_score(y_test, pred)),
    "precision": float(precision_score(y_test, pred)),
    "roc_auc": float(roc_auc_score(y_test, proba)),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    "train_rows": int(len(train)),
    "test_rows": int(len(test)),
    "storm_rate": float(data["storm_next_6h"].mean()),
    "rain_mm": RAIN_MM,
    "gust_kmh": GUST_KMH,
}
json.dump(metrics, open("metrics.json", "w"), indent=2)
joblib.dump({"model": model, "features": FEATURES}, "model.pkl")

print(json.dumps(metrics, indent=2))
print()
print(pd.Series(model.feature_importances_, index=FEATURES)
        .sort_values(ascending=False))
