import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "originalPrice",
    "price",
    "discount_percentage",
    "has_free_shipping",
    "is_best_seller",
    "price_bucket_encoded",
]

PRICE_BUCKET_MAP = {"low": 0, "mid": 1, "medium": 1, "high": 2}


def build_feature_row(data: dict) -> pd.DataFrame:
    bucket_raw = str(data.get("price_bucket", "low")).strip().lower()
    bucket_encoded = PRICE_BUCKET_MAP.get(bucket_raw, 0)

    row = {
        "originalPrice":       float(data.get("originalPrice", 100)),
        "price":               float(data.get("price", 50)),
        "discount_percentage": float(data.get("discount_percentage", 0)),
        "has_free_shipping":   int(data.get("has_free_shipping", 0)),
        "is_best_seller":      int(data.get("is_best_seller", 0)),
        "price_bucket_encoded": bucket_encoded,
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict(model, data: dict) -> int:
    try:
        X = build_feature_row(data)
        X = X.replace([np.inf, -np.inf], 0).fillna(0)
        result = model.predict(X)[0]
        return int(max(0, round(float(result))))
    except Exception as e:
        logger.error("predict() failed — %s", e)
        return 0


def train_model(df: pd.DataFrame):
    df = df.copy()

    df["price_bucket_encoded"] = (
        df["price_bucket"].astype(str).str.strip().str.lower()
        .map(PRICE_BUCKET_MAP).fillna(0).astype(int)
    )

    X = df[FEATURE_COLUMNS]
    y = df["sales"]

    clf = RandomForestRegressor(n_estimators=100, random_state=42)
    clf.fit(X, y)

    with open("model.pkl", "wb") as f:
        pickle.dump(clf, f)

    logger.info("model.pkl saved.")
    return clf