import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

FEATURE_COLUMNS = [
    "originalPrice",
    "price",
    "discount_percentage",
    "has_free_shipping",
    "is_best_seller",
    "price_bucket_encoded",
]

PRICE_BUCKET_MAP = {"low": 0, "mid": 1, "medium": 1, "high": 2}

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("furniture_data.csv")
print(f"Loaded {len(df)} rows")
print(f"Columns: {list(df.columns)}")

# ── Numeric clean ─────────────────────────────────────────────────────────────
for col in ["originalPrice", "price"]:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace(r"[$,₹€£\s]", "", regex=True),
        errors="coerce"
    ).fillna(0)

# ── Derive missing columns from what we have ──────────────────────────────────

# discount_percentage — originalPrice se calculate karo
df["discount_percentage"] = np.where(
    df["originalPrice"] > 0,
    ((df["originalPrice"] - df["price"]) / df["originalPrice"] * 100).round(2),
    0
).clip(0)

# has_free_shipping — tagText mein "free" check karo
df["has_free_shipping"] = (
    df["tagText"].astype(str).str.lower().str.contains("free", na=False)
).astype(int)

# is_best_seller — tagText mein "best" check karo
df["is_best_seller"] = (
    df["tagText"].astype(str).str.lower().str.contains("best", na=False)
).astype(int)

# price_bucket
df["price_bucket"] = pd.cut(
    df["price"],
    bins=[0, 500, 2000, float("inf")],
    labels=["Low", "Mid", "High"]
)
df["price_bucket_encoded"] = (
    df["price_bucket"].astype(str).str.lower()
    .map(PRICE_BUCKET_MAP).fillna(0).astype(int)
)

# sales — 'sold' column use karo
df["sales"] = pd.to_numeric(df["sold"], errors="coerce").fillna(0).astype(int)

print(f"Sales sample: {df['sales'].head().tolist()}")
print(f"Discount sample: {df['discount_percentage'].head().tolist()}")

# ── Train ─────────────────────────────────────────────────────────────────────
X = df[FEATURE_COLUMNS]
y = df["sales"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅  model.pkl saved!")
print(f"    Rows: {len(X)}, Avg sales: {y.mean():.1f}, Max: {y.max()}")