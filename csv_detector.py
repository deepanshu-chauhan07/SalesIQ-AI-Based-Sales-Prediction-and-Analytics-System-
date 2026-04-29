"""
csv_detector.py — Universal CSV Column Detector
Kisi bhi CSV file ke columns automatically detect karta hai
aur standard format mein convert karta hai.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  COLUMN ALIAS MAP — jitne bhi possible names ho sakte hain
# ══════════════════════════════════════════════════════════════════════════════

ALIAS_MAP = {
    "productTitle": [
        "producttitle", "title", "name", "product_name", "item", "item_name",
        "product", "description", "product_title", "item_title", "listing_title",
        "goods_name", "commodity", "article", "label", "heading"
    ],
    "originalPrice": [
        "originalprice", "mrp", "list_price", "original_price", "max_price",
        "market_price", "retail_price", "msrp", "regular_price", "full_price",
        "base_price", "standard_price", "tag_price", "marked_price"
    ],
    "price": [
        "price", "selling_price", "sale_price", "discounted_price", "final_price",
        "current_price", "actual_price", "offer_price", "net_price", "cost",
        "amount", "value", "rate", "unit_price", "purchase_price"
    ],
    "discount_percentage": [
        "discount", "discount_percentage", "discount_pct", "off", "savings_pct",
        "percent_off", "discount_rate", "reduction", "markdown", "savings_percent"
    ],
    "price_bucket": [
        "price_bucket", "bucket", "price_range", "price_tier", "tier",
        "price_category", "price_segment", "segment", "range"
    ],
    "shipping": [
        "shipping", "free_shipping", "is_free_shipping", "freeshipping",
        "delivery", "free_delivery", "shipping_free", "shipment"
    ],
    "bestseller": [
        "bestseller", "is_best_seller", "best_seller", "top_seller",
        "is_bestseller", "bestselling", "popular", "is_popular", "trending",
        "is_trending", "hot", "featured"
    ],
    "sales": [
        "sales", "sold", "units_sold", "quantity_sold", "total_sold",
        "orders", "total_orders", "purchases", "bought", "demand",
        "volume", "sales_count", "order_count", "sell_count"
    ],
    "rating": [
        "rating", "stars", "review_score", "avg_rating", "score",
        "customer_rating", "product_rating", "review_rating"
    ],
    "reviews": [
        "reviews", "review_count", "num_reviews", "total_reviews",
        "ratings_count", "feedback_count", "comments"
    ],
    "category": [
        "category", "cat", "product_category", "type", "product_type",
        "department", "section", "genre", "class", "group"
    ],
    "brand": [
        "brand", "brand_name", "manufacturer", "maker", "company",
        "vendor", "supplier", "seller"
    ],
    "stock": [
        "stock", "inventory", "quantity", "qty", "available", "in_stock",
        "stock_quantity", "units_available", "availability"
    ],
}

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: COLUMN NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def detect_and_rename_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    CSV ke columns ko scan karo aur standard names pe map karo.
    Returns: (renamed_df, mapping_report)
    """
    # lowercase + strip for matching
    col_lower = {col.strip().lower().replace(" ", "_"): col for col in df.columns}
    
    rename_map = {}
    report = {}

    for canonical, aliases in ALIAS_MAP.items():
        for alias in aliases:
            if alias in col_lower and canonical not in rename_map.values():
                original_col = col_lower[alias]
                rename_map[original_col] = canonical
                report[canonical] = original_col
                break

    df = df.rename(columns=rename_map)
    
    logger.info("Column mapping: %s", report)
    logger.info("Final columns: %s", list(df.columns))
    
    return df, report


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: AUTO DETECT PRICE COLUMNS (jab standard names bhi nahi hon)
# ══════════════════════════════════════════════════════════════════════════════

def auto_detect_numeric_as_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agar price/originalPrice column nahi mila to —
    numeric columns mein se price-like columns dhundho.
    """
    # Pehle clean karo — currency symbols hata ke numeric try karo
    for col in df.columns:
        if df[col].dtype == object:
            cleaned = (
                df[col].astype(str)
                .str.replace(r"[$,₹€£¥\s]", "", regex=True)
                .pipe(pd.to_numeric, errors="coerce")
            )
            if cleaned.notna().mean() > 0.7:  # 70%+ parseable = numeric column
                df[col] = cleaned
                logger.info("Auto-converted '%s' to numeric", col)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # price column missing hai?
    if "price" not in df.columns:
        price_candidates = [
            c for c in numeric_cols
            if any(w in c.lower() for w in ["price", "cost", "amount", "rate", "value", "fee"])
        ]
        if price_candidates:
            # Sabse relevant column use karo
            df["price"] = df[price_candidates[0]]
            logger.info("Auto-detected price column: '%s'", price_candidates[0])
        elif numeric_cols:
            # Last resort: pehla numeric column
            df["price"] = df[numeric_cols[0]]
            logger.info("Fallback price column: '%s'", numeric_cols[0])

    # originalPrice missing hai?
    if "originalPrice" not in df.columns:
        if "price" in df.columns:
            df["originalPrice"] = df["price"] * 1.2  # estimate 20% markup
            logger.info("originalPrice estimated as price * 1.2")

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: DERIVE MISSING FEATURE COLUMNS
# ══════════════════════════════════════════════════════════════════════════════

def derive_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Jo columns model ko chahiye but CSV mein nahi hain —
    unhe existing data se derive karo.
    """

    # ── discount_percentage ───────────────────────────────────────────────────
    if "discount_percentage" not in df.columns:
        if "originalPrice" in df.columns and "price" in df.columns:
            df["discount_percentage"] = np.where(
                df["originalPrice"] > 0,
                ((df["originalPrice"] - df["price"]) / df["originalPrice"] * 100).round(2),
                0
            ).clip(0, 100)
        else:
            df["discount_percentage"] = 0
        logger.info("discount_percentage derived")

    # ── has_free_shipping ─────────────────────────────────────────────────────
    if "shipping" not in df.columns:
        # Kisi bhi text column mein "free" dhundho
        text_cols = df.select_dtypes(include=["object"]).columns
        found = False
        for col in text_cols:
            if df[col].astype(str).str.lower().str.contains("free", na=False).mean() > 0.05:
                df["shipping"] = df[col].astype(str).str.lower().str.contains("free", na=False).astype(int)
                logger.info("shipping derived from '%s'", col)
                found = True
                break
        if not found:
            df["shipping"] = 0

    # ── is_best_seller ────────────────────────────────────────────────────────
    if "bestseller" not in df.columns:
        text_cols = df.select_dtypes(include=["object"]).columns
        found = False
        for col in text_cols:
            col_text = df[col].astype(str).str.lower()
            if col_text.str.contains("best|top|popular|trending|hot", na=False).mean() > 0.02:
                df["bestseller"] = col_text.str.contains(
                    "best|top|popular|trending|hot", na=False
                ).astype(int)
                logger.info("bestseller derived from '%s'", col)
                found = True
                break
        if not found:
            # rating se derive karo agar available ho
            if "rating" in df.columns:
                df["bestseller"] = (df["rating"] >= 4.5).astype(int)
                logger.info("bestseller derived from rating >= 4.5")
            else:
                df["bestseller"] = 0

    # ── price_bucket ──────────────────────────────────────────────────────────
    if "price_bucket" not in df.columns:
        q33 = df["price"].quantile(0.33)
        q66 = df["price"].quantile(0.66)
        df["price_bucket"] = pd.cut(
            df["price"],
            bins=[-np.inf, q33, q66, np.inf],
            labels=["Low", "Mid", "High"]
        )
        logger.info("price_bucket derived using percentiles: Low<%.0f, Mid<%.0f, High+", q33, q66)

    # ── productTitle ─────────────────────────────────────────────────────────
    if "productTitle" not in df.columns:
        text_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if text_cols:
            # Sabse unique values wala text column = title
            best = max(text_cols, key=lambda c: df[c].nunique())
            df["productTitle"] = df[best].astype(str)
            logger.info("productTitle derived from '%s'", best)
        else:
            df["productTitle"] = [f"Product {i+1}" for i in range(len(df))]

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4: TYPE CASTING
# ══════════════════════════════════════════════════════════════════════════════

def cast_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """Final type enforcement — sab columns correct dtype mein."""
    
    TRUTHY = {"1", "yes", "true", "y", "on"}

    # Numeric
    for col in ["originalPrice", "price", "discount_percentage"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(r"[$,₹€£¥\s]", "", regex=True)
                .pipe(pd.to_numeric, errors="coerce")
                .fillna(0)
            )

    # Boolean
    for col in ["shipping", "bestseller"]:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip().str.lower().isin(TRUTHY).astype(int)
            else:
                df[col] = df[col].fillna(0).astype(int)

    # String
    for col in ["productTitle", "price_bucket", "category", "brand"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace("nan", "Unknown")

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5: PRICE BUCKET ENCODING
# ══════════════════════════════════════════════════════════════════════════════

PRICE_BUCKET_MAP = {
    "low":    0,
    "mid":    1,
    "medium": 1,
    "high":   2,
}

def encode_price_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df["price_bucket_encoded"] = (
        df["price_bucket"].astype(str).str.strip().str.lower()
        .map(PRICE_BUCKET_MAP)
        .fillna(0)
        .astype(int)
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT — app.py yahi call karega
# ══════════════════════════════════════════════════════════════════════════════

def process_csv(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Koi bhi CSV lo — cleaned, feature-complete DataFrame wapas milega.
    
    Returns:
        df       : processed DataFrame ready for prediction
        report   : column mapping report (for logging/debugging)
    """
    # Step 1: Known column names map karo
    df, report = detect_and_rename_columns(df)

    # Step 2: Unknown numeric columns detect karo
    df = auto_detect_numeric_as_price(df)

    # Step 3: Missing feature columns derive karo
    df = derive_missing_columns(df)

    # Step 4: Types enforce karo
    df = cast_column_types(df)

    # Step 5: Price bucket encode karo
    df = encode_price_bucket(df)

    logger.info("CSV processing complete. Shape: %s", df.shape)
    return df, report


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION — model ke liye row → dict
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(row: pd.Series) -> dict:
    """Processed row se model feature dict banao."""
    return {
        "productTitle":        str(row.get("productTitle", "Unknown")),
        "originalPrice":       float(row.get("originalPrice", 100)),
        "price":               float(row.get("price", 50)),
        "discount_percentage": float(row.get("discount_percentage", 0)),
        "price_bucket":        str(row.get("price_bucket", "Low")),
        "has_free_shipping":   int(row.get("shipping", 0)),
        "is_best_seller":      int(row.get("bestseller", 0)),
    }