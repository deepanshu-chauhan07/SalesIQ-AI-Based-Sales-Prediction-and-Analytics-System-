from flask import Flask, render_template, request, redirect, session, send_file
from model import train_model, predict
from db import get_db
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import io
import os
import pickle
import logging
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "secret123")   # prefer env var

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Model ──────────────────────────────────────────────────────────────────────
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully.")
except FileNotFoundError:
    model = None
    logger.error("model.pkl not found — predictions will be 0.")

last_df: pd.DataFrame = pd.DataFrame()   # typed global; avoids NameError


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 ── CSV INGESTION & TYPE INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

# Column aliases: canonical name → list of possible CSV headers (case-insensitive)
COLUMN_ALIASES: dict[str, list[str]] = {
    "productTitle":       ["producttitle", "title", "name", "product_name", "item"],
    "originalPrice":      ["originalprice", "mrp", "list_price", "original_price"],
    "price":              ["price", "selling_price", "sale_price", "discounted_price"],
    "discount_percentage":["discount", "discount_percentage", "discount_pct", "off"],
    "price_bucket":       ["price_bucket", "bucket", "price_range"],
    "shipping":           ["shipping", "free_shipping", "is_free_shipping"],
    "bestseller":         ["bestseller", "is_best_seller", "best_seller", "top_seller"],
}

NUMERIC_FIELDS  = {"originalPrice", "price", "discount_percentage"}
BOOLEAN_FIELDS  = {"shipping", "bestseller"}


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename CSV columns to canonical names using COLUMN_ALIASES.
    Matching is case- and whitespace-insensitive.
    Unrecognised columns are kept as-is (they won't break anything).
    """
    lower_map = {col.strip().lower(): col for col in df.columns}
    rename: dict[str, str] = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower_map and canonical not in df.columns:
                rename[lower_map[alias]] = canonical
                break

    df = df.rename(columns=rename)
    logger.info("Columns after normalisation: %s", list(df.columns))
    return df


def infer_and_cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dynamically cast columns to the correct dtype:
      • Numeric  — strip currency symbols / commas, coerce to float, fill NaN → 0
      • Boolean  — map yes/true/1 → 1, everything else → 0
      • String   — strip whitespace, fill NaN → 'unknown'

    Any column NOT in the alias map is auto-inspected: if pandas can cast it
    to numeric after stripping common currency chars, it does so.
    """
    known_cols = {c for aliases in COLUMN_ALIASES.values() for c in aliases}
    known_cols |= set(COLUMN_ALIASES.keys())

    # ── Numeric canonical fields ───────────────────────────────────────────────
    for field in NUMERIC_FIELDS:
        if field in df.columns:
            df[field] = (
                df[field]
                .astype(str)
                .str.replace(r"[$,₹€£\s]", "", regex=True)   # multi-currency
                .pipe(pd.to_numeric, errors="coerce")
                .fillna(0)
            )

    # ── Boolean canonical fields ───────────────────────────────────────────────
    TRUTHY = {"1", "yes", "true", "y", "on"}
    for field in BOOLEAN_FIELDS:
        if field in df.columns:
            df[field] = (
                df[field]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(TRUTHY)
                .astype(int)
            )

    # ── String canonical fields ────────────────────────────────────────────────
    for field in ("productTitle", "price_bucket"):
        if field in df.columns:
            df[field] = df[field].astype(str).str.strip().replace("nan", "unknown")

    # ── Auto-detect remaining unknown numeric columns ──────────────────────────
    for col in df.columns:
        if col in known_cols:
            continue
        if df[col].dtype == object:
            cleaned = (
                df[col]
                .astype(str)
                .str.replace(r"[$,₹€£\s]", "", regex=True)
                .pipe(pd.to_numeric, errors="coerce")
            )
            if cleaned.notna().mean() > 0.5:          # >50 % parseable → numeric
                df[col] = cleaned.fillna(0)
                logger.info("Auto-cast column '%s' to numeric.", col)

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 ── ROW-LEVEL FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(row: pd.Series) -> dict:
    """
    Build a clean feature dict from a normalised row.
    Falls back to safe defaults when a field is absent.
    """
    orig_price = float(row.get("originalPrice", 100) or 100)
    sell_price = float(row.get("price", orig_price * 0.9) or orig_price * 0.9)

    # Derive discount when not explicit
    if "discount_percentage" in row and row["discount_percentage"] != 0:
        discount = float(row["discount_percentage"])
    elif orig_price > 0:
        discount = round((orig_price - sell_price) / orig_price * 100, 2)
    else:
        discount = 0.0

    # Infer price_bucket when absent
    if "price_bucket" in row and str(row["price_bucket"]) not in ("nan", "unknown", ""):
        bucket = str(row["price_bucket"])
    elif sell_price < 500:
        bucket = "Low"
    elif sell_price < 2000:
        bucket = "Mid"
    else:
        bucket = "High"

    return {
        "productTitle":       str(row.get("productTitle", "unknown")),
        "originalPrice":      orig_price,
        "price":              sell_price,
        "discount_percentage":discount,
        "price_bucket":       bucket,
        "has_free_shipping":  int(row.get("shipping", 0)),
        "is_best_seller":     int(row.get("bestseller", 0)),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 ── PREDICTION WITH GRANULAR ERROR HANDLING
# ══════════════════════════════════════════════════════════════════════════════

def safe_predict(row: pd.Series, idx: int) -> int:
    """
    Wraps predict() with per-row error isolation.
    Logs the failure reason instead of silently returning 0,
    making debugging vastly easier.
    """
    if model is None:
        return 0
    try:
        features = extract_features(row)
        result = predict(model, features)
        return int(result)
    except KeyError as e:
        logger.warning("Row %d — missing feature key: %s", idx, e)
    except ValueError as e:
        logger.warning("Row %d — type/value error: %s", idx, e)
    except Exception as e:
        logger.error("Row %d — unexpected prediction error: %s", idx, e)
    return 0


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES  (unchanged surface API — only internals improved)
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def root():
    return redirect('/home' if 'user' in session else '/login')


@app.route('/home')
def home():
    if 'user' not in session:
        return redirect('/login')
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        return redirect('/login')

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not file.filename:
            return render_template('upload.html', error="No file selected.")

        # ── 1. Read ────────────────────────────────────────────────────────────
        try:
            df = pd.read_csv(file)
        except Exception as e:
            logger.error("CSV parse failure: %s", e)
            return render_template('upload.html', error=f"Could not read CSV: {e}")

        if df.empty:
            return render_template('upload.html', error="Uploaded CSV is empty.")

        # ── 2. Normalise + cast ────────────────────────────────────────────────
        df = normalise_columns(df)
        df = infer_and_cast_types(df)


        # Batch prediction — sab rows ek saath predict hongi (fast)
        try:
           from model import build_feature_row
           feature_rows = [build_feature_row(extract_features(row)) for _, row in df.iterrows()]
           X_batch = pd.concat(feature_rows, ignore_index=True)
           df["Predicted_Sales"] = model.predict(X_batch).astype(int)
           logger.info("Batch prediction successful.")
        except Exception as e:
           logger.error("Batch prediction failed, falling back: %s", e)
           df["Predicted_Sales"] = [
               safe_predict(row, idx) for idx, row in df.iterrows()
    ]
        nonzero = (df["Predicted_Sales"] != 0).sum()
        logger.info(
            "Predictions complete — %d/%d rows with non-zero sales.",
            nonzero, len(df)
        )

        # ── 4. Persist history ─────────────────────────────────────────────────
        try:
            conn = get_db()
            conn.execute(
                "INSERT INTO history (username, total_rows, avg_sales) VALUES (?, ?, ?)",
                (session['user'], len(df), round(df["Predicted_Sales"].mean(), 2)),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("DB write failed: %s", e)   # non-fatal — don't crash

        # ── 5. Stash for download ──────────────────────────────────────────────
        global last_df
        last_df = df

        # ── 6. Choose price column for display ────────────────────────────────
        price_col = next(
            (c for c in ("price", "originalPrice") if c in df.columns),
            None,
        )
        prices = df[price_col].tolist() if price_col else [0] * len(df)

        # Ensure productTitle exists for chart labels
        titles = (
            df["productTitle"].fillna("Unknown").tolist()
            if "productTitle" in df.columns
            else [f"Row {i}" for i in range(len(df))]
        )

        return render_template(
            'dashboard.html',
            total_rows=len(df),
            avg_sales=round(float(df["Predicted_Sales"].mean()), 2),
            max_sales=int(df["Predicted_Sales"].max()),
            prices=prices,
            sales=df["Predicted_Sales"].tolist(),
            titles=titles,
        )

    return render_template('upload.html')


@app.route('/download')
def download():
    if last_df.empty:
        return "No data to download yet.", 400
    buf = io.BytesIO()
    last_df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(buf, mimetype='text/csv', as_attachment=True,
                     download_name='predictions.csv')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name  = request.form['last_name']
        email      = request.form['email']
        username   = request.form['username']
        password   = generate_password_hash(request.form['password'])

        conn   = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM users WHERE email=? OR username=?", (email, username))
        if cursor.fetchone():
            conn.close()
            return render_template('register.html', error="User already exists.")

        cursor.execute(
            "INSERT INTO users (first_name,last_name,email,username,password) VALUES (?,?,?,?,?)",
            (first_name, last_name, email, username, password),
        )
        conn.commit()
        conn.close()
        return redirect('/login')

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn   = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[5], password):
            session['user'] = username
            return redirect('/home')
        return render_template('login.html', error="Invalid credentials.")

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')


@app.route('/history')
def history():
    if 'user' not in session:
        return redirect('/login')

    conn   = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM history WHERE username=?", (session['user'],))
    data = cursor.fetchall()
    conn.close()
    return render_template('history.html', data=data)


@app.route('/download_pdf')
def download_pdf():
    filepath = "report.pdf"
    doc      = SimpleDocTemplate(filepath)
    styles   = getSampleStyleSheet()
    doc.build([
        Paragraph("AI Sales Report", styles['Title']),
        Paragraph(f"User: {session.get('user','unknown')}", styles['Normal']),
    ])
    return send_file(filepath, as_attachment=True)

@app.route('/debug_model')
def debug_model():
    """Temporary diagnostic — remove before production."""
    if 'user' not in session:
        return redirect('/login')

    import json

    # Simulate a single prediction with known values
    test_input = {
        "productTitle":        "Test Chair",
        "originalPrice":       199.99,
        "price":               149.99,
        "discount_percentage": 25.0,
        "price_bucket":        "Mid",
        "has_free_shipping":   1,
        "is_best_seller":      0,
    }

    report = {"test_input": test_input}

    # Check model exists
    report["model_loaded"] = model is not None
    if model:
        report["model_type"] = str(type(model))
        try:
            report["model_features"] = list(
                getattr(model, "feature_names_in_", ["(not stored — sklearn < 1.0)"])
            )
        except Exception:
            report["model_features"] = "unavailable"

        # Run prediction
        from model import build_feature_row
        X = build_feature_row(test_input)
        report["feature_row_sent"] = X.to_dict(orient="records")[0]

        try:
            raw = model.predict(X)[0]
            report["raw_prediction"] = float(raw)
        except Exception as e:
            report["prediction_error"] = str(e)

    return f"<pre>{json.dumps(report, indent=2)}</pre>"
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)