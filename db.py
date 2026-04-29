from flask import Flask, render_template, request, redirect, session, send_file
from model import predict
from mydb import get_db                                    # ← mydb
from werkzeug.security import generate_password_hash, check_password_hash
from csv_detector import process_csv, extract_features
import pandas as pd
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
app.secret_key = os.environ.get("SECRET_KEY", "secret123")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Model load ────────────────────────────────────────────────────────────────
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    logger.info("✅ Model loaded successfully.")
except FileNotFoundError:
    model = None
    logger.error("❌ model.pkl not found — run retrain.py first.")

last_df: pd.DataFrame = pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
#  SAFE PREDICT
# ══════════════════════════════════════════════════════════════════════════════

def safe_predict(row: pd.Series, idx: int) -> int:
    if model is None:
        return 0
    try:
        features = extract_features(row)
        return int(predict(model, features))
    except KeyError as e:
        logger.warning("Row %d — missing key: %s", idx, e)
    except ValueError as e:
        logger.warning("Row %d — value error: %s", idx, e)
    except Exception as e:
        logger.error("Row %d — error: %s", idx, e)
    return 0


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
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
            return render_template('upload.html', error="Koi file select nahi ki.")

        # ── 1. Read CSV ────────────────────────────────────────────────────────
        try:
            df = pd.read_csv(file)
        except Exception as e:
            logger.error("CSV read error: %s", e)
            return render_template('upload.html', error=f"CSV read nahi ho saka: {e}")

        if df.empty:
            return render_template('upload.html', error="CSV file empty hai.")

        logger.info("Uploaded CSV — rows: %d, columns: %s", len(df), list(df.columns))

        # ── 2. Universal processing ────────────────────────────────────────────
        try:
            df, col_report = process_csv(df)
            logger.info("Column mapping report: %s", col_report)
        except Exception as e:
            logger.error("CSV processing error: %s", e)
            return render_template('upload.html', error=f"CSV process nahi ho saka: {e}")

        # ── 3. Predict ─────────────────────────────────────────────────────────
        df["Predicted_Sales"] = [
            safe_predict(row, idx) for idx, row in df.iterrows()
        ]

        nonzero = (df["Predicted_Sales"] != 0).sum()
        logger.info("Predictions done — %d/%d non-zero", nonzero, len(df))

        # ── 4. Save history ────────────────────────────────────────────────────
        try:
            conn = get_db()
            conn.execute(
                "INSERT INTO history (username, total_rows, avg_sales) VALUES (?, ?, ?)",
                (session['user'], len(df), round(df["Predicted_Sales"].mean(), 2)),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("DB error (non-fatal): %s", e)

        # ── 5. Store for download ──────────────────────────────────────────────
        global last_df
        last_df = df

        # ── 6. Price column for chart ──────────────────────────────────────────
        price_col = next(
            (c for c in ["price", "originalPrice"] if c in df.columns), None
        )
        prices = df[price_col].fillna(0).tolist() if price_col else [0] * len(df)
        titles = (
            df["productTitle"].fillna("Unknown").tolist()
            if "productTitle" in df.columns
            else [f"Row {i+1}" for i in range(len(df))]
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
        return "Pehle koi CSV upload karo.", 400
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
    doc = SimpleDocTemplate(filepath)
    styles = getSampleStyleSheet()
    doc.build([
        Paragraph("AI Sales Report", styles['Title']),
        Paragraph(f"User: {session.get('user', 'unknown')}", styles['Normal']),
    ])
    return send_file(filepath, as_attachment=True)


# ── Debug route (temporary) ───────────────────────────────────────────────────
@app.route('/debug')
def debug():
    if 'user' not in session:
        return redirect('/login')
    import json
    from csv_detector import process_csv, extract_features

    # Dummy test CSV
    test_data = {
        "Item Name":      ["Sofa Set", "Coffee Table", "Wardrobe"],
        "MRP":            ["$599", "$199", "$899"],
        "Selling Price":  ["$499", "$149", "$699"],
        "Tag":            ["Free Shipping", "Best Seller", "Sale"],
        "Units Sold":     [45, 23, 67],
    }
    df_test = pd.DataFrame(test_data)
    df_processed, report = process_csv(df_test)

    result = {
        "original_columns": list(test_data.keys()),
        "mapped_columns":   report,
        "final_columns":    list(df_processed.columns),
        "sample_row":       df_processed.iloc[0].to_dict(),
        "features_extracted": extract_features(df_processed.iloc[0]),
    }
    return f"<pre style='background:#111;color:#0f0;padding:20px'>{json.dumps(result, indent=2, default=str)}</pre>"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)