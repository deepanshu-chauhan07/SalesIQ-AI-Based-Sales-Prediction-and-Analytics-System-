from flask import Flask, render_template, request, redirect, session, send_file
from model import train_model, predict
from db import get_db
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import os
import pickle

app = Flask(__name__)
app.secret_key = "secret123"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = pickle.load(open("model.pkl", "rb"))
# ================= HELPERS =================
def clean_number(val):
    if val is None:
        return 0
    val = str(val)
    val = val.replace('$', '').replace(',', '').strip()
    try:
        return float(val)
    except:
        return 0

def safe_get(row, keys, default=0):
    for key in keys:
        if key in row and pd.notna(row[key]):
            return row[key]
    return default

# ================= ROOT =================
@app.route('/')
def root():
    if 'user' in session:
        return redirect('/home')
    return redirect('/login')

# ================= HOME =================
@app.route('/home')
def home():
    if 'user' not in session:
        return redirect('/login')
    return render_template('index.html')

# ================= UPLOAD =================
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        return redirect('/login')

    if request.method == 'POST':
        file = request.files.get('file')

        if not file:
            return "No file uploaded"

        try:
            df = pd.read_csv(file)
        except:
            return "Invalid CSV file"

        # 🔥 CLEAN ALL NUMERIC DATA
        for col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna()

        # 🔥 SAFE COLUMN DETECTION
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) < 2:
            return "CSV must contain at least 2 numeric columns"

        x_col = numeric_cols[0]
        y_col = numeric_cols[1]

        predictions = []

        for _, row in df.iterrows():
            data = {
                "productTitle": safe_get(row, ['productTitle','title','name'], 'unknown'),
                "originalPrice": clean_number(safe_get(row, ['originalPrice','mrp', x_col], 100)),
                "price": clean_number(safe_get(row, ['price','selling_price', x_col], 50)),
                "discount_percentage": clean_number(safe_get(row, ['discount'], 10)),
                "price_bucket": "Low",
                "has_free_shipping": 0,
                "is_best_seller": 0,
            }

            try:
                result = int(predict(model, data))
            except:
                result = int(row[y_col]) if y_col in row else 0

            predictions.append(result)

        df['Predicted_Sales'] = predictions
        df['Predicted_Sales'] = df['Predicted_Sales'].fillna(0)
        df = df.replace([float('inf'), -float('inf')], 0)

        # 🔥 HISTORY SAVE SAFE
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO history (username, total_rows, avg_sales)
        VALUES (?, ?, ?)
        """, (session['user'], len(df), round(df['Predicted_Sales'].mean(), 2)))

        conn.commit()
        conn.close()

        global last_df
        last_df = df

        # 🔥 FINAL OUTPUT SAFE
        return render_template(
    'dashboard.html',
    total_rows=len(df),

    avg_sales=round(df['Predicted_Sales'].fillna(0).mean(), 2),
    max_sales=int(df['Predicted_Sales'].fillna(0).max()),

    prices=df[price_col].fillna(0).tolist(),
    sales=df['Predicted_Sales'].tolist(),

    titles=df.get('productTitle', df.index).tolist()
)     

    return render_template('upload.html')

# ================= DOWNLOAD =================
@app.route('/download')
def download():
    global last_df

    buffer = io.StringIO()
    last_df.to_csv(buffer, index=False)
    buffer.seek(0)

    return send_file(
        io.BytesIO(buffer.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='predictions.csv'
    )

# ================= REGISTER =================
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        username = request.form['username']
        password = generate_password_hash(request.form['password'])

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE email=? OR username=?", (email, username))
        existing = cursor.fetchone()

        if existing:
            conn.close()
            return render_template('register.html', error="User already exists")

        cursor.execute("""
        INSERT INTO users (first_name, last_name, email, username, password)
        VALUES (?, ?, ?, ?, ?)
        """, (first_name, last_name, email, username, password))

        conn.commit()
        conn.close()

        return redirect('/login')

    return render_template('register.html')

# ================= LOGIN =================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[5], password):
            session['user'] = username
            return redirect('/home')
        else:
            return render_template('login.html', error="Invalid credentials")

    return render_template('login.html')

# ================= LOGOUT =================
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')
@app.route('/history')
def history():
    if 'user' not in session:
        return redirect('/login')

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM history WHERE username=?", (session['user'],))
    data = cursor.fetchall()

    conn.close()

    return render_template('history.html', data=data)
@app.route('/download_pdf')
def download_pdf():
    file = "report.pdf"
    doc = SimpleDocTemplate(file)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("AI Sales Report", styles['Title']))
    content.append(Paragraph(f"User: {session['user']}", styles['Normal']))

    doc.build(content)

    return send_file(file, as_attachment=True)

# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)