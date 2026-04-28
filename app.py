from flask import Flask, render_template, request, redirect, session
import pandas as pd
import numpy as np
import pickle
import sqlite3

app = Flask(__name__)
app.secret_key = "secret123"

# 🔥 Load Model (safe)
try:
    model = pickle.load(open("model.pkl", "rb"))
except:
    model = None

# 🔥 DB connection
def get_db():
    return sqlite3.connect("database.db")


# 🔹 HOME
@app.route('/')
def home():
    return render_template("upload.html")


# 🔹 UPLOAD + ANALYSIS
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('file')

        if not file:
            return "No file uploaded"

        # 🔥 Read CSV
        try:
            df = pd.read_csv(file)
        except:
            return "Invalid CSV file"

        # 🔥 CLEAN DATA (VERY IMPORTANT)
        for col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 🔥 REMOVE BAD VALUES
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()

        # 🔥 GET NUMERIC COLUMNS
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) < 2:
            return "Need at least 2 numeric columns"

        x_col = numeric_cols[0]
        y_col = numeric_cols[1]

        # 🔥 PREDICTION SAFE
        predictions = []
        for _, row in df.iterrows():
            try:
                if model:
                    pred = model.predict([[row[x_col], row[y_col]]])[0]
                else:
                    pred = row[y_col]
            except:
                pred = 0

            predictions.append(pred)

        df['Predicted_Sales'] = predictions

        # 🔥 FINAL CLEAN (NO NaN)
        df['Predicted_Sales'] = pd.to_numeric(df['Predicted_Sales'], errors='coerce')
        df['Predicted_Sales'] = df['Predicted_Sales'].fillna(0)

        # 🔥 SAFE STATS
        total_rows = len(df)
        avg_sales = round(df['Predicted_Sales'].mean(), 2) if total_rows > 0 else 0
        max_sales = int(df['Predicted_Sales'].max()) if total_rows > 0 else 0

        # 🔥 PIE DATA
        mean_val = df['Predicted_Sales'].mean()
        low = len(df[df['Predicted_Sales'] < mean_val])
        medium = len(df[(df['Predicted_Sales'] >= mean_val) & (df['Predicted_Sales'] < mean_val * 1.5)])
        high = len(df[df['Predicted_Sales'] >= mean_val * 1.5])

        # 🔥 RETURN DASHBOARD
        return render_template(
            "dashboard.html",
            total_rows=total_rows,
            avg_sales=avg_sales,
            max_sales=max_sales,
            prices=df[x_col].fillna(0).tolist(),
            sales=df['Predicted_Sales'].tolist(),
            titles=df.index.tolist(),
            low=low,
            medium=medium,
            high=high,
            x_label=x_col,
            y_label="Predicted Sales"
        )

    return render_template("upload.html")


if __name__ == "__main__":
    app.run(debug=True)