import sqlite3

conn = sqlite3.connect("database.db")
cursor = conn.cursor()

# USERS TABLE
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    email TEXT UNIQUE,
    username TEXT UNIQUE,
    password TEXT
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    total_rows INTEGER,
    avg_sales REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# PREDICTIONS TABLE
cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    productTitle TEXT,
    price REAL,
    predicted_sales REAL
)
""")
 
conn.commit()
conn.close()

print("Database created successfully ✅")