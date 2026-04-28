import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

pickle.dump(model, open("model.pkl", "wb"))

def train_model():
    df = pd.read_csv("furniture_data.csv")

    # --- CLEANING ---
    def clean_currency(x):
        if pd.isna(x): return np.nan
        return float(str(x).replace('$','').replace(',',''))

    df['price'] = df['price'].apply(clean_currency)
    df['originalPrice'] = df['originalPrice'].apply(clean_currency)

    df['tagText'] = df['tagText'].fillna("none").str.lower()
    df['productTitle'] = df['productTitle'].fillna("unknown")

    # --- FEATURE ENGINEERING ---
    df['discount_percentage'] = ((df['originalPrice'] - df['price']) / df['originalPrice']) * 100
    df['has_free_shipping'] = df['tagText'].str.contains('free shipping').astype(int)
    df['is_best_seller'] = df['tagText'].str.contains('best seller').astype(int)
    df['price_bucket'] = pd.qcut(df['price'], 3, labels=['Low','Medium','High'])

    X = df[['productTitle','originalPrice','price','discount_percentage','price_bucket','has_free_shipping','is_best_seller']]
    y = df['sold']

    # --- PIPELINE ---
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['originalPrice','price','discount_percentage','has_free_shipping','is_best_seller']),
        ('cat', OneHotEncoder(drop='first'), ['price_bucket']),
        ('text', TfidfVectorizer(max_features=10), 'productTitle')
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor())
    ])

    model.fit(X, y)
    return model


def predict(model, input_data):
    df = pd.DataFrame([input_data])
    return model.predict(df)[0]