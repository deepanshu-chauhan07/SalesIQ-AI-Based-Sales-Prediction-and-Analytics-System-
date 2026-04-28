import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# ==========================================
# 0. HELPER: GENERATE SYNTHETIC DATASET
# ==========================================
# (Since the actual dataset is not provided, we generate a realistic synthetic one to ensure this script runs without errors)
def generate_synthetic_data(filename="furniture_data.csv"):
    if not os.path.exists(filename):
        print(f"Dataset '{filename}' not found. Generating synthetic dataset for demonstration...")
        np.random.seed(42)
        n = 1000
        titles = ["Modern Sofa", "Wooden Dining Table", "Ergonomic Office Chair", "Glass Coffee Table", "King Size Bed Frame", "Bookshelf", "Leather Recliner"]
        tags = ["Best Seller", "Free shipping", "Sale", "", "New Arrival", "Free shipping, Sale"]
        
        data = {
            "productTitle": np.random.choice(titles, n),
            "originalPrice": np.random.uniform(100, 2000, n),
            "tagText": np.random.choice(tags, n, p=[0.2, 0.3, 0.15, 0.2, 0.1, 0.05])
        }
        
        df = pd.DataFrame(data)
        # Apply discount to some to create 'price'
        df["discount"] = np.where(df["tagText"].str.contains("Sale"), np.random.uniform(0.1, 0.4, n), 0)
        df["price"] = df["originalPrice"] * (1 - df["discount"])
        
        # Add some dirty data to demonstrate cleaning
        df["originalPrice"] = df["originalPrice"].apply(lambda x: f"${x:,.2f}")
        df["price"] = df["price"].apply(lambda x: f"${x:,.2f}")
        
        # Introduce a few NaNs for cleaning demonstration
        df.loc[np.random.choice(n, 20, replace=False), "price"] = np.nan
        df.loc[np.random.choice(n, 15, replace=False), "tagText"] = np.nan
        
        # Target variable "sold"
        # Sold depends inversely on price, and positively on "Best Seller" / "Free shipping"
        base_sold = 5000 / (df["price"].str.replace("$", "").str.replace(",", "").astype(float).fillna(500) + 50)
        tag_multiplier = np.where(df["tagText"].str.contains("Best Seller", na=False), 2.5, 1)
        shipping_multiplier = np.where(df["tagText"].str.contains("Free shipping", na=False), 1.5, 1)
        
        sold = (base_sold * tag_multiplier * shipping_multiplier * np.random.uniform(0.8, 1.2, n)).astype(int)
        df["sold"] = sold + np.random.randint(0, 50, n)  # Add some noise
        
        df.drop(columns=["discount"], inplace=True)
        df.to_csv(filename, index=False)
        print("Generated and saved synthetic dataset.\n")

generate_synthetic_data()

# ==========================================
# 1. DATA UNDERSTANDING
# ==========================================
print("--- 1. DATA UNDERSTANDING ---")
df = pd.read_csv("furniture_data.csv")

print(f"Dataset Shape: {df.shape}")
print("\nColumns:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nMissing Values Count:\n", df.isnull().sum())
print("\nFirst 5 rows:")
print(df.head())

# ==========================================
# 2. DATA CLEANING
# ==========================================
print("\n--- 2. DATA CLEANING ---")

# A. Clean price columns (Remove '$', commas, convert to float)
def clean_currency(x):
    if pd.isna(x): return np.nan
    if isinstance(x, str):
        cleaned = x.replace('$', '').replace(',', '').strip()
        return float(cleaned) if cleaned != '' else np.nan
    return float(x)

for col in ['originalPrice', 'price']:
    df[col] = df[col].apply(clean_currency)

# B. Handle Missing Values
df['originalPrice'] = df['originalPrice'].fillna(df['price'])
df['price'] = df['price'].fillna(df['originalPrice'])

median_p = df['price'].median()
if pd.isna(median_p): median_p = 100.0  # Fallback
df['price'] = df['price'].fillna(median_p)
df['originalPrice'] = df['originalPrice'].fillna(median_p)

# Handle text columns
df['productTitle'] = df['productTitle'].fillna("Unknown Product")
df['tagText'] = df['tagText'].fillna("none").astype(str).str.lower().replace("", "none")

# C. Handle Target
df['sold'] = pd.to_numeric(df['sold'], errors='coerce').fillna(0)

# D. Remove Outliers in "sold"
# Using IQR method
Q1 = df['sold'].quantile(0.25)
Q3 = df['sold'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR

# Cap outliers to upper_bound to prevent data loss while minimizing extreme value impact
df['sold'] = np.where(df['sold'] > upper_bound, upper_bound, df['sold'])

print("After Cleaning - Missing Values:\n", df.isnull().sum())


# ==========================================
# 3. EXPLORATORY DATA ANALYSIS (EDA) & 8. VISUALIZATION
# ==========================================
print("\n--- 3. EXPLORATORY DATA ANALYSIS (EDA) ---")
print("Generating Plots... (Check your local folder for PNG files)")

sns.set_theme(style="whitegrid")

# 3.1 Distribution Plots
plt.figure(figsize=(10, 5))
sns.histplot(df['price'], bins=30, kde=True, color='teal')
plt.title('Distribution of Product Prices')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.savefig('distribution_price.png')
plt.close()

plt.figure(figsize=(10, 5))
sns.histplot(df['sold'], bins=30, kde=True, color='coral')
plt.title('Distribution of Products Sold')
plt.xlabel('Units Sold')
plt.ylabel('Frequency')
plt.savefig('distribution_sold.png')
plt.close()

# 3.2 Scatter Plot: Price vs Sold
plt.figure(figsize=(10, 5))
sns.scatterplot(x='price', y='sold', data=df, alpha=0.6, color='purple')
plt.title('Impact of Price on Units Sold')
plt.xlabel('Price ($)')
plt.ylabel('Units Sold')
plt.savefig('scatter_price_vs_sold.png')
plt.close()

# 3.3 Correlation Heatmap
plt.figure(figsize=(8, 6))
correlation = df[['originalPrice', 'price', 'sold']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()

# 3.4 TagText vs Sold Analysis
plt.figure(figsize=(12, 6))
top_tags = df['tagText'].value_counts().index[:5]
sns.boxplot(x='tagText', y='sold', data=df[df['tagText'].isin(top_tags)], palette="Set2")
plt.title('Units Sold by Top Tags')
plt.xticks(rotation=45)
plt.savefig('boxplot_tags_vs_sold.png')
plt.close()


# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================
print("\n--- 4. FEATURE ENGINEERING ---")

# A. Discount Percentage
df['discount_percentage'] = np.where(
    df['originalPrice'] > 0,
    ((df['originalPrice'] - df['price']) / df['originalPrice']) * 100,
    0
)
df['discount_percentage'] = df['discount_percentage'].replace([np.inf, -np.inf], 0).fillna(0)

# B. Price Bucket (Low, Medium, High)
df['price_bucket'] = pd.qcut(df['price'].rank(method='first'), q=3, labels=['Low', 'Medium', 'High'])

# C. Boolean Features for key tags
df['has_free_shipping'] = df['tagText'].astype(str).str.contains('free shipping').astype(int)
df['is_best_seller'] = df['tagText'].astype(str).str.contains('best seller').astype(int)

print("Engineered Features preview:")
print(df[['price', 'discount_percentage', 'price_bucket', 'has_free_shipping', 'is_best_seller']].head())


# ==========================================
# 5. MODEL PREPARATION & BUILDING
# ==========================================
print("\n--- 5. MODEL PREPARATION & BUILDING ---")

# Define features (X) and target (y)
X = df[['productTitle', 'originalPrice', 'price', 'discount_percentage', 'price_bucket', 'has_free_shipping', 'is_best_seller']]
y = df['sold']

# Splitting data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing Pipeline
# - TF-IDF for Text (productTitle)
# - Standard Scaler for Numericals
# - One-Hot Encoding for Categorical (price_bucket)

numeric_features = ['originalPrice', 'price', 'discount_percentage', 'has_free_shipping', 'is_best_seller']
numeric_transformer = StandardScaler()

categorical_features = ['price_bucket']
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

text_feature = 'productTitle'
text_transformer = TfidfVectorizer(max_features=10) # Limit features to avoid curse of dimensionality

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('text', text_transformer, text_feature)
    ])

# Initialize Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
trained_models = {}

print("Training Models...")
for name, model in models.items():
    # Create Pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_test)
    
    # Evaluate
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    results[name] = {'R2 Score': r2, 'RMSE': rmse, 'MAE': mae}
    trained_models[name] = pipeline

# ==========================================
# 6. MODEL EVALUATION
# ==========================================
print("\n--- 6. MODEL EVALUATION ---")
results_df = pd.DataFrame(results).T
print(results_df)

best_model_name = results_df['R2 Score'].idxmax()
print(f"\nBest Model: {best_model_name}")
print(f"Why? It has the highest R2 Score and lower error metrics (RMSE/MAE).")

# ==========================================
# 9. BONUS: PERFORMANCES & FEATURE IMPORTANCE
# ==========================================
print("\n--- 9. BONUS VISUALIZATIONS ---")

best_pipeline = trained_models["Random Forest"]

# Prediction vs Actual Graph
y_pred_best = best_pipeline.predict(X_test)

plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_best, alpha=0.5, color='darkblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Prediction vs Actual (Random Forest)')
plt.xlabel('Actual Sold')
plt.ylabel('Predicted Sold')
plt.savefig('prediction_vs_actual.png')
plt.close()

# Feature Importance extraction for Random Forest
# We have to extract customized feature names from our ColumnTransformer
fitted_rf = best_pipeline.named_steps['regressor']
num_cols = numeric_features
cat_cols = best_pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features).tolist()
text_cols = best_pipeline.named_steps['preprocessor'].transformers_[2][1].get_feature_names_out([text_feature]).tolist()

all_feature_names = num_cols + cat_cols + text_cols
importances = fitted_rf.feature_importances_

feature_imp_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values('Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.savefig('feature_importance.png')
plt.close()

print("All tasks completed successfully. Visualizations saved as PNG files in current directory.")
