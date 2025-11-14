import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib # Used for saving/loading models

print("Starting model training script...")

# --- 1. Load Data ---
try:
    data = pd.read_csv("C:\MSc Data Science\PROJECT WORK\Diabetes\Streamlit\diabetes.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'diabetes.csv' not found. Make sure it's in the same directory.")
    exit()

# --- 2. Clean Data (Handle '0' values) ---
# We must clean the data before training
print("Cleaning data... replacing 0s with median.")
features_to_clean = ['Glucose', 'BloodPressure', 'BMI']
for col in features_to_clean:
    # Replace 0 with NaN
    data[col] = data[col].replace(0, np.nan)
    # Fill NaN with the median of the column
    data[col] = data[col].fillna(data[col].median())

# --- 3. Define Features and Target ---
# Using the 4 features you requested
features = ['Glucose', 'BMI', 'Age', 'BloodPressure']
target = 'Outcome'

X = data[features]
y = data[target]

print(f"Using features: {features}")

# --- 4. Split the Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 5. Scale the Data ---
# We fit the scaler ONLY on the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# --- 6. Train the XGBoost Model ---
print("Training XGBoost model...")
# Handle class imbalance using scale_pos_weight
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

xgb_model = XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train_scaled, y_train)
print("Model training complete.")

# --- 7. Save the Model and Scaler ---
# These two files are what the Streamlit app will load
joblib.dump(xgb_model, 'xgb_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("\n--- Success! ---")
print("Model saved as 'xgb_model.joblib'")
print("Scaler saved as 'scaler.joblib'")
print("You can now run 'streamlit run streamlit_app.py'")