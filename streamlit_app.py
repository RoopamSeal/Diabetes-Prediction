import streamlit as st
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# --- 1. Load Saved Model and Scaler ---
# Use st.cache_resource to load these assets only once
@st.cache_resource
def load_assets():
    """
    Loads the saved StandardScaler and XGBoost model from disk.
    """
    try:
        # Load the scaler fit on the 4 RFE features
        scaler = joblib.load('scaler_rfe_4.joblib')
        
        # Load the trained XGBoost model (best Recall)
        model = joblib.load('xgb_rfe_model.joblib')
        
        return scaler, model
    except FileNotFoundError:
        # Display an error if the files are missing
        st.error(
            "Error: Model or Scaler file not found. "
            "Please ensure 'scaler_rfe_4.joblib' and 'xgb_rfe_model.joblib' "
            "are in the same directory as this app."
        )
        return None, None

# Load the assets
scaler, model = load_assets()

# --- 2. Page Configuration and Title ---
st.set_page_config(
    page_title="Pima Diabetes Predictor",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🩺 Pima Diabetes Risk Predictor")
st.markdown(
    "This app uses an **XGBoost** model, trained on the 4 most "
    "predictive features, to estimate the risk of Type 2 Diabetes."
)
st.markdown("---")

# Only run the app if the model and scaler loaded successfully
if model is not None and scaler is not None:
    
    # --- 3. User Input Fields ---
    st.subheader("Input Patient Data (Top 4 Features)")
    st.caption(
        "Please enter the values for the features identified by "
        "Recursive Feature Elimination (RFE)."
    )
    
    # Create two columns for a cleaner layout
    col1, col2 = st.columns(2)
    
    with col1:
        glucose = st.number_input(
            "Glucose (mg/dL)",
            min_value=40.0, max_value=200.0, value=120.0, step=1.0,
            help="Plasma glucose concentration (2-hour oral test)."
        )
        
        age = st.number_input(
            "Age (Years)",
            min_value=21, max_value=100, value=35, step=1,
            help="Patient's age."
        )
    
    with col2:
        bmi = st.number_input(
            "BMI (Body Mass Index)",
            min_value=15.0, max_value=70.0, value=32.0, step=0.1,
            help="Weight (kg) / (height (m))^2."
        )
        
        dpf = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.07, max_value=2.50, value=0.47, step=0.01,
            format="%.3f",
            help="Genetic risk factor based on family history."
        )

    st.markdown("---")

    # --- 4. Prediction Logic ---
    if st.button("Analyze Risk", type="primary"):
        # Create a 2D NumPy array in the correct feature order
        # This order MUST match the one used to train the scaler
        input_data = np.array([[
            glucose,
            bmi,
            age,
            dpf
        ]])
        
        # Apply the loaded StandardScaler
        input_scaled = scaler.transform(input_data)
        
        # Get the probability of the positive class (Diabetes)
        prediction_proba = model.predict_proba(input_scaled)[0][1]
        risk_percentage = prediction_proba * 100
        
        # --- 5. Display Results ---
        st.subheader("Prediction Result")
        
        # Use a 0.5 threshold for classification
        if prediction_proba >= 0.5:
            st.error(f"**High Risk ({risk_percentage:.1f}%)**", icon="🚨")
            st.warning(
                "The model predicts a high likelihood of diabetes. "
                "Consultation with a healthcare professional is strongly recommended."
            )
        else:
            st.success(f"**Low Risk ({risk_percentage:.1f}%)**", icon="✅")
            st.info(
                "The model predicts a low likelihood of diabetes. "
                "Continue to maintain a healthy lifestyle and monitor vitals."
            )
        
        # Show technical details in an expander
        with st.expander("Show Technical Details"):
            st.write(f"**Model Used:** XGBoost Classifier (RFE-selected features)")
            st.write(f"**Probability (P(Diabetes=1)):** {prediction_proba:.4f}")
            st.write(f"**Input Features (Unscaled):**")
            st.json({
                "Glucose": glucose,
                "BMI": bmi,
                "Age": age,
                "DiabetesPedigreeFunction": dpf
            })

else:
    st.error(
        "Application cannot start. Please check the console for file-loading errors."
    )
