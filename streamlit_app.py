import streamlit as st
import joblib
import numpy as np
from xgboost import XGBClassifier # Required to load the model class
from sklearn.preprocessing import StandardScaler # Required to load the scaler class

# --- 1. Load the Saved Model and Scaler ---
# Using st.cache_resource ensures the app loads these large files only once, making it fast.
@st.cache_resource
def load_assets():
    """Loads the saved model and scaler from the disk."""
    try:
        # Load the fitted model and scaler
        model = joblib.load('xgb_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        st.error("Error: Model or Scaler files not found.")
        st.error("Please run the 'train_model.py' script first to create these assets.")
        return None, None

# Load the assets
model, scaler = load_assets()

# --- 2. Set Up the Page Configuration ---
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩸",
    layout="centered"
)

st.title("🩸 Pima Diabetes Risk Predictor")
st.markdown(
    "Enter your health metrics below. This app uses a pre-trained **XGBoost model** "
    "to estimate the likelihood of Type 2 Diabetes."
)
st.markdown("---")

# --- 3. Create Input Widgets for the 4 Prediction Features ---

if model is not None and scaler is not None:
    
    st.subheader("Enter Health Metrics (4 Key Features)")
    
    # Use columns for a cleaner, two-column layout
    col1, col2 = st.columns(2)
    
    # Input Fields (must be gathered in the correct order for scaling)
    with col1:
        # 1. Glucose
        glucose = st.number_input(
            "Glucose (mg/dL)",
            min_value=40.0, max_value=200.0, value=120.0, step=1.0,
            help="Plasma glucose concentration (2-hour oral test)."
        )
        
        # 2. BMI
        bmi = st.number_input(
            "BMI (Body Mass Index)",
            min_value=15.0, max_value=70.0, value=32.0, step=0.1,
            help="Weight (kg) / (height (m))^2."
        )

    with col2:
        # 3. Age
        age = st.number_input(
            "Age (Years)",
            min_value=21, max_value=100, value=35, step=1
        )
        
        # 4. Blood Pressure
        bp = st.number_input(
            "Blood Pressure (mm Hg)",
            min_value=40.0, max_value=140.0, value=72.0, step=1.0,
            help="Diastolic blood pressure."
        )

    st.markdown("---")

    # --- 4. Prediction Button and Logic ---
    if st.button("Analyze Risk", type="primary"):
        
        # 1. Prepare Input Data (CRITICAL STEP: Order MUST match training order)
        # Assumed Order: ['Glucose', 'BMI', 'Age', 'BloodPressure']
        input_data = np.array([[
            glucose,
            bmi,
            age,
            bp
        ]])
        
        # 2. Scale the input data using the loaded scaler.
        input_scaled = scaler.transform(input_data)
        
        # 3. Make the prediction (Get the probability of the positive class: Diabetes=1)
        prediction_proba = model.predict_proba(input_scaled)[0][1]
        risk_percentage = prediction_proba * 100
        
        # --- 5. Display the Result ---
        st.subheader("Final Risk Assessment")
        
        # Classify and display result
        if prediction_proba >= 0.5:
            st.error(f"**High Risk** - Probability: {risk_percentage:.1f}%", icon="🚨")
            st.warning(
                "The model indicates a high likelihood of a positive diagnosis. "
                "Immediate medical consultation is essential for accurate assessment."
            )
        else:
            st.success(f"**Low Risk** - Probability: {risk_percentage:.1f}%", icon="✅")
            st.info(
                "The predicted risk is low. Maintain a healthy lifestyle and continue "
                "with regular monitoring."
            )
        
        # Show technical details
        with st.expander("Show Model Details"):
            st.write(f"**Model Used:** XGBoost Classifier")
            st.write(f"**Prediction Probability (P(Diabetes=1)):** {prediction_proba:.4f}")
            st.write(f"**Features Used for Prediction:** Glucose, BMI, Age, Blood Pressure (Scaled)")

else:
    st.warning("Application assets are missing. Please run the training script first.")
