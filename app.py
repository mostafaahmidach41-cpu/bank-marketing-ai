import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Customer Assessment Terminal",
    page_icon="📊",
    layout="wide"
)

# --- 2. HEADER (English Interface) ---
st.title("Customer Assessment Terminal")
st.subheader("Enterprise AI Prediction System")
st.markdown("---")

# --- 3. AI MODEL LOADING ---
@st.cache_resource
def load_assets():
    try:
        # Loading the models from your GitHub repository
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("Critical Error: Model files not found in repository.")
        return None, None

model, scaler = load_assets()

# --- 4. PREDICTION INTERFACE ---
if model and scaler:
    # Sidebar status matching your latest screenshots
    st.sidebar.success("System Status: Online")
    st.sidebar.info("License Verification: DISABLED (Open Access)")
    
    st.write("### Customer Input Data")
    
    # Input fields organized into columns to match your UI layout
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Customer Age", 18, 95, 35)
        balance = st.number_input("Average Yearly Balance", 0, 500000, 2500)
        day = st.slider("Last Contact Day of Month", 1, 31, 15)
        
    with col2:
        duration = st.number_input("Last Contact Duration (seconds)", 0, 5000, 300)
        campaign = st.number_input("Number of Contacts during Campaign", 1, 50, 1)

    st.markdown("---")
    
    # Prediction Trigger
    if st.button("Run AI Assessment"):
        try:
            # FIX for ValueError: 
            # Ensure features match the exact number of columns used during model training
            # Based on your UI, we use: Age, Balance, Day, Duration
            features = np.array([[age, balance, day, duration]])
            
            # Scaling and Predicting
            scaled_features = scaler.transform(features)
            prediction = model.predict(scaled_features)
            
            # Displaying Results
            st.subheader("Assessment Result:")
            if prediction[0] == 1:
                st.balloons()
                st.success("✅ ELIGIBLE: Customer is likely to subscribe to the bank product.")
            else:
                st.warning("❌ NOT ELIGIBLE: Customer is unlikely to subscribe at this time.")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

else:
    st.warning("Please ensure model files are uploaded to GitHub to enable predictions.")

# --- 5. FOOTER ---
st.sidebar.markdown("---")
st.sidebar.write("Bank AI v2.0 - Stable Edition")
