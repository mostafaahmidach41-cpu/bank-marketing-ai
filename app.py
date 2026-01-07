import streamlit as st
from supabase import create_client
import pickle
import numpy as np

# --- 1. DATABASE CONNECTION SETUP ---
# Project credentials based on your project settings
URL = "https://ixwvplxnfdjbmdsvdpu.supabase.co"
KEY = "sb_publishable_666yE2Qkv09Y5NQ_QlQaEg_L8fneOgL"

try:
    supabase = create_client(URL, KEY)
except Exception:
    st.error("Connection Error: Database unreachable.")

# --- 2. AUTHENTICATION FUNCTION ---
def verify_license(user_input):
    try:
        # Checking the license table after your successful SQL execution
        response = supabase.table("licenses").select("*").eq("key_value", user_input.strip()).eq("is_active", True).execute()
        return len(response.data) > 0
    except Exception:
        return False

# --- 3. LOGIN INTERFACE ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.set_page_config(page_title="Bank AI Login")
    st.title("Bank AI - Enterprise Edition")
    st.info("Access Restricted: Please activate your license.")
    
    # Matching your app's current UI
    user_key = st.text_input("Enter License Key", type="password")
    
    if st.button("Activate via Cloud"):
        # Testing with your key: PREMIUM-BANK-2026
        if verify_license(user_key):
            st.session_state.authenticated = True
            st.success("Identity Verified!")
            st.rerun()
        else:
            # Error handling for invalid attempts
            st.error("Invalid or Expired License Key.")
    st.stop()

# --- 4. MAIN APPLICATION (AFTER LOGIN) ---
st.set_page_config(page_title="AI Prediction Dashboard", layout="wide")
st.title("Customer Assessment Terminal")

# Sidebar for logout and status
with st.sidebar:
    st.success("System Status: Online")
    if st.button("Sign Out"):
        st.session_state.authenticated = False
        st.rerun()

# --- 5. AI MODEL LOADING ---
@st.cache_resource
def load_ai_assets():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("Error: AI Model files ('model.pkl', 'scaler.pkl') missing.")
        return None, None

model, scaler = load_ai_assets()

# --- 6. PREDICTION INTERFACE ---
if model and scaler:
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Customer Age", 18, 95, 30)
        balance = st.number_input("Yearly Balance", 0, 500000, 1500)
    with col2:
        duration = st.number_input("Last Contact Duration (sec)", 0, 5000, 200)
        day = st.slider("Day of Month", 1, 31, 15)

    if st.button("Predict Subscription Eligibility"):
        # Simple prediction logic
        features = np.array([[age, balance, day, duration]])
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)
        
        if prediction[0] == 1:
            st.balloons()
            st.success("Prediction: Customer is ELIGIBLE.")
        else:
            st.warning("Prediction: Customer is NOT ELIGIBLE.")
