import streamlit as st
from supabase import create_client
import pickle
import numpy as np
import pandas as pd

# --- 1. CONNECTION CONFIGURATION ---
# Using the credentials from your project settings
URL = "https://ixwvplxnfdjbmdsvdpu.supabase.co"
KEY = "sb_publishable_666yE2Qkv09Y5NQ_QlQaEg_L8fneOgL"

try:
    supabase = create_client(URL, KEY)
except Exception:
    st.error("Database connection failed. Please check your Supabase settings.")

# --- 2. LICENSE VERIFICATION ---
def check_key_status(input_key):
    try:
        # Querying the table where you successfully disabled RLS
        result = supabase.table("licenses").select("*").eq("key_value", input_key.strip()).eq("is_active", True).execute()
        return len(result.data) > 0
    except Exception as e:
        return False

# --- 3. LOGIN INTERFACE ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.set_page_config(page_title="Bank AI - Activation")
    st.title("🛡️ Bank AI - Enterprise Edition")
    st.info("Access Restricted: Please enter a valid license key.")
    
    # Matches the input field in your screenshot
    license_input = st.text_input("Enter License Key", type="password")
    
    if st.button("Activate via Cloud"):
        # The key we verified: PREMIUM-BANK-2026
        if check_key_status(license_input):
            st.session_state.logged_in = True
            st.success("Identity Verified. Loading Dashboard...")
            st.rerun()
        else:
            # Handles the error seen in your app
            st.error("Invalid or Expired License Key.")
    st.stop()

# --- 4. MAIN DASHBOARD (AFTER ACTIVATION) ---
st.set_page_config(page_title="Customer Assessment Terminal", layout="wide")
st.title("📊 Customer Assessment Terminal")

with st.sidebar:
    st.success("Connection: Online")
    if st.button("Log Out"):
        st.session_state.logged_in = False
        st.rerun()

# --- 5. AI PREDICTION ENGINE ---
@st.cache_resource
def load_assets():
    try:
        with open("model.pkl", "rb") as f:
            m = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            s = pickle.load(f)
        return m, s
    except FileNotFoundError:
        st.error("Model files ('model.pkl', 'scaler.pkl') missing from GitHub.")
        return None, None

model, scaler = load_assets()

if model and scaler:
    st.subheader("Customer Data Input")
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 18, 90, 35)
        balance = st.number_input("Annual Balance (USD)", 0, 500000, 10000)
    with c2:
        duration = st.number_input("Last Interaction Duration (sec)", 0, 5000, 250)
        day = st.slider("Day of the Month", 1, 31, 15)

    if st.button("Run AI Assessment"):
        # Predict subscription eligibility
        data = np.array([[age, balance, day, duration]])
        scaled_data = scaler.transform(data)
        prediction = model.predict(scaled_data)
        
        if prediction[0] == 1:
            st.balloons()
            st.success("✅ Assessment: Customer is HIGHLY ELIGIBLE.")
        else:
            st.warning("⚠️ Assessment: Customer is NOT ELIGIBLE at this time.")
