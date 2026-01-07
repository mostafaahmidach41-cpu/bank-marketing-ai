import streamlit as st
import pickle
import numpy as np
from datetime import datetime
from fpdf import FPDF
from supabase import create_client

# --- 1. CLOUD CONNECTION SETUP ---
# Project URL from your Supabase settings
SUPABASE_URL = "https://ixwvplxnfdjbmdsvdpu.supabase.co"

# API Key confirmed from your dashboard
SUPABASE_KEY = "sb_publishable_666yE2Qkv09Y5NQ_QlQaEg_L8fneOgL"

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception:
    st.error("Database connection failed.")

# Function to verify the key from 'licenses' table
def verify_license_cloud(key_input):
    try:
        # Querying the table where RLS is now disabled
        response = supabase.table("licenses").select("*").eq("key_value", key_input.strip()).eq("is_active", True).execute()
        return len(response.data) > 0
    except Exception as e:
        return False

# --- 2. AUTHENTICATION LAYER ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.set_page_config(page_title="Bank AI Terminal")
    st.title("Bank AI - Enterprise Edition")
    
    st.info("Please enter your License Key to access the terminal.")
    
    # User inputs the key confirmed in database
    user_key = st.text_input("License Key", type="password")
    
    if st.button("Activate"):
        if verify_license_cloud(user_key):
            st.session_state.authenticated = True
            st.success("Access Granted")
            st.rerun()
        else:
            # Handles the error seen in previous attempts
            st.error("Invalid or Expired License Key.")
    st.stop()

# --- 3. MAIN DASHBOARD ---
st.set_page_config(page_title="Bank AI Dashboard", layout="wide")
st.title("Customer Assessment Terminal")

if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.rerun()

# --- 4. AI MODEL LOGIC ---
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("AI Model files are missing from the server.")
    st.stop()

# User Inputs for Assessment
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Customer Age", 18, 90, 35)
    balance = st.number_input("Account Balance (USD)", 0.0, 1000000.0, 50000.0)
    duration = st.slider("Interaction Duration (Days)", 1, 3600, 150)

if st.button("Run Prediction"):
    # Data transformation and prediction
    input_data = np.array([[age, balance, duration]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    
    result = "Eligible" if prediction[0] == 1 else "Not Eligible"
    
    if prediction[0] == 1:
        st.success(f"Final Decision: {result}")
    else:
        st.error(f"Final Decision: {result}")
