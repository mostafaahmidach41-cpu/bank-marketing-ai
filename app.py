import streamlit as st
import pickle
import numpy as np
from supabase import create_client
from fpdf import FPDF

# --- 1. SUPABASE CONNECTION ---
# Using project credentials from your verified cloud setup
URL = "https://ixwvplxnfdjbmdsvdpu.supabase.co"
KEY = "sb_publishable_666yE2Qkv09Y5NQ_QlQaEg_L8fneOgL"
supabase = create_client(URL, KEY)

# --- 2. AUTHENTICATION LOGIC ---
def verify_license(user_key):
    try:
        # Strict verification against your 'licenses' table
        response = supabase.table("licenses").select("*").eq("key_value", user_key.strip()).eq("is_active", True).execute()
        return len(response.data) > 0
    except Exception:
        return False

# Session State to prevent logout on every click
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# --- 3. LOGIN INTERFACE ---
if not st.session_state.authenticated:
    st.title("🛡️ Bank AI - License Verification")
    st.info("Please enter your premium license key.")
    
    # Matching your UI input for PREMIUM-BANK-2026
    license_input = st.text_input("License Key", type="password")
    
    if st.button("Activate System"):
        if verify_license(license_input):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid or Expired License Key.") #
    st.stop()

# --- 4. MAIN ASSESSMENT TERMINAL ---
st.set_page_config(page_title="Bank AI Terminal", layout="wide")

@st.cache_resource
def load_assets():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Asset Error: {e}")
        return None, None

model, scaler = load_assets()

if model and scaler:
    st.sidebar.success("Authorized ✅")
    st.title("Customer Assessment Terminal")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Customer Age", 18, 95, 35)
        balance = st.number_input("Yearly Balance ($)", 0, 1000000, 250000)
    with col2:
        # Tenure is the relationship duration in years
        duration = st.number_input("Relationship Tenure (Years)", 0, 50, 8) 
        day = st.slider("Reference Day", 1, 31, 15)

    if st.button("Run Assessment"):
        try:
            # FIX: Only sending 3 features as required by your scaler
            input_data = np.array([[age, balance, duration]])
            scaled_features = scaler.transform(input_data)
            
            # Prediction with confidence score
            prediction = model.predict(scaled_features)
            probs = model.predict_proba(scaled_features)[0]
            confidence = max(probs) * 100
            
            st.markdown("---")
            result = "Eligible" if prediction[0] == 1 else "Not Eligible"
            st.subheader(f"AI Decision: {result}")
            st.write(f"Confidence Level: {confidence:.2f}%")
            st.progress(confidence / 100) #
            
            if prediction[0] == 1:
                st.success(f"System is highly confident ({confidence:.2f}%) in success.")
            else:
                st.warning(f"System prediction indicates low success probability.")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")
