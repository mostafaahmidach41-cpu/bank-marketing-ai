import streamlit as st
import pickle
import numpy as np
from supabase import create_client
from fpdf import FPDF

# --- 1. DATABASE CONFIGURATION (SUPABASE) ---
# Project credentials for license management
URL = "https://ixwvplxnfdjbmdsvdpu.supabase.co"
KEY = "sb_publishable_666yE2Qkv09Y5NQ_QlQaEg_L8fneOgL"
supabase = create_client(URL, KEY)

# --- 2. LICENSE VERIFICATION FUNCTION ---
def verify_license(user_key):
    try:
        # Check if key exists and is active
        query = supabase.table("licenses").select("*").eq("key_value", user_key.strip()).eq("is_active", True).execute()
        return len(query.data) > 0
    except Exception:
        return False

# --- 3. SESSION MANAGEMENT ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Login Interface
if not st.session_state.authenticated:
    st.title("🛡️ Bank AI - License Verification")
    st.info("Please enter your premium license key to access the terminal.")
    
    license_input = st.text_input("License Key", type="password", placeholder="PREMIUM-BANK-2026")
    
    if st.button("Activate System"):
        if verify_license(license_input):
            st.session_state.authenticated = True
            st.success("Access Granted! Loading System...")
            st.rerun()
        else:
            st.error("Invalid or Expired License Key.") #
    st.stop()

# --- 4. PREDICTION TERMINAL (POST-VERIFICATION) ---
st.set_page_config(page_title="Bank AI Terminal", layout="wide")

@st.cache_resource
def load_assets():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

if model and scaler:
    st.sidebar.success("Status: Authorized ✅")
    st.title("Customer Eligibility Assessment")
    
    # Input Interface
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Customer Age", 18, 95, 35)
        balance = st.number_input("Average Yearly Balance ($)", 0, 1000000, 250000)
    with col2:
        # Tenure reflects years of relationship
        duration = st.number_input("Relationship Tenure (Years)", 0, 50, 8) 
        day = st.slider("Reference Day of Month", 1, 31, 24)

    if st.button("Run AI Prediction"):
        try:
            # Using 3 features to avoid StandardScaler mismatch
            features = np.array([[age, balance, duration]])
            scaled_data = scaler.transform(features)
            
            # Predict outcome and confidence
            prediction = model.predict(scaled_data)
            probabilities = model.predict_proba(scaled_data)[0]
            confidence = max(probabilities) * 100
            
            st.markdown("---")
            res_label = "Eligible" if prediction[0] == 1 else "Not Eligible"
            st.subheader(f"Decision: {res_label}")
            
            # Display confidence metrics
            st.write(f"AI Confidence Level: {confidence:.2f}%")
            st.progress(confidence / 100)
            
            if prediction[0] == 1:
                st.balloons()
                st.success(f"System is {confidence:.2f}% confident the customer will subscribe.")
            else:
                st.warning(f"System is {confidence:.2f}% confident the customer is not a fit.")

            # PDF Download Option
            st.download_button("📥 Download Summary PDF", "Report Data Content", "Bank_Report.pdf")
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# --- 5. FOOTER ---
st.sidebar.markdown("---")
st.sidebar.write("Bank AI v2.6 - Premium Edition")
