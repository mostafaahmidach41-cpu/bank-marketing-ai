import streamlit as st
import pickle
import numpy as np
from datetime import datetime
from fpdf import FPDF
from supabase import create_client

# --- 1. CLOUD CONNECTION SETUP ---
# Replace the URL with your actual Project URL from Supabase Settings
SUPABASE_URL = "https://ixwvplxnfdjbmdsvdpu.supabase.co" 
SUPABASE_KEY = "sb_publishable_666yE2Qkv09Y5NQ_QlQaEg_L8fneOgL"

# Initialize Supabase Client
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error("Connection to Cloud Database failed.")

# Function to verify key in Supabase
def verify_license_cloud(key_input):
    try:
        response = supabase.table("licenses").select("*").eq("key_value", key_input).eq("is_active", True).execute()
        return len(response.data) > 0
    except Exception:
        return False

# --- 2. SAAS SECURITY LAYER ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.set_page_config(page_title="Bank AI - Enterprise Portal")
    st.title("Bank AI - Enterprise Edition")
    st.info("Authorized Personnel Only: Please enter your License Key to access the terminal.")
    
    st.markdown("Don't have a license? [Click here to purchase via Stripe](https://buy.stripe.com/your_link)")
    
    user_key = st.text_input("Enter License Key", type="password")
    
    if st.button("Activate via Cloud"):
        if verify_license_cloud(user_key):
            st.session_state.authenticated = True
            st.success("Cloud Verification Successful!")
            st.rerun()
        else:
            st.error("Invalid or Expired License Key.")
    st.stop() 

# --- 3. MAIN APPLICATION CONFIGURATION ---
st.set_page_config(page_title="Bank AI Management System", layout="wide")

# --- 4. PDF REPORT GENERATION FUNCTION ---
def create_pdf(age, balance, duration, decision, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Customer Eligibility Assessment Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(200, 10, txt="--------------------------------------------------", ln=True)
    pdf.cell(200, 10, txt=f"Customer Age: {age} Years", ln=True)
    pdf.cell(200, 10, txt=f"Account Balance: ${balance:,.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Contact Duration: {duration} Days", ln=True)
    pdf.cell(200, 10, txt="--------------------------------------------------", ln=True)
    pdf.cell(200, 10, txt=f"Final Decision: {decision}", ln=True)
    pdf.cell(200, 10, txt=f"Model Confidence: {confidence}", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# --- 5. LOAD MODEL & SCALER ---
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Error: Critical model files (model.pkl/scaler.pkl) missing from server.")
    st.stop()

# --- 6. PROFESSIONAL DASHBOARD INTERFACE ---
st.title("Bank AI Customer Assessment Dashboard")
st.subheader("Enterprise-Grade Decision Support System")

if st.sidebar.button("Logout / Deactivate"):
    st.session_state.authenticated = False
    st.rerun()

st.markdown("---")
left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown("### Customer Profile Input")
    age = st.slider("Customer Age", 18, 80, 35)
    balance = st.number_input("Account Balance (USD)", 0.0, 500000.0, 25000.0, step=1000.0)
    duration = st.slider("Engagement Duration (Days)", 1, 3650, 180)
    assess_btn = st.button("Run AI Assessment")

with right_col:
    st.markdown("### Assessment Results")
    if assess_btn:
        input_data = np.array([[age, balance, duration]])
        scaled_data = scaler.transform(input_data)
        
        prob = model.predict_proba(scaled_data)[0][1]
        decision = "Eligible (YES)" if prob >= 0.5 else "Not Eligible (NO)"
        conf_score = round(prob, 4)
        
        if prob >= 0.5:
            st.success(f"**Outcome:** {decision}")
        else:
            st.error(f"**Outcome:** {decision}")
            
        st.write(f"**Confidence Score:** {conf_score}")
        
        risk_level = "Low Risk" if prob >= 0.7 else "Medium Risk" if prob >= 0.4 else "High Risk"
        st.info(f"**Assessed Risk Level:** {risk_level}")

        st.markdown("---")
        
        pdf_report = create_pdf(age, balance, duration, decision, conf_score)
        st.download_button(
            label="Download Official PDF Report",
            data=pdf_report,
            file_name=f"Bank_AI_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )
    else:
        st.info("System ready. Please enter data and run assessment.")

st.markdown("---")
st.caption("Secure Enterprise System | Powered by Scikit-Learn | Session Managed")
