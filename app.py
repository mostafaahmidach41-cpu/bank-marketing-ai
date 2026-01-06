import streamlit as st
import pickle
import numpy as np
from datetime import datetime
from fpdf import FPDF
from supabase import create_client

# --- 1. CLOUD CONNECTION SETUP ---
# Connection details from your Supabase Project Settings
SUPABASE_URL = "https://ixwvplxnfdjbmdsvdpu.supabase.co" 
SUPABASE_KEY = "sb_publishable_666yE2Qkv09Y5NQ_QlQaEg_L8fneOgL"

# Initialize Supabase Client
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception:
    st.error("Cloud Database Connection Failed.")

# Function to verify the License Key in your 'licenses' table
def verify_license_cloud(key_input):
    try:
        # Check if key exists and is_active is true
        response = supabase.table("licenses").select("*").eq("key_value", key_input).eq("is_active", True).execute()
        return len(response.data) > 0
    except Exception:
        return False

# --- 2. SAAS SECURITY LAYER (GATEWAY) ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.set_page_config(page_title="Bank AI - Enterprise Edition")
    st.title("Bank AI - Enterprise Edition")
    st.info("Authorized Personnel Only: Please enter your License Key to access the terminal.")
    
    # Stripe Payment Link (Replace 'your_link' with the link from your Stripe Dashboard)
    st.markdown("Don't have a license? [Click here to purchase via Stripe](https://buy.stripe.com/your_link)")
    
    user_key = st.text_input("Enter License Key", type="password")
    
    if st.button("Activate via Cloud"):
        # Test with key: PREMIUM-BANK-2026
        if verify_license_cloud(user_key):
            st.session_state.authenticated = True
            st.success("Verification Successful!")
            st.rerun()
        else:
            st.error("Invalid or Expired License Key.")
    st.stop() 

# --- 3. MAIN APP CONFIGURATION (POST-LOGIN) ---
st.set_page_config(page_title="Bank AI Management System", layout="wide")

# --- 4. PDF REPORT GENERATION ---
def create_pdf(age, balance, duration, decision, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Customer Eligibility Assessment Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(200, 10, txt="--------------------------------------------------", ln=True)
    pdf.cell(200, 10, txt=f"Customer Age: {age}", ln=True)
    pdf.cell(200, 10, txt=f"Account Balance: ${balance:,.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Contact Duration: {duration} Days", ln=True)
    pdf.cell(200, 10, txt="--------------------------------------------------", ln=True)
    pdf.cell(200, 10, txt=f"Final Decision: {decision}", ln=True)
    pdf.cell(200, 10, txt=f"Model Confidence: {confidence}", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# --- 5. LOAD AI MODELS ---
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Critical model files missing.")
    st.stop()

# --- 6. USER INTERFACE ---
st.title("Bank AI Customer Assessment Dashboard")

if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.rerun()

left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown("### Input Profile")
    age = st.slider("Age", 18, 80, 35)
    balance = st.number_input("Balance (USD)", 0.0, 500000.0, 25000.0)
    duration = st.slider("Duration", 1, 3650, 180)
    assess_btn = st.button("Run Assessment")

with right_col:
    st.markdown("### Results")
    if assess_btn:
        input_data = np.array([[age, balance, duration]])
        scaled_data = scaler.transform(input_data)
        
        prob = model.predict_proba(scaled_data)[0][1]
        decision = "Eligible" if prob >= 0.5 else "Not Eligible"
        
        if prob >= 0.5:
            st.success(f"Outcome: {decision}")
        else:
            st.error(f"Outcome: {decision}")
            
        pdf_report = create_pdf(age, balance, duration, decision, prob)
        st.download_button("Download PDF", pdf_report, "Report.pdf", "application/pdf")
