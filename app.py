import streamlit as st
import pickle
import numpy as np
from datetime import datetime
from fpdf import FPDF

# --- Page Configuration ---
st.set_page_config(page_title="Bank AI Management System", layout="wide")

# --- PDF Report Generation Function ---
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

# --- Simple Login System ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.header("Bank Portal Login")
    email = st.text_input("Email Address")
    password = st.text_input("Password", type="password")
    if st.button("Sign In"):
        # Simple validation for testing purposes
        if "@" in email and len(password) >= 6:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid email format or password (min 6 characters).")
    st.stop()

# --- Load Model & Scaler (After Login) ---
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Error: Model files (model.pkl or scaler.pkl) not found.")
    st.stop()

# --- Main Application Interface ---
st.title("Bank Customer Assessment Dashboard")
st.subheader("Decision Support System for Marketing Eligibility")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

st.markdown("---")

left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown("### Customer Profile")
    age = st.slider("Customer Age", 18, 80, 35)
    balance = st.number_input("Account Balance (USD)", 0.0, 500000.0, 25000.0, step=1000.0)
    duration = st.slider("Engagement Duration (Days)", 1, 3650, 180)
    
    assess_btn = st.button("Run AI Assessment")

with right_col:
    st.markdown("### Decision Results")
    if assess_btn:
        # Prepare data for prediction
        input_data = np.array([[age, balance, duration]])
        scaled_data = scaler.transform(input_data)
        
        # Prediction and Probability
        prob = model.predict_proba(scaled_data)[0][1]
        decision = "Eligible (YES)" if prob >= 0.5 else "Not Eligible (NO)"
        conf_score = round(prob, 4)
        
        # Styling the result based on decision
        if prob >= 0.5:
            st.success(f"**Outcome:** {decision}")
        else:
            st.error(f"**Outcome:** {decision}")
            
        st.write(f"**Confidence Score:** {conf_score}")
        
        # Risk Categorization logic
        risk_level = "Low Risk" if prob >= 0.7 else "Medium Risk" if prob >= 0.4 else "High Risk"
        st.info(f"**Assessed Risk Level:** {risk_level}")

        st.markdown("---")
        
        # --- PDF Download Button ---
        pdf_report = create_pdf(age, balance, duration, decision, conf_score)
        st.download_button(
            label="Download Detailed PDF Report",
            data=pdf_report,
            file_name=f"Assessment_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf"
        )
    else:
        st.info("Awaiting input: Please enter customer data and click assess.")

st.markdown("---")
st.caption("Secure System | Powered by Scikit-Learn | Authorized Personnel Only")
