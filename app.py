import streamlit as st
import pickle
import numpy as np
from supabase import create_client
from fpdf import FPDF

# --- 1. SUPABASE CONFIGURATION ---
# The verified working URL
URL = "https://ixwvplxnfndjbmdsvdpu.supabase.co"
KEY = "sb_publishable_666yE2Qkv09Y5NQ_QlQaEg_L8fneOgL"
supabase = create_client(URL, KEY)

# --- 2. AUTHENTICATION SYSTEM ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🛡️ Enterprise Security Portal")
    # Using strip() to ensure no accidental spaces break the check
    user_key = st.text_input("License Key", type="password").strip()
    
    if st.button("Activate"):
        try:
            # Checking against the licenses table
            res = supabase.table("licenses").select("*").eq("key_value", user_key).eq("is_active", True).execute()
            if res.data and len(res.data) > 0:
                st.session_state.authenticated = True
                st.success("System Activated Successfully!")
                st.rerun()
            else:
                st.error("Invalid or Expired License Key.")
        except Exception as e:
            st.error(f"Connection Error: {e}")
    st.stop()

# --- 3. PDF GENERATION ENGINE ---
def generate_pdf(age, balance, tenure, result, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="AI Eligibility Assessment Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Customer Age: {age}", ln=True)
    pdf.cell(200, 10, txt=f"Account Balance: ${balance:,}", ln=True)
    pdf.cell(200, 10, txt=f"Tenure: {tenure} Years", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=f"Decision: {result}", ln=True)
    pdf.cell(200, 10, txt=f"Model Confidence: {confidence:.2f}%", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# --- 4. MAIN ASSESSMENT TERMINAL ---
@st.cache_resource
def load_assets():
    try:
        with open("model.pkl", "rb") as f: model = pickle.load(f)
        with open("scaler.pkl", "rb") as f: scaler = pickle.load(f)
        return model, scaler
    except: return None, None

model, scaler = load_assets()

if model and scaler:
    st.title("Customer AI Assessment Terminal")
    st.sidebar.success("License Status: ACTIVE ✅")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Customer Age", 18, 95, 35)
        balance = st.number_input("Yearly Balance ($)", 0, 1000000, 250000)
    with col2:
        tenure = st.number_input("Relationship Tenure (Years)", 0, 50, 8)
        _ref_day = st.slider("Reference Day (Info Only)", 1, 31, 15)

    if st.button("Generate AI Decision"):
        try:
            # Fixed input for 3 features only to match model requirement
            input_data = np.array([[age, balance, tenure]])
            scaled_data = scaler.transform(input_data)
            
            prediction = model.predict(scaled_data)
            probs = model.predict_proba(scaled_data)[0]
            confidence = max(probs) * 100
            
            st.markdown("---")
            final_result = "ELIGIBLE" if prediction[0] == 1 else "NOT ELIGIBLE"
            
            if prediction[0] == 1:
                st.success(f"Result: {final_result} | Confidence: {confidence:.2f}%")
            else:
                st.warning(f"Result: {final_result} | Confidence: {confidence:.2f}%")
            
            st.progress(confidence / 100)

            # PDF Download Button
            pdf_bytes = generate_pdf(age, balance, tenure, final_result, confidence)
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=f"Report_{age}_{balance}.pdf",
                mime="application/pdf"
            )
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")
