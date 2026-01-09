import streamlit as st
import pickle
import numpy as np
from supabase import create_client, Client
from fpdf import FPDF

# --- 1. DATABASE CONNECTION ---
# Verified Project URL and Publishable Key from your Supabase settings
URL = "https://ixwvplxnfndjbmdsvdpu.supabase.co"
KEY = "sb_publishable_666yE2Qkv09Y5NQ_QlQaEg_L8fneOgL" 
supabase: Client = create_client(URL, KEY)

# --- 2. AUTHENTICATION LOGIC ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🛡️ Enterprise Security Portal")
    st.write("System activation required to access AI tools.")
    
    # User enters the name used during Stripe checkout
    user_key = st.text_input("Username / License Key", placeholder="Enter your registered name").strip()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Activate System"):
            try:
                # Searching for the license in your Supabase table
                res = supabase.table("licenses").select("*").eq("key_value", user_key).eq("is_active", True).execute()
                if res.data and len(res.data) > 0:
                    st.session_state.authenticated = True
                    st.success("Access Granted!")
                    st.rerun()
                else:
                    st.error("Invalid Username or Inactive License.")
            except Exception as e:
                st.error(f"Authentication Failure: {e}")
    
    with col2:
        # Link to your Stripe test payment page
        payment_url = "https://buy.stripe.com/test_8x29ATe0ubQ06NX1qyfUQ00"
        st.markdown(f'''<a href="{payment_url}" target="_blank">
            <button style="width:100%; height:42px; background-color:#6772E5; color:white; border:none; border-radius:4px; cursor:pointer; font-weight:bold;">
                Get License - $49.00
            </button></a>''', unsafe_allow_html=True)
    st.stop()

# --- 3. PDF REPORT GENERATOR ---
def create_assessment_report(age, balance, tenure, result, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Customer AI Assessment Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
    pdf.cell(200, 10, txt=f"Balance: ${balance:,}", ln=True)
    pdf.cell(200, 10, txt=f"Relationship Tenure: {tenure} Years", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=f"Final Decision: {result}", ln=True)
    pdf.cell(200, 10, txt=f"AI Confidence Score: {confidence:.2f}%", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# --- 4. ASSET LOADING & MAIN UI ---
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
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Customer Age", 18, 95, 35)
        balance = st.number_input("Yearly Balance ($)", 0, 1000000, 250000)
    with col2:
        tenure = st.number_input("Relationship Tenure (Years)", 0, 50, 8)
        st.info("Predicting based on Age, Balance, and Tenure.")

    if st.button("Generate AI Decision"):
        try:
            input_array = np.array([[age, balance, tenure]])
            scaled_input = scaler.transform(input_array)
            
            prediction = model.predict(scaled_input)
            probs = model.predict_proba(scaled_input)[0]
            confidence = max(probs) * 100
            
            st.markdown("---")
            assessment = "ELIGIBLE" if prediction[0] == 1 else "NOT ELIGIBLE"
            
            if prediction[0] == 1:
                st.success(f"Result: {assessment} | Confidence: {confidence:.2f}%")
            else:
                st.warning(f"Result: {assessment} | Confidence: {confidence:.2f}%")
            
            st.progress(confidence / 100)

            pdf_out = create_assessment_report(age, balance, tenure, assessment, confidence)
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_out,
                file_name=f"Assessment_{user_key}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Assessment Error: {e}")
else:
    st.error("System Error: Model or Scaler files not found.")
