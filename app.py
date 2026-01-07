import streamlit as st
import pickle
import numpy as np
from fpdf import FPDF
import base64

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Bank AI - Report Edition", page_icon="📊", layout="wide")

# --- 2. PDF GENERATION FUNCTION ---
def create_pdf(age, balance, duration, result):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Customer Assessment Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Customer Age: {age}", ln=True)
    pdf.cell(200, 10, f"Account Balance: ${balance}", ln=True)
    pdf.cell(200, 10, f"Relationship Duration: {duration} Years", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, f"AI Assessment Result: {result}", ln=True)
    
    return pdf.output(dest='S').encode('latin-1')

# --- 3. AI ASSET LOADING ---
@st.cache_resource
def load_assets():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("Error: Model files not found.")
        return None, None

model, scaler = load_assets()

# --- 4. PREDICTION INTERFACE ---
if model and scaler:
    st.sidebar.success("System Status: Online")
    st.sidebar.info("License: DISABLED (Open Access)")
    
    st.title("Customer Assessment & Reporting")
    st.write("### Enter Customer Profile")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Customer Age", 18, 95, 35)
        balance = st.number_input("Yearly Balance ($)", 0, 500000, 2500)
    with col2:
        duration = st.number_input("Relationship Duration (Years)", 0, 50, 5)
        day = st.slider("Reference Day", 1, 31, 15)

    if st.button("Run AI Assessment"):
        try:
            # Preparing 3 features as required by the scaler
            features = np.array([[age, balance, duration]])
            scaled_features = scaler.transform(features)
            prediction = model.predict(scaled_features)
            
            res_text = "ELIGIBLE" if prediction[0] == 1 else "NOT ELIGIBLE"
            
            if prediction[0] == 1:
                st.balloons()
                st.success(f"✅ Result: {res_text}")
            else:
                st.warning(f"❌ Result: {res_text}")
            
            # PDF Generation Section
            st.markdown("---")
            st.subheader("Download Assessment Report")
            pdf_data = create_pdf(age, balance, duration, res_text)
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_data,
                file_name=f"Customer_Report_{age}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Error: {e}")

# --- 5. FOOTER ---
st.sidebar.markdown("---")
st.sidebar.write("Bank AI v2.2 - Reporting Edition")
