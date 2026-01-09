import streamlit as st
import pickle
import numpy as np
from supabase import create_client, Client
from fpdf import FPDF

URL = "https://ixwvplxnfndjbmdsvdpu.supabase.co"
KEY = "sb_publishable_666yE2Qkv09Y5NQ_QlQaEg_L8fneOgL"
supabase: Client = create_client(URL, KEY)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_user" not in st.session_state:
    st.session_state.current_user = ""

if not st.session_state.authenticated:
    st.set_page_config(page_title="Enterprise Security Portal", layout="centered")
    st.title("Enterprise Security Portal")

    user_input = st.text_input(
        "Username or License Key",
        placeholder="Enter registered license key"
    ).strip()

    if st.button("Activate System"):
        try:
            res = (
                supabase
                .table("licenses")
                .select("*")
                .eq("key_value", user_input)
                .eq("is_active", True)
                .execute()
            )
            if res.data and len(res.data) > 0:
                st.session_state.authenticated = True
                st.session_state.current_user = user_input
                st.success("Access granted")
                st.rerun()
            else:
                st.error("Invalid or inactive license")
        except Exception as e:
            st.error(f"Authentication error: {e}")
    st.stop()

def create_assessment_report(age, balance, tenure, result, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Customer AI Assessment Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Age: {age}", ln=True)
    pdf.cell(200, 10, f"Balance: ${balance:,}", ln=True)
    pdf.cell(200, 10, f"Tenure: {tenure} years", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, f"Decision: {result}", ln=True)
    pdf.cell(200, 10, f"Confidence: {confidence:.2f}%", ln=True)

    return pdf.output(dest="S").encode("latin-1")

@st.cache_resource
def load_assets():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception:
        return None, None

model, scaler = load_assets()

if model and scaler:
    st.set_page_config(page_title="Customer AI Assessment Terminal", layout="wide")
    st.title("Customer AI Assessment Terminal")

    st.sidebar.success(f"License: {st.session_state.current_user}")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Customer Age", 18, 95, 35)
        balance = st.number_input("Yearly Balance ($)", 0, 1_000_000, 250_000)
    with col2:
        tenure = st.number_input("Relationship Tenure (Years)", 0, 50, 8)
        st.info("Prediction uses age, balance, and tenure")

    if st.button("Generate AI Decision"):
        try:
            features = np.array([[age, balance, tenure]])
            scaled = scaler.transform(features)

            prediction = model.predict(scaled)
            probabilities = model.predict_proba(scaled)[0]
            confidence = max(probabilities) * 100

            decision = "ELIGIBLE" if prediction[0] == 1 else "NOT ELIGIBLE"

            st.markdown("---")
            if prediction[0] == 1:
                st.success(f"Result: {decision} | Confidence: {confidence:.2f}%")
            else:
                st.warning(f"Result: {decision} | Confidence: {confidence:.2f}%")

            st.progress(confidence / 100)

            pdf_data = create_assessment_report(
                age,
                balance,
                tenure,
                decision,
                confidence
            )

            st.download_button(
                label="Download PDF Report",
                data=pdf_data,
                file_name=f"Report_{st.session_state.current_user}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Assessment error: {e}")
