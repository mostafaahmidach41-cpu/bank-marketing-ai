import streamlit as st
import pickle
import numpy as np
from supabase import create_client
from fpdf import FPDF

# --------------------------------------------------
# 1. Supabase Database Configuration
# --------------------------------------------------
SUPABASE_URL = "https://ixwvplxnfdjbmdsvdpu.supabase.co"
SUPABASE_KEY = "sb_publishable_666yE2Qkv09Y5NQ_QlQaEg_L8fneOgL"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --------------------------------------------------
# 2. License Verification Function
# --------------------------------------------------
def verify_license(user_key: str) -> bool:
    try:
        response = (
            supabase
            .table("licenses")
            .select("*")
            .eq("key_value", user_key.strip())
            .eq("is_active", True)
            .execute()
        )
        return len(response.data) > 0
    except Exception:
        return False

# --------------------------------------------------
# 3. Application Security Layer
# --------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.set_page_config(page_title="Bank AI - License Verification")

    st.title("Bank AI - License Verification")
    st.info("Please enter your premium license key to access the system.")

    input_key = st.text_input(
        "License Key",
        type="password",
        placeholder="PREMIUM-XXXX-2026"
    )

    if st.button("Activate System"):
        if verify_license(input_key):
            st.session_state.authenticated = True
            st.success("License verified successfully. Redirecting...")
            st.rerun()
        else:
            st.error("Invalid or expired license key.")

    st.stop()

# --------------------------------------------------
# 4. Main Application Interface
# --------------------------------------------------
st.set_page_config(page_title="Bank AI Terminal", layout="wide")

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
    st.sidebar.success("Status: Authorized")
    st.title("Customer Assessment with Confidence Analysis")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Customer Age", 18, 95, 35)
        balance = st.number_input(
            "Yearly Balance (USD)",
            min_value=0,
            max_value=1_000_000,
            value=2500
        )

    with col2:
        duration = st.number_input(
            "Relationship Tenure (Years)",
            min_value=0,
            max_value=50,
            value=5
        )
        day = st.slider("Reference Day", 1, 31, 15)

    if st.button("Run AI Assessment"):
        try:
            features = np.array([[age, balance, duration]])
            scaled_data = scaler.transform(features)

            prediction = model.predict(scaled_data)
            probabilities = model.predict_proba(scaled_data)[0]

            confidence = max(probabilities) * 100
            decision = "Eligible" if prediction[0] == 1 else "Not Eligible"

            st.markdown(f"### Decision: **{decision}**")
            st.write(f"Confidence Level: {confidence:.2f}%")
            st.progress(confidence / 100)

            st.markdown("---")
            if st.download_button(
                "Download PDF Report",
                data="Report Content",
                file_name="Report.pdf"
            ):
                st.info("Report generated successfully.")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
