import streamlit as st
from supabase import create_client
import pickle
import numpy as np

URL = "https://ixwvplxnfdjbmdsvdpu.supabase.co"
KEY = "sb_publishable_666yE2Qkv09Y5NQ_QlQaEg_L8fneOgL"

try:
    supabase = create_client(URL, KEY)
except Exception:
    st.error("Database connection failed.")

def verify_license(user_input):
    try:
        response = (
            supabase.table("licenses")
            .select("*")
            .eq("key_value", user_input.strip())
            .eq("is_active", True)
            .execute()
        )
        return len(response.data) > 0
    except Exception:
        return False

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.set_page_config(page_title="Login - Bank AI", page_icon="🛡️")
    st.title("🛡️ Bank AI - Enterprise Edition")
    st.info("Access restricted. Please enter a valid license key.")

    user_key = st.text_input("License Key", type="password", placeholder="PREMIUM-XXXX-XXXX")

    if st.button("Activate via Cloud"):
        if verify_license(user_key):
            st.session_state.authenticated = True
            st.success("License verified successfully. Loading system...")
            st.rerun()
        else:
            st.error("Invalid or expired license key.")
    st.stop()

st.set_page_config(page_title="Dashboard - Bank AI", layout="wide")
st.title("📊 Smart Client Evaluation Terminal")

with st.sidebar:
    st.success("System Status: Connected")
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found (model.pkl, scaler.pkl).")
        return None, None

model, scaler = load_model()

if model and scaler:
    st.subheader("Client Input Data")
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 95, 30)
        balance = st.number_input("Annual Balance (USD)", 0, 500000, 2000)

    with col2:
        duration = st.number_input("Last Call Duration (seconds)", 0, 5000, 300)
        day = st.slider("Contact Day of Month", 1, 31, 15)

    if st.button("Run AI Evaluation"):
        features = np.array([[age, balance, day, duration]])
        prediction = model.predict(scaler.transform(features))

        if prediction[0] == 1:
            st.balloons()
            st.success("Client is eligible for the banking product.")
        else:
            st.warning("Client is not eligible at this time.")

