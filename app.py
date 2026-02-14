import streamlit as st
import pickle
import numpy as np
from supabase import create_client, Client
from fpdf import FPDF
import pandas as pd
import plotly.express as px

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Secure AI Terminal",
    page_icon="🛡️",
    layout="wide"
)

# --- 2. Secure Credentials from Secrets ---
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

# --- 3. FIX: Secure Client with Proper Header Support ---
def get_user_client(license_key):
    """
    Creates a Supabase client that sends the license key in headers.
    This fixes the AttributeError and matches your RLS logic.
    """
    custom_headers = {
        "x-license-key": license_key,
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    # Using the standardized way to pass headers to avoid AttributeErrors
    return create_client(SUPABASE_URL, SUPABASE_KEY, options={"headers": custom_headers})

# --- 4. Session State & Auth ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_user" not in st.session_state:
    st.session_state.current_user = ""

if not st.session_state.authenticated:
    st.title("🔐 Enterprise Portal")
    license_input = st.text_input("Enter License Key", type="password")
    if st.button("Login"):
        # Simple verification check
        check_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        res = check_client.table("licenses").select("*").eq("key_value", license_input).eq("is_active", True).execute()
        if res.data:
            st.session_state.authenticated = True
            st.session_state.current_user = license_input
            st.rerun()
        else:
            st.error("Invalid License Key")
    st.stop()

# --- 5. Main Application Logic ---
# Initialize the secure user client
user_client = get_user_client(st.session_state.current_user)

st.title("🚀 Customer AI Assessment")

with st.sidebar:
    st.success("Premium Account Active")
    st.caption(f"User: {st.session_state.current_user[:10]}...")
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

# Input Fields
c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Customer Age", 18, 100, 30)
    balance = st.number_input("Balance (USD)", 0, 1000000, 5000)
with c2:
    tenure = st.slider("Tenure (Years)", 0, 50, 5)

if st.button("Run Neural Analysis", type="primary"):
    # Logic for Prediction (Placeholder for your model)
    decision = "ELIGIBLE" if balance > 10000 else "NOT ELIGIBLE"
    confidence = 85.5
    
    try:
        # Saving with the secure client
        # Important: Ensure you added 'INSERT' permission to your RLS Policy!
        user_client.table("audit_logs").insert({
            "license_key": st.session_state.current_user,
            "customer_age": age,
            "balance": float(balance),
            "tenure": tenure,
            "decision": decision,
            "confidence": confidence
        }).execute()
        st.success(f"Analysis Complete: {decision}")
    except Exception as e:
        st.error(f"Security Policy Block: {e}")

# --- 6. View Historical Data (Isolated) ---
st.markdown("---")
st.subheader("Your Recent Activity")
try:
    # RLS ensures only this user's data is returned
    history = user_client.table("audit_logs").select("*").order("created_at", desc=True).limit(5).execute()
    if history.data:
        st.table(pd.DataFrame(history.data)[['customer_age', 'balance', 'decision', 'confidence']])
    else:
        st.info("No records found for your account.")
except Exception as e:
    st.warning("Could not load activity log.")
