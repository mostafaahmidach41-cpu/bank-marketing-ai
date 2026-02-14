import streamlit as st
import pickle
import numpy as np
from supabase import create_client, Client
from fpdf import FPDF
import datetime
import plotly.express as px
import pandas as pd

# --- Secure Configuration ---
# Direct fetching from Secrets to ensure stability
URL = "https://ixwvplxnfndjbmdsvdpu.supabase.co"
try:
    KEY = st.secrets["SUPABASE_KEY"]
except:
    st.error("Missing SUPABASE_KEY in Secrets.")
    st.stop()

# Reverted to stable direct connection to fix AttributeError
supabase: Client = create_client(URL, KEY)

# --- Session Management ---
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# --- Authentication Portal ---
if not st.session_state.auth_user:
    st.set_page_config(page_title="SaaS AI Login", layout="centered")
    st.title("🔐 Enterprise AI Portal")
    
    tab1, tab2 = st.tabs(["Sign In", "Register"])
    
    with tab1:
        email = st.text_input("Email")
        pw = st.text_input("Password", type="password")
        if st.button("Access System", use_container_width=True):
            try:
                # Direct Supabase Auth
                res = supabase.auth.sign_in_with_password({"email": email, "password": pw})
                st.session_state.auth_user = res.user
                st.rerun()
            except Exception:
                st.error("Authentication failed.")
    
    with tab2:
        new_email = st.text_input("Corporate Email")
        new_pw = st.text_input("Create Password", type="password")
        if st.button("Create Account", use_container_width=True):
            try:
                supabase.auth.sign_up({"email": new_email, "password": new_pw})
                st.success("Account created! Check your email.")
            except Exception as e:
                st.error(f"Failed: {e}")
    st.stop()

# --- Assets & UI ---
@st.cache_resource
def load_assets():
    try:
        with open("model.pkl", "rb") as f: model = pickle.load(f)
        with open("scaler.pkl", "rb") as f: scaler = pickle.load(f)
        return model, scaler
    except: return None, None

model, scaler = load_assets()

if model and scaler:
    st.set_page_config(page_title="Banking AI", layout="wide")
    current_email = st.session_state.auth_user.email
    
    # Sidebar Info
    st.sidebar.info(f"User: {current_email}")
    if st.sidebar.button("Logout"):
        supabase.auth.sign_out()
        st.session_state.auth_user = None
        st.rerun()

    # Manual Data Isolation (Safe & Proven)
    response = supabase.table("audit_logs").select("*").eq("email", current_email).order("created_at", desc=True).execute()
    user_logs = pd.DataFrame(response.data) if response.data else pd.DataFrame()

    # Main Terminal
    st.title("🚀 AI Assessment Terminal")
    c1, c2 = st.columns(2)
    with c1: age = st.slider("Customer Age", 18, 95, 35)
    with c2: balance = st.number_input("Yearly Balance ($)", 0, 1000000, 25000)
    tenure = st.number_input("Tenure (Years)", 0, 50, 5)

    if st.button("Execute AI Analysis", use_container_width=True, type="primary"):
        try:
            feats = scaler.transform([[age, balance, tenure]])
            pred = model.predict(feats)[0]
            conf = round(np.max(model.predict_proba(feats)) * 100, 2)
            decision = "ELIGIBLE" if pred == 1 else "NOT ELIGIBLE"

            # Log to DB
            supabase.table("audit_logs").insert({
                "email": current_email, "customer_age": age, "balance": float(balance),
                "tenure": tenure, "decision": decision, "confidence": conf
            }).execute()
            
            st.session_state.last_result = {"decision": decision, "confidence": conf}
            st.rerun()
        except Exception as e:
            st.error(f"System Error: {e}")

    # Display Last Result
    if st.session_state.last_result:
        res = st.session_state.last_result
        color = "#2ecc71" if res['decision'] == "ELIGIBLE" else "#e74c3c"
        st.markdown(f"""<div style="background-color: {color}22; border: 2px solid {color}; padding: 20px; border-radius: 10px; text-align: center;">
                    <h2 style="color: {color};">{res['decision']}</h2>
                    <h4>Confidence: {res['confidence']}%</h4></div>""", unsafe_allow_html=True)

    # Activity Log
    st.markdown("---")
    st.subheader("📜 Recent History")
    if not user_logs.empty:
        st.table(user_logs[["customer_age", "balance", "decision", "confidence"]].head(5))
else:
    st.error("System Failure: AI Assets Missing.")
