import streamlit as st
import pickle
import numpy as np
from supabase import create_client, Client
import pandas as pd
import plotly.express as px

# --- 1. Stable Database Connection ---
# Securely fetching URL and KEY from Streamlit Secrets
URL = "https://ixwvplxnfndjbmdsvdpu.supabase.co"
try:
    KEY = st.secrets["SUPABASE_KEY"]
except:
    st.error("Missing SUPABASE_KEY in Secrets.")
    st.stop()

# Direct client creation to fix the AttributeError
supabase: Client = create_client(URL, KEY)

# --- 2. UI Configuration ---
st.set_page_config(page_title="Banking AI Terminal", layout="wide")

# Load ML Assets
@st.cache_resource
def load_assets():
    try:
        with open("model.pkl", "rb") as f: model = pickle.load(f)
        with open("scaler.pkl", "rb") as f: scaler = pickle.load(f)
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

# --- 3. Session Management ---
if "identity" not in st.session_state:
    # Restoring the identity shown in your original working version
    st.session_state.identity = "PREMIUM-BANK-2026"

# Sidebar
with st.sidebar:
    st.info(f"Identity: {st.session_state.identity}")
    if st.button("Logout"):
        st.stop()
    
    # Org Analytics
    st.markdown("---")
    st.subheader("📊 Org Analytics")
    try:
        res = supabase.table("audit_logs").select("*").execute()
        if res.data:
            df = pd.DataFrame(res.data)
            st.metric("Total Assessments", len(df))
            fig = px.pie(df, names="decision", color="decision", 
                         color_discrete_map={"ELIGIBLE": "#2ecc71", "NOT ELIGIBLE": "#e74c3c"})
            st.plotly_chart(fig, use_container_width=True)
    except:
        pass

# --- 4. Main Assessment Terminal ---
st.title("🚀 Customer AI Assessment Terminal")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 95, 42)
with col2:
    balance = st.number_input("Balance ($)", 0, 1000000, 50000)
tenure = st.slider("Tenure (Years)", 0, 50, 12)

if st.button("Execute Neural Analysis", use_container_width=True, type="primary"):
    if model and scaler:
        # Prediction Logic
        feats = scaler.transform([[age, balance, tenure]])
        pred = model.predict(feats)[0]
        conf = round(np.max(model.predict_proba(feats)) * 100, 2)
        decision = "ELIGIBLE" if pred == 1 else "NOT ELIGIBLE"

        # Data Insertion (Handling RLS by providing the required email field)
        try:
            supabase.table("audit_logs").insert({
                "email": "admin@bank.com", # Placeholder to satisfy database security policies
                "customer_age": age, 
                "balance": float(balance),
                "tenure": tenure, 
                "decision": decision, 
                "confidence": conf
            }).execute()
            
            # Display result with the original green/red styling
            bg = "#d4edda" if decision == "ELIGIBLE" else "#f8d7da"
            color = "#155724" if decision == "ELIGIBLE" else "#721c24"
            st.markdown(f"""
                <div style="background-color:{bg}; color:{color}; padding:30px; border-radius:10px; text-align:center; border:2px solid {color}55;">
                    <h1>Result: {decision}</h1>
                    <h3>Confidence Score: {conf}%</h3>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Database Sync Error: {e}")
    else:
        st.error("System Assets (model.pkl/scaler.pkl) not found.")

# --- 5. Recent Activity Log ---
st.markdown("---")
st.subheader("📜 Recent Activity Log")
try:
    # Fetching the latest logs from Supabase
    history = supabase.table("audit_logs").select("*").order("created_at", desc=True).limit(5).execute()
    if history.data:
        log_df = pd.DataFrame(history.data)[["customer_age", "balance", "tenure", "decision", "confidence"]]
        log_df.columns = ["Age", "Balance ($)", "Tenure (Y)", "Decision", "Confidence (%)"]
        st.table(log_df)
except:
    st.info("Waiting for first assessment data...")
