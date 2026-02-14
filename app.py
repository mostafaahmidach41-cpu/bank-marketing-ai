import streamlit as st
import pickle
import numpy as np
from supabase import create_client, Client
import pandas as pd
import plotly.express as px

# --- 1. Core Configuration ---
# Restoring stable connection and fetching KEY from secrets
URL = "https://ixwvplxnfndjbmdsvdpu.supabase.co"
try:
    KEY = st.secrets["SUPABASE_KEY"]
except:
    st.error("Check Streamlit Secrets for SUPABASE_KEY.")
    st.stop()

# Fixed connection logic to resolve AttributeError
supabase: Client = create_client(URL, KEY)

# --- 2. Model Loading ---
@st.cache_resource
def load_assets():
    try:
        with open("model.pkl", "rb") as f: model = pickle.load(f)
        with open("scaler.pkl", "rb") as f: scaler = pickle.load(f)
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

# --- 3. App Layout & Session ---
st.set_page_config(page_title="Banking AI Terminal", layout="wide")

if "identity" not in st.session_state:
    # Restoring original Premium Identity
    st.session_state.identity = "PREMIUM-BANK-2026"

with st.sidebar:
    st.info(f"Identity: {st.session_state.identity}")
    if st.button("Logout"):
        st.stop()
    
    st.markdown("---")
    st.subheader("📊 Org Analytics")
    try:
        # Fetching all logs to build the pie chart
        res = supabase.table("audit_logs").select("*").execute()
        if res.data:
            df = pd.DataFrame(res.data)
            st.metric("Total Assessments", len(df))
            fig = px.pie(df, names="decision", color="decision", 
                         color_discrete_map={"ELIGIBLE": "#2ecc71", "NOT ELIGIBLE": "#e74c3c"})
            st.plotly_chart(fig, use_container_width=True)
    except:
        pass

# --- 4. Main Interface ---
st.title("🚀 Customer AI Assessment Terminal")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 95, 51)
with col2:
    balance = st.number_input("Balance ($)", 0, 1000000, 50000)
tenure = st.slider("Tenure (Years)", 0, 50, 16)

if st.button("Execute Neural Analysis", use_container_width=True, type="primary"):
    if model and scaler:
        # Run AI prediction
        feats = scaler.transform([[age, balance, tenure]])
        pred = model.predict(feats)[0]
        conf = round(np.max(model.predict_proba(feats)) * 100, 2)
        decision = "ELIGIBLE" if pred == 1 else "NOT ELIGIBLE"

        # FIX: Removed the 'email' field to match your current schema
        try:
            supabase.table("audit_logs").insert({
                "customer_age": age, 
                "balance": float(balance),
                "tenure": tenure, 
                "decision": decision, 
                "confidence": conf
            }).execute()
            
            # Restoring high-visibility result box
            bg = "#d4edda" if decision == "ELIGIBLE" else "#f8d7da"
            color = "#155724" if decision == "ELIGIBLE" else "#721c24"
            st.markdown(f"""
                <div style="background-color:{bg}; color:{color}; padding:30px; border-radius:10px; text-align:center; border:2px solid {color}55;">
                    <h1>Result: {decision}</h1>
                    <h3>Confidence Score: {conf}%</h3>
                </div>
                """, unsafe_allow_html=True)
            st.rerun()
        except Exception as e:
            st.error(f"Database Sync Error: {e}")
    else:
        st.error("AI Assets Missing.")

# --- 5. Activity Log ---
st.markdown("---")
st.subheader("📜 Recent Activity Log")
try:
    history = supabase.table("audit_logs").select("*").order("created_at", desc=True).limit(5).execute()
    if history.data:
        log_df = pd.DataFrame(history.data)[["customer_age", "balance", "tenure", "decision", "confidence"]]
        log_df.columns = ["Age", "Balance ($)", "Tenure (Y)", "Decision", "Confidence (%)"]
        st.table(log_df)
except:
    st.info("No assessment history found.")
