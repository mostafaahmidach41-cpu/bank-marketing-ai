import streamlit as st
import pickle
import numpy as np
from supabase import create_client, Client
from fpdf import FPDF
import datetime
import plotly.express as px
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Bank-Marketing AI | Enterprise",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Secure Supabase Configuration ---
# Credentials fetched from Streamlit Secrets
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

def get_secure_client(license_key):
    """Creates a client that passes the license key in headers for RLS"""
    headers = {"x-license-key": license_key}
    return create_client(SUPABASE_URL, SUPABASE_KEY, options={"headers": headers})

# --- Custom CSS ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stButton>button { border-radius: 5px; height: 3em; font-weight: bold; }
    .stTable { background-color: white; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- Session State ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_user" not in st.session_state:
    st.session_state.current_user = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# --- Auth Portal ---
if not st.session_state.authenticated:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100)
        st.title("Enterprise AI Gateway")
        st.subheader("Login to your Premium Account")
        license_input = st.text_input("License Key", type="password", placeholder="PREMIUM-XXXX-2026")
        
        if st.button("Authorize Access", use_container_width=True):
            try:
                # Initial check using standard client to verify key exists and is active
                base_client = create_client(SUPABASE_URL, SUPABASE_KEY)
                res = base_client.table("licenses").select("*").eq("key_value", license_input).eq("is_active", True).execute()
                if res.data:
                    st.session_state.authenticated = True
                    st.session_state.current_user = license_input
                    st.rerun()
                else:
                    st.error("Authentication failed. Invalid or inactive license.")
            except Exception:
                st.error("System connection error. Check your network or credentials.")
    st.stop()

# --- Load ML Assets ---
@st.cache_resource
def load_assets():
    try:
        with open("model.pkl", "rb") as f: model = pickle.load(f)
        with open("scaler.pkl", "rb") as f: scaler = pickle.load(f)
        return model, scaler
    except: return None, None

model, scaler = load_assets()

# --- Initialize Secure Client ---
# Every database call from here on includes the x-license-key header
user_client = get_secure_client(st.session_state.current_user)

# --- Sidebar UI ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=50)
    st.title("SaaS Control Panel")
    st.caption(f"Active Key: {st.session_state.current_user[:10]}...")
    st.markdown("---")
    
    try:
        # RLS automatically filters logs based on the header
        stats_res = user_client.table("audit_logs").select("decision").execute()
        if stats_res.data:
            df_stats = pd.DataFrame(stats_res.data)
            st.metric("Total Assessments", len(df_stats))
            approval_rate = (len(df_stats[df_stats['decision'] == 'ELIGIBLE']) / len(df_stats)) * 100
            st.metric("Approval Rate", f"{approval_rate:.1f}%")
    except: pass

    if st.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

# --- Main Dashboard ---
st.title("🚀 Customer AI Assessment Terminal")
st.markdown("Provide customer financial data to receive real-time eligibility scoring.")

with st.container():
    c1, c2, c3 = st.columns(3)
    with c1: age = st.number_input("Customer Age", 18, 95, 35)
    with c2: balance = st.number_input("Yearly Balance (USD)", 0, 1000000, 50000, step=1000)
    with c3: tenure = st.slider("Relationship Tenure (Years)", 0, 40, 5)

if st.button("Execute Neural Analysis", use_container_width=True, type="primary"):
    if model and scaler:
        try:
            feats = scaler.transform([[age, balance, tenure]])
            pred = model.predict(feats)[0]
            conf = max(model.predict_proba(feats)[0]) * 100
            decision = "ELIGIBLE" if pred == 1 else "NOT ELIGIBLE"
            
            st.session_state.last_result = {
                "age": age, "balance": balance, "tenure": tenure,
                "decision": decision, "confidence": conf
            }
            
            # This insert is now secured by your RLS INSERT policy
            user_client.table("audit_logs").insert({
                "license_key": st.session_state.current_user,
                "customer_age": age, "balance": float(balance),
                "tenure": tenure, "decision": decision, "confidence": float(conf)
            }).execute()
            st.rerun()
        except Exception as e:
            # Displays the RLS violation if the INSERT policy is missing
            st.error(f"Security/Analysis Error: {e}")

# --- Results ---
if st.session_state.last_result:
    res = st.session_state.last_result
    st.markdown("---")
    col_res, col_chart = st.columns([1, 1])
    
    with col_res:
        st.subheader("Analysis Result")
        color = "#2ecc71" if res['decision'] == "ELIGIBLE" else "#e74c3c"
        st.markdown(f'<div style="padding:20px; border-radius:10px; border-left: 10px solid {color}; background-color:white;"><h2 style="color:{color}; margin:0;">{res['decision']}</h2><p style="margin:0;">Confidence Score: <b>{res['confidence']:.2f}%</b></p></div>', unsafe_allow_html=True)
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "Official AI Assessment", ln=True, align='C')
        pdf.set_font("Arial", size=12)
        pdf.ln(10)
        pdf.cell(200, 10, f"Decision: {res['decision']}", ln=True)
        pdf.cell(200, 10, f"Confidence: {res['confidence']:.2f}%", ln=True)
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        st.download_button("Download PDF Report", pdf_bytes, "Assessment.pdf", "application/pdf")

    with col_chart:
        st.subheader("Risk Distribution")
        fig = px.bar(x=["Confidence", "Risk"], y=[res['confidence'], 100-res['confidence']], 
                     color=["Confidence", "Risk"], color_discrete_sequence=["#2ecc71", "#dfe6e9"])
        fig.update_layout(height=250, margin=dict(t=0, b=0, l=0, r=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# --- Activity Log (Secured) ---
st.markdown("---")
st.subheader("📜 Your Recent Activity")
try:
    # RLS ensures this query only returns data for the logged-in key
    log_res = user_client.table("audit_logs").select("*").order("created_at", desc=True).limit(5).execute()
    if log_res.data:
        df_logs = pd.DataFrame(log_res.data)[['customer_age', 'balance', 'decision', 'confidence']]
        df_logs.columns = ['Age', 'Balance ($)', 'Decision', 'Confidence (%)']
        
        def color_decision(val):
            color = '#27ae60' if val == 'ELIGIBLE' else '#c0392b'
            return f'color: {color}; font-weight: bold'

        st.table(df_logs.style.applymap(color_decision, subset=['Decision']).format({'Balance ($)': '{:,.0f}', 'Confidence (%)': '{:.2f}'}))
    else:
        st.info("No records found for this license key.")
except:
    st.info("Security layer active. No unauthorized logs visible.")
