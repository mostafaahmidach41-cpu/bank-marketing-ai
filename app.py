import streamlit as st
import pickle
import numpy as np
from supabase import create_client, Client
from fpdf import FPDF
import datetime
import plotly.express as px
import pandas as pd

# --- Configuration ---
# Your Supabase credentials
URL = "https://ixwvplxnfndjbmdsvdpu.supabase.co"
KEY = "sb_publishable_666yE2Qkv09Y5NQ_QlQaEg_L8fneOgL"
supabase: Client = create_client(URL, KEY)

# --- Session State ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_user" not in st.session_state:
    st.session_state.current_user = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# --- Security Portal (Premium Key Check) ---
if not st.session_state.authenticated:
    st.set_page_config(page_title="Enterprise AI Gateway", layout="centered")
    st.title("🛡️ Enterprise AI Gateway")
    st.markdown("---")
    st.info("Access Restricted. Enter your $29/mo Premium License Key.")
    
    user_input = st.text_input("License Key", type="password", placeholder="PREMIUM-BANK-XXXX").strip()
    
    if st.button("Activate Terminal", use_container_width=True):
        try:
            # Checking license table in Supabase
            res = supabase.table("licenses").select("*").eq("key_value", user_input).eq("is_active", True).execute()
            if res.data:
                st.session_state.authenticated = True
                st.session_state.current_user = user_input
                st.rerun()
            else:
                st.error("Authentication Failed: Invalid or Expired License.")
        except Exception as e:
            st.error(f"System Error: {e}")
    st.stop()

# --- Helper: PDF Report Generator ---
def create_pdf_report(age, balance, tenure, result, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Banking AI Assessment Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(200, 10, f"Customer Age: {age}", ln=True)
    pdf.cell(200, 10, f"Account Balance: ${balance:,}", ln=True)
    pdf.cell(200, 10, f"Relationship Tenure: {tenure} years", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, f"AI Decision: {result}", ln=True)
    pdf.cell(200, 10, f"Model Confidence: {confidence:.2f}%", ln=True)
    return pdf.output(dest="S").encode("latin-1")

@st.cache_resource
def load_ml_assets():
    try:
        with open("model.pkl", "rb") as f: model = pickle.load(f)
        with open("scaler.pkl", "rb") as f: scaler = pickle.load(f)
        return model, scaler
    except: return None, None

# --- Main SaaS Dashboard ---
model, scaler = load_ml_assets()

if model and scaler:
    st.set_page_config(page_title="Banking AI Terminal", layout="wide")
    st.title("🚀 Customer AI Assessment Terminal")

    # --- Sidebar ---
    st.sidebar.success("Subscription: PREMIUM ACTIVE")
    st.sidebar.info(f"User Key: {st.session_state.current_user}")
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

    # --- Input Fields ---
    st.subheader("Customer Parameters")
    col_a, col_b = st.columns(2)
    with col_a:
        age = st.slider("Customer Age", 18, 95, 35)
        balance = st.number_input("Yearly Balance ($)", 0, 1000000, 250000)
    with col_b:
        tenure = st.number_input("Relationship Tenure (Years)", 0, 50, 8)
        st.info("System uses Neural Analysis for risk assessment.")

    # --- AI Analysis Execution ---
    if st.button("Run AI Diagnostics", use_container_width=True):
        # Processing inputs through ML pipeline
        scaled_input = scaler.transform([[age, balance, tenure]])
        pred_class = model.predict(scaled_input)[0]
        prob_score = max(model.predict_proba(scaled_input)[0]) * 100
        decision_label = "ELIGIBLE" if pred_class == 1 else "NOT ELIGIBLE"
        
        st.session_state.last_result = {
            "age": age, "balance": balance, "tenure": tenure, 
            "decision": decision_label, "confidence": prob_score
        }
        
        # Logging entry to Supabase for audit
        try:
            supabase.table("audit_logs").insert({
                "license_key": st.session_state.current_user,
                "customer_age": age, 
                "balance": float(balance),
                "tenure": tenure, 
                "decision": decision_label, 
                "confidence": float(prob_score)
            }).execute()
        except Exception as e:
            st.warning(f"Logging error: {e}")
            
        st.rerun()

    # --- Results & Visual Feedback ---
    if st.session_state.last_result:
        res = st.session_state.last_result
        st.markdown("---")
        res_col, pdf_col = st.columns([2, 1])
        
        with res_col:
            color = "green" if res['decision'] == "ELIGIBLE" else "red"
            st.markdown(f"### Assessment: :{color}[{res['decision']}]")
            st.metric("Model Confidence", f"{res['confidence']:.2f}%")
            st.progress(res['confidence'] / 100)
            
        with pdf_col:
            st.write("📄 **Documentation**")
            pdf_data = create_pdf_report(res['age'], res['balance'], res['tenure'], res['decision'], res['confidence'])
            st.download_button("Download Official Report", pdf_data, "Report.pdf", "application/pdf", use_container_width=True)

    # --- Color-Coded Activity Log (Last 5 Entries) ---
    st.markdown("---")
    st.subheader("📜 Recent Activity Log")
    
    try:
        # Fetching records from audit_logs table
        log_res = supabase.table("audit_logs").select("*").order("created_at", desc=True).limit(5).execute()
        
        if log_res.data:
            df_logs = pd.DataFrame(log_res.data)[['customer_age', 'balance', 'tenure', 'decision', 'confidence']]
            df_logs.columns = ['Age', 'Balance ($)', 'Tenure (Y)', 'Decision', 'Confidence (%)']

            # CSS Color Mapping: Green for ELIGIBLE, Red for NOT ELIGIBLE
            def apply_status_color(val):
                color = '#2ecc71' if val == 'ELIGIBLE' else '#e74c3c'
                return f'color: {color}; font-weight: bold'

            # Displaying the color-coded table
            st.table(df_logs.style.applymap(apply_status_color, subset=['Decision']).format({
                'Balance ($)': '{:,.0f}', 
                'Confidence (%)': '{:.2f}'
            }))
        else:
            st.info("No audit data found in database.")
    except Exception as e:
        st.error(f"Failed to load logs: {e}")

else:
    st.error("Critical System Error: ML models (model.pkl / scaler.pkl) missing from repository.")
