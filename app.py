import streamlit as st
import pickle
import numpy as np
from supabase import create_client, Client
from fpdf import FPDF
import datetime
import plotly.express as px
import pandas as pd
import os

# --- Secure Configuration ---
# Use Streamlit secrets or environment variables for production
URL = "https://ixwvplxnfndjbmdsvdpu.supabase.co"
KEY = "sb_publishable_666yE2Qkv09Y5NQ_QlQaEg_L8fneOgL"

supabase: Client = create_client(URL, KEY)

# --- Session State Management ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_user" not in st.session_state:
    st.session_state.current_user = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# --- Access Control Portal ---
if not st.session_state.authenticated:
    st.set_page_config(page_title="Enterprise AI Gateway", layout="centered")
    st.title("🛡️ Enterprise AI Gateway")
    st.markdown("### Secure Access Required")
    
    user_key = st.text_input("License Key", placeholder="Enter your 29$ subscription key").strip()
    
    if st.button("Authorize Access", use_container_width=True):
        try:
            # Validate license key against Supabase database
            res = supabase.table("licenses").select("*").eq("key_value", user_key).eq("is_active", True).execute()
            if res.data and len(res.data) > 0:
                st.session_state.authenticated = True
                st.session_state.current_user = user_key
                st.rerun()
            else:
                st.error("Access Denied: Invalid or expired license key.")
        except Exception as e:
            st.error(f"System Error: {e}")
    st.stop()

# --- Utility Functions ---
def generate_pdf_report(age, balance, tenure, result, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Banking AI Assessment Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Account Holder ID: {st.session_state.current_user}", ln=True)
    pdf.cell(200, 10, f"Customer Age: {age}", ln=True)
    pdf.cell(200, 10, f"Account Balance: ${balance:,}", ln=True)
    pdf.cell(200, 10, f"Tenure: {tenure} years", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, f"Classification: {result}", ln=True)
    pdf.cell(200, 10, f"Confidence Score: {confidence:.2f}%", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", "I", 8)
    pdf.cell(200, 10, f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    return pdf.output(dest="S").encode("latin-1")

@st.cache_resource
def load_ml_assets():
    try:
        with open("model.pkl", "rb") as f: model = pickle.load(f)
        with open("scaler.pkl", "rb") as f: scaler = pickle.load(f)
        return model, scaler
    except Exception:
        return None, None

# --- Primary Application Interface ---
model, scaler = load_ml_assets()

if model and scaler:
    st.set_page_config(page_title="AI Assessment Terminal", layout="wide")
    st.title("🚀 Banking AI Assessment Terminal")

    # --- Sidebar & Analytics Dashboard ---
    st.sidebar.info(f"Session Active: {st.session_state.current_user}")
    if st.sidebar.button("Terminate Session", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Fleet Analytics")

    # Fetch global logs for analytical display
    try:
        log_res = supabase.table("audit_logs").select("*").order("created_at", desc=True).execute()
        if log_res.data:
            logs_df = pd.DataFrame(log_res.data)
            
            # Pie Chart: Decision Distribution
            fig_pie = px.pie(logs_df, names='decision', title='Eligibility Distribution',
                             color='decision', color_discrete_map={'ELIGIBLE':'#2ecc71', 'NOT ELIGIBLE':'#e74c3c'})
            fig_pie.update_layout(showlegend=False, height=200, margin=dict(t=30, b=0, l=0, r=0))
            st.sidebar.plotly_chart(fig_pie, use_container_width=True)

            # Metric Display
            st.sidebar.metric("Total Assessments Processed", len(logs_df))
            
            # Export Functionality
            st.sidebar.markdown("---")
            csv = logs_df.to_csv(index=False).encode('utf-8')
            st.sidebar.download_button("Export Data (CSV)", csv, "Audit_Log.csv", "text/csv", use_container_width=True)
    except Exception:
        st.sidebar.warning("Analytics temporarily unavailable.")

    # --- Customer Input Module ---
    st.subheader("Customer Parameters")
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 18, 95, 35)
        balance = st.number_input("Yearly Balance (USD)", 0, 1_000_000, 250_000)
    with c2:
        tenure = st.number_input("Tenure (Years)", 0, 50, 8)
        st.info("AI Model: Gradient Boosted Classification v2.4")

    # --- Analysis Execution ---
    if st.button("Run AI Diagnostics", use_container_width=True):
        try:
            input_data = np.array([[age, balance, tenure]])
            input_scaled = scaler.transform(input_data)
            pred = model.predict(input_scaled)
            probs = model.predict_proba(input_scaled)[0]
            conf = max(probs) * 100
            decision = "ELIGIBLE" if pred[0] == 1 else "NOT ELIGIBLE"

            # Feature Importance (Logic extracted from model)
            try: importances = model.feature_importances_
            except AttributeError: importances = [0.33, 0.34, 0.33]
            
            st.session_state.last_result = {
                "age": age, "balance": balance, "tenure": tenure, 
                "decision": decision, "confidence": conf, "importances": importances
            }
            
            # Record entry in Supabase audit log
            audit_data = {
                "license_key": st.session_state.current_user, 
                "customer_age": age, 
                "balance": float(balance), 
                "tenure": tenure, 
                "decision": decision, 
                "confidence": float(conf)
            }
            supabase.table("audit_logs").insert(audit_data).execute()
            st.rerun()
        except Exception as e:
            st.error(f"Analysis Failed: {e}")

    # --- Results & Insight Visualization ---
    if st.session_state.last_result:
        res = st.session_state.last_result
        st.markdown("---")
        
        r1, r2 = st.columns([1, 1])
        with r1:
            st.markdown(f"### Result: **{res['decision']}**")
            st.write(f"Confidence Level: {res['confidence']:.2f}%")
            st.progress(res['confidence'] / 100)
        
        with r2:
            st.write("🔍 **Decision Drivers**")
            imp_df = pd.DataFrame({'Feature': ['Age', 'Balance', 'Tenure'], 'Impact': res['importances']}).sort_values(by='Impact')
            fig_imp = px.bar(imp_df, x='Impact', y='Feature', orientation='h', color='Impact', color_continuous_scale='Viridis')
            fig_imp.update_layout(height=180, margin=dict(t=0, b=0, l=0, r=0), showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_imp, use_container_width=True)

        # Download PDF Report
        pdf_file = generate_pdf_report(res['age'], res['balance'], res['tenure'], res['decision'], res['confidence'])
        st.download_button("💾 Download Official PDF Report", pdf_file, f"Assessment_{datetime.date.today()}.pdf", "application/pdf", use_container_width=True)

else:
    st.error("Critical Failure: Machine Learning assets not found.")
