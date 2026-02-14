import streamlit as st
import pickle
import numpy as np
from fpdf import FPDF
import datetime
import plotly.express as px
import pandas as pd
import os

# --- Configuration ---
# The logic now uses a hardcoded premium key for simplicity as requested
VALID_LICENSE_KEY = "PREMIUM-BANK-2026"

# --- Session State Management ---
if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False

# --- Simplified Premium Login UI ---
if not st.session_state.is_authenticated:
    st.set_page_config(page_title="SaaS Banking AI Login", layout="centered")
    st.title("🔐 Enterprise AI Portal")
    st.info("Please enter your $29/mo Premium Subscription Key to access the system.")
    
    license_input = st.text_input("License Key", type="password", placeholder="XXXX-XXXX-XXXX")
    
    if st.button("Activate Access", use_container_width=True):
        if license_input == VALID_LICENSE_KEY:
            st.session_state.is_authenticated = True
            st.success("Access Granted! Loading AI Models...")
            st.rerun()
        else:
            st.error("Invalid or Expired Key. Contact support for your $29 subscription.")
    st.stop()

# --- Helper Functions ---
def create_assessment_report(age, balance, tenure, result, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Premium AI Assessment Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Age: {age}", ln=True)
    pdf.cell(200, 10, f"Balance: ${balance:,}", ln=True)
    pdf.cell(200, 10, f"Tenure: {tenure} years", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, f"Decision: {result}", ln=True)
    pdf.cell(200, 10, f"Confidence: {confidence:.2f}%", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", "I", 8)
    pdf.cell(200, 10, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    return pdf.output(dest="S").encode("latin-1")

@st.cache_resource
def load_assets():
    try:
        with open("model.pkl", "rb") as f: model = pickle.load(f)
        with open("scaler.pkl", "rb") as f: scaler = pickle.load(f)
        return model, scaler
    except: return None, None

# --- Main SaaS Dashboard ---
model, scaler = load_assets()

if model and scaler:
    st.set_page_config(page_title="Banking AI Terminal", layout="wide")
    
    # Sidebar Logout
    if st.sidebar.button("Logout / Lock Terminal", use_container_width=True):
        st.session_state.is_authenticated = False
        st.rerun()

    st.sidebar.success("Subscription: PREMIUM ACTIVE")
    st.sidebar.markdown("---")
    st.sidebar.write("This terminal is licensed to your organization under the $29/mo plan.")

    # --- Assessment Input ---
    st.title("🚀 Customer AI Assessment Terminal")
    c1, c2 = st.columns(2)
    with c1: age = st.slider("Customer Age", 18, 95, 35)
    with c2: balance = st.number_input("Yearly Balance ($)", 0, 1000000, 25000)
    tenure = st.number_input("Relationship Tenure (Years)", 0, 50, 5)

    if st.button("Execute AI Analysis", use_container_width=True):
        feats = scaler.transform([[age, balance, tenure]])
        pred = model.predict(feats)[0]
        conf = round(np.max(model.predict_proba(feats)) * 100, 2)
        decision = "ELIGIBLE" if pred == 1 else "NOT ELIGIBLE"

        st.markdown("---")
        col_res, col_pdf = st.columns([2, 1])
        
        with col_res:
            icon = "✅" if decision == "ELIGIBLE" else "❌"
            st.subheader(f"{icon} Final Decision: {decision}")
            st.metric("AI Confidence Score", f"{conf}%")
            st.progress(conf / 100)

        with col_pdf:
            st.write("📜 **Documentation**")
            pdf_bytes = create_assessment_report(age, balance, tenure, decision, conf)
            st.download_button("Download Official PDF Report", pdf_bytes, "Assessment_Report.pdf", "application/pdf", use_container_width=True)

else:
    st.error("System Failure: ML Assets (model.pkl / scaler.pkl) not found in the repository.")
