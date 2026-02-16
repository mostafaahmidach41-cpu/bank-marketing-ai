import streamlit as st
import pickle
import numpy as np
import pandas as pd
import datetime
from supabase import create_client, Client
import plotly.express as px
from fpdf import FPDF
from pathlib import Path

# --------------------------------------------------
# CONFIG & STABLE CONNECTION
# --------------------------------------------------
st.set_page_config(page_title="Enterprise AI Terminal", layout="wide")

# Improved connection logic to prevent "Invalid API key" errors
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error("Connection Error: Please check your Streamlit Secrets Configuration.")
    st.stop()

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "license_key" not in st.session_state:
    st.session_state.license_key = None

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# --------------------------------------------------
# AUTHENTICATION
# --------------------------------------------------
def authenticate_license(key):
    try:
        response = (
            supabase.table("licenses")
            .select("key_value")
            .eq("key_value", key)
            .eq("is_active", True)
            .execute()
        )
        return bool(response.data)
    except:
        return False

if not st.session_state.authenticated:
    st.title("🛡️ Enterprise Security Portal")
    license_input = st.text_input("Enter License Key", placeholder="PREMIUM-BANK-2026")

    if st.button("Activate System", use_container_width=True):
        if authenticate_license(license_input):
            st.session_state.authenticated = True
            st.session_state.license_key = license_input
            st.rerun()
        else:
            st.error("Invalid or inactive license key.")
    st.stop()

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None, None

model, scaler = load_model()

# --------------------------------------------------
# PDF GENERATOR (FIXED VERSION)
# --------------------------------------------------
def generate_pdf_report(result, license_key):
    # Fixed to resolve AttributeError and encoding issues
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 15, "Customer AI Assessment Report", ln=True, align="C")
    pdf.ln(5)
    
    # Line Decoration
    pdf.set_draw_color(0, 70, 140)
    pdf.set_line_width(0.8)
    pdf.line(10, 35, 200, 35)
    pdf.ln(10)

    # Content
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"License ID: {license_key}", ln=True)
    pdf.cell(0, 10, f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Assessment Details:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"- Age: {result['age']}", ln=True)
    pdf.cell(0, 10, f"- Balance: ${result['balance']:,.2f}", ln=True)
    pdf.cell(0, 10, f"- Tenure: {result['tenure']} Years", ln=True)
    pdf.ln(5)
    
    # Decision Highlight
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Final Decision: {result['decision']}", ln=True)
    pdf.cell(0, 10, f"AI Confidence Score: {result['confidence']:.2f}%", ln=True)

    # Footer
    pdf.set_y(-25)
    pdf.set_font("Arial", "I", 8)
    pdf.cell(0, 10, "Secure Enterprise AI Document - Confidential", align="C")

    # Returning as bytes to fix Streamlit download
    return pdf.output()

# --------------------------------------------------
# SIDEBAR ANALYTICS
# --------------------------------------------------
st.sidebar.success(f"License: {st.session_state.license_key}")

if st.sidebar.button("System Logout", use_container_width=True):
    st.session_state.authenticated = False
    st.rerun()

@st.cache_data(ttl=30)
def load_user_logs(license_key):
    try:
        response = (
            supabase.table("audit_logs")
            .select("*")
            .eq("license_key", license_key)
            .order("created_at", desc=True)
            .execute()
        )
        return pd.DataFrame(response.data) if response.data else pd.DataFrame()
    except:
        return pd.DataFrame()

logs_df = load_user_logs(st.session_state.license_key)

if not logs_df.empty:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Org Analytics")
    fig = px.pie(logs_df, names="decision", hole=0.4,
                 color="decision", color_discrete_map={'ELIGIBLE':'#2ecc71', 'NOT ELIGIBLE':'#e74c3c'})
    fig.update_layout(showlegend=False, height=200, margin=dict(t=0, b=0, l=0, r=0))
    st.sidebar.plotly_chart(fig, use_container_width=True)
    st.sidebar.metric("Processed Assessments", len(logs_df))

# --------------------------------------------------
# MAIN TERMINAL
# --------------------------------------------------
st.title("🚀 Customer AI Assessment Terminal")

if model and scaler:
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Customer Age", 18, 95, 35)
        balance = st.number_input("Yearly Balance ($)", 0, 1_000_000, 50000)
    with col2:
        tenure = st.number_input("Relationship Tenure (Years)", 0, 50, 10)

    if st.button("Execute Neural Analysis", use_container_width=True, type="primary"):
        # Prediction Engine
        features = np.array([[age, balance, tenure]])
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]
        confidence = float(max(model.predict_proba(scaled)[0]) * 100)
        decision = "ELIGIBLE" if prediction == 1 else "NOT ELIGIBLE"

        # State storage
        st.session_state.last_result = {
            "age": age, "balance": balance, "tenure": tenure,
            "decision": decision, "confidence": confidence
        }

        # Sync to Supabase
        try:
            supabase.table("audit_logs").insert({
                "license_key": st.session_state.license_key,
                "customer_age": age,
                "balance": float(balance),
                "tenure": tenure,
                "decision": decision,
                "confidence": confidence
            }).execute()
            st.rerun()
        except Exception as e:
            st.warning(f"Database Sync Pending: {e}")

    # Display Results
    if st.session_state.last_result:
        res = st.session_state.last_result
        st.markdown("---")
        
        c1, c2 = st.columns([1, 1])
        with c1:
            color = "#2ecc71" if res["decision"] == "ELIGIBLE" else "#e74c3c"
            st.markdown(f"""<div style="border: 2px solid {color}; padding:20px; border-radius:10px; text-align:center;">
                        <h2 style="color:{color}; margin:0;">{res['decision']}</h2>
                        <p style="margin:0;">Confidence: {res['confidence']:.2f}%</p></div>""", unsafe_allow_html=True)
        
        with c2:
            # Fixed PDF integration
            pdf_data = generate_pdf_report(res, st.session_state.license_key)
            st.download_button(
                label="📥 Download Official Report",
                data=pdf_data,
                file_name=f"Assessment_{res['decision']}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    # Activity Log
    st.markdown("---")
    st.subheader("📜 Recent Activity Log")
    if not logs_df.empty:
        st.dataframe(logs_df[["customer_age", "balance", "tenure", "decision", "confidence"]].head(10), use_container_width=True)
else:
    st.error("System assets missing. Please upload model.pkl and scaler.pkl.")
