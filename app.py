import streamlit as st
import pickle
import numpy as np
import pandas as pd
import datetime
from supabase import create_client, Client
import plotly.express as px
from fpdf import FPDF

# --- Page Configuration ---
st.set_page_config(page_title="Enterprise AI Terminal", layout="wide")

# --- Database Connection ---
try:
    # Securely connecting via Streamlit Secrets
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception:
    st.error("Connection Error: Please check Streamlit Secrets configuration.")
    st.stop()

# --- Session State Management ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "license_key" not in st.session_state:
    st.session_state.license_key = None

# --- Authentication Portal ---
if not st.session_state.authenticated:
    st.title("🛡️ Enterprise Security Portal")
    license_input = st.text_input("Enter License Key", placeholder="e.g., PREMIUM-BANK-2026")
    
    if st.button("Activate System", use_container_width=True):
        # Validating license key against database
        try:
            res = supabase.table("licenses").select("key_value").eq("key_value", license_input).eq("is_active", True).execute()
            if res.data:
                st.session_state.authenticated = True
                st.session_state.license_key = license_input
                st.rerun()
            else:
                st.error("Invalid or inactive license key.")
        except Exception as e:
            st.error(f"Auth Error: {e}")
    st.stop()

# --- Asset Loading ---
@st.cache_resource
def load_assets():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Failed to load AI assets: {e}")
        return None, None

model, scaler = load_assets()

# --- PDF Generation Logic (Fixed for stability) ---
def generate_pdf_report(result, license_key):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Customer AI Assessment Report", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"License ID: {license_key}", ln=True)
    pdf.cell(0, 10, f"Decision Result: {result['decision']}", ln=True)
    pdf.cell(0, 10, f"Confidence Score: {result['confidence']:.2f}%", ln=True)
    pdf.cell(0, 10, f"Generation Date: {datetime.datetime.now().strftime('%Y-%m-%d')}", ln=True)
    
    # Converting output to Bytes to prevent Streamlit download errors
    pdf_output = pdf.output()
    if isinstance(pdf_output, str):
        return pdf_output.encode('latin-1')
    return bytes(pdf_output)

# --- Sidebar Analytics ---
with st.sidebar:
    st.success(f"System Active: {st.session_state.license_key}")
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

# --- Main Assessment Interface ---
st.title("🚀 Customer AI Assessment Terminal")

if model and scaler:
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Customer Age", 18, 95, 35)
        balance = st.number_input("Yearly Balance ($)", 0, 1_000_000, 50000)
    with col2:
        tenure = st.number_input("Relationship Tenure (Years)", 0, 50, 10)

    if st.button("Execute Neural Analysis", use_container_width=True, type="primary"):
        # AI Processing
        features = np.array([[age, balance, tenure]])
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]
        confidence = float(max(model.predict_proba(scaled)[0]) * 100)
        decision = "ELIGIBLE" if prediction == 1 else "NOT ELIGIBLE"
        
        st.session_state.last_result = {
            "age": age, 
            "balance": balance, 
            "tenure": tenure, 
            "decision": decision, 
            "confidence": confidence
        }
        
        # Logging to Database
        try:
            supabase.table("audit_logs").insert({
                "license_key": st.session_state.license_key,
                "customer_age": age,
                "balance": float(balance),
                "tenure": tenure,
                "decision": decision,
                "confidence": confidence
            }).execute()
        except Exception as e:
            st.warning(f"Note: Could not sync log entry ({e})")
        st.rerun()

# --- Results Display ---
if st.session_state.last_result:
    res = st.session_state.last_result
    st.divider()
    
    color = "#2ecc71" if res["decision"] == "ELIGIBLE" else "#e74c3c"
    st.markdown(f"""
        <div style="border: 2px solid {color}; padding:20px; border-radius:10px; text-align:center;">
            <h2 style="color:{color}; margin:0;">{res['decision']}</h2>
            <h4 style="margin:0;">Confidence Level: {res['confidence']:.2f}%</h4>
        </div>
    """, unsafe_allow_html=True)
    
    # Download Report
    pdf_bytes = generate_pdf_report(res, st.session_state.license_key)
    st.download_button(
        label="📥 Download Official Assessment (PDF)",
        data=pdf_bytes,
        file_name=f"Assessment_Report_{res['decision']}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

# --- Activity Log ---
st.divider()
st.subheader("📜 Recent Activity Log")
try:
    logs = supabase.table("audit_logs").select("*").eq("license_key", st.session_state.license_key).order("created_at", desc=True).limit(5).execute()
    if logs.data:
        log_df = pd.DataFrame(logs.data)
        # Displaying specific columns found in your successful build
        st.dataframe(log_df[["customer_age", "balance", "tenure", "decision", "confidence"]], use_container_width=True)
    else:
        st.info("No recent assessment logs found.")
except Exception:
    st.info("Log visualization unavailable.")
