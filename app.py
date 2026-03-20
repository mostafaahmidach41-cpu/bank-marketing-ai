import streamlit as st
import pickle
import numpy as np
import pandas as pd
import datetime
import os
from supabase import create_client, Client
import plotly.express as px
from fpdf import FPDF

# --- Page Configuration ---
st.set_page_config(page_title="VisionPro AI | Terminal", layout="wide", page_icon="🚀")

# --- Custom CSS for Modern UI ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 15px; border: 1px solid #eee; }
    .stButton>button { border-radius: 50px; font-weight: bold; transition: 0.3s; }
    .reasoning-box { background-color: white; border-left: 5px solid #2ecc71; padding: 15px; border-radius: 8px; color: #333; }
    div[data-testid="stSidebar"] { background-color: #2c3e50; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- Database Connection (Supabase) ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception:
    st.error("Connection Error: Please check Supabase credentials in Streamlit Secrets.")
    st.stop()

# --- Session State Management ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# --- Authentication Portal ---
if not st.session_state.authenticated:
    st.title("🛡️ Enterprise Security Portal")
    st.markdown("Please enter your professional license key to activate system access.")
    
    license_input = st.text_input("License Key", placeholder="PREMIUM-BANK-2026", type="password")
    
    if st.button("Activate System Now", use_container_width=True, type="primary"):
        try:
            res = supabase.table("licenses").select("key_value").eq("key_value", license_input).eq("is_active", True).execute()
            if res.data:
                st.session_state.authenticated = True
                st.session_state.license_key = license_input
                st.success("Authenticated successfully! Loading AI Engine...")
                st.rerun()
            else:
                st.error("Invalid or expired license key.")
        except Exception as e:
            st.error(f"Authentication Error: {e}")
    st.stop()

# --- AI Asset Loading ---
@st.cache_resource
def load_assets():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Asset Loading Error: {e}")
        return None, None

model, scaler = load_assets()

# --- AI Reasoning Logic ---
def get_decision_reasoning(age, balance, tenure, decision):
    reasons = []
    if decision == "ELIGIBLE":
        if balance > 150000: reasons.append("Strong liquidity detected supporting eligibility.")
        if tenure > 5: reasons.append("Established long-term relationship enhances trust score.")
        if 25 <= age <= 60: reasons.append("Client falls within optimal stable demographic range.")
    else:
        if balance < 100000: reasons.append("Current balance is below minimum premium threshold.")
        if tenure < 3: reasons.append("Relationship duration does not meet maturity requirements.")
        if age < 21: reasons.append("Age is below standard regulatory requirements.")
    
    return " | ".join(reasons) if reasons else "Profile meets standard evaluation parameters."

# --- Professional PDF Report Generation ---
def generate_pdf_report(result, license_key):
    pdf = FPDF()
    pdf.add_page()
    
    # Logo detection
    for logo in ["logo.png", "logo.png.png"]:
        if os.path.exists(logo):
            pdf.image(logo, 10, 8, 33)
            pdf.ln(20)
            break
    
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "AI Customer Assessment Report", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"License ID: {license_key}", ln=True)
    pdf.cell(0, 10, f"AI Decision: {result['decision']}", ln=True)
    pdf.cell(0, 10, f"Confidence Score: {result['confidence']:.2f}%", ln=True)
    pdf.ln(5)
    
    pdf.multi_cell(0, 10, f"AI Reasoning: {result.get('reasoning', 'Standard Assessment performed.')}")
    pdf.ln(10)
    pdf.cell(0, 10, f"Issued Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    
    return bytes(pdf.output(dest='S'))

# --- Sidebar (Dashboard) ---
with st.sidebar:
    st.title("📊 VisionPro Dashboard")
    st.write(f"Active User: **{st.session_state.license_key}**")
    st.divider()
    
    try:
        # Live Analytics from Supabase
        analytics = supabase.table("audit_logs").select("*").eq("license_key", st.session_state.license_key).execute()
        if analytics.data:
            df = pd.DataFrame(analytics.data)
            st.metric("Total Assessments", len(df))
            fig = px.pie(df, names="decision", hole=0.6, color="decision",
                         color_discrete_map={'ELIGIBLE':'#2ecc71', 'NOT ELIGIBLE':'#e74c3c'})
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Synchronizing cloud data...")

    if st.button("Logout System", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

# --- Main Assessment Engine ---
st.title("🚀 Customer AI Assessment Terminal")
st.markdown("Automated risk analysis and eligibility verification powered by VisionPro Neural Engine.")

if model and scaler:
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Customer Age", 18, 95, 40)
        with col2:
            balance = st.number_input("Yearly Balance ($)", 0, 1000000, 150000)
        with col3:
            tenure = st.number_input("Tenure (Years)", 0, 50, 5)

    if st.button("Run Smart Diagnostic", use_container_width=True, type="primary"):
        features = np.array([[age, balance, tenure]])
        scaled = scaler.transform(features)
        
        pred = model.predict(scaled)[0]
        conf = float(max(model.predict_proba(scaled)[0]) * 100)
        decision = "ELIGIBLE" if pred == 1 else "NOT ELIGIBLE"
        reasoning = get_decision_reasoning(age, balance, tenure, decision)
        
        st.session_state.last_result = {
            "age": age, "balance": balance, "tenure": tenure, 
            "decision": decision, "confidence": conf, "reasoning": reasoning
        }
        
        # Log to Supabase
        try:
            supabase.table("audit_logs").insert({
                "license_key": st.session_state.license_key,
                "customer_age": age,
                "balance": float(balance),
                "tenure": tenure,
                "decision": decision,
                "confidence": conf
            }).execute()
        except:
            pass
        st.rerun()

# --- Result Rendering ---
if st.session_state.last_result:
    res = st.session_state.last_result
    color = "#2ecc71" if res["decision"] == "ELIGIBLE" else "#e74c3c"
    
    st.divider()
    st.markdown(f"""
        <div style="border: 2px solid {color}; padding:30px; border-radius:15px; background:white; text-align:center;">
            <h1 style="color:{color}; margin:0;">{res['decision']}</h1>
            <p style="font-size:1.2em; color:#666;">Confidence Level: {res['confidence']:.2f}%</p>
            <div style="text-align:left; border-left:5px solid {color}; padding:15px; background:#f9f9f9; color:#333;">
                <strong>AI Analytical Reasoning:</strong><br>{res['reasoning']}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Download Report
    report = generate_pdf_report(res, st.session_state.license_key)
    st.download_button(
        label="📥 Download Official Assessment (PDF)",
        data=report,
        file_name=f"AI_Report_{res['decision']}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

st.divider()
st.caption("VisionPro AI © 2026 | Enterprise Security Protocol Enabled")
