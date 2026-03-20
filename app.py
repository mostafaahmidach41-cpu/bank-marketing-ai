import streamlit as st
import joblib  # Using joblib for compressed models
import numpy as np
import pandas as pd
import datetime
import os
from supabase import create_client, Client
import plotly.express as px
from fpdf import FPDF

# --- Page Configuration ---
st.set_page_config(page_title="VisionPro AI | Enterprise Terminal", layout="wide", page_icon="🚀")

# --- Custom CSS for Enterprise UI ---
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
    st.error("Connection Error: Please verify Supabase credentials in Secrets.")
    st.stop()

# --- Session State Management ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "license_key" not in st.session_state:
    st.session_state.license_key = None

# --- Security Portal ---
if not st.session_state.authenticated:
    st.title("🛡️ Enterprise Security Portal")
    st.markdown("Enter your professional license key to activate the neural engine.")
    
    license_input = st.text_input("License Key", placeholder="ENTER-KEY-HERE", type="password")
    
    if st.button("Activate System", use_container_width=True, type="primary"):
        try:
            # Check license against Supabase
            res = supabase.table("licenses").select("key_value").eq("key_value", license_input).eq("is_active", True).execute()
            if res.data:
                st.session_state.authenticated = True
                st.session_state.license_key = license_input
                st.success("Access Granted. Synchronizing assets...")
                st.rerun()
            else:
                st.error("Invalid or expired license key.")
        except Exception as e:
            st.error(f"Auth Error: {e}")
    st.stop()

# --- AI Model Loading ---
@st.cache_resource
def load_assets():
    try:
        # Loading compressed files created with joblib in Colab
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Asset Error: {e}")
        return None, None

model, scaler = load_assets()

# --- AI Logic Engine ---
def get_decision_reasoning(age, balance, tenure, decision):
    reasons = []
    if decision == "ELIGIBLE":
        if balance > 150000: reasons.append("Strong capital liquidity detected.")
        if tenure > 5: reasons.append("Proven historical stability.")
        if 25 <= age <= 60: reasons.append("Client is in the optimal demographic range.")
    else:
        if balance < 100000: reasons.append("Capital below the safety threshold.")
        if tenure < 3: reasons.append("Relationship history is insufficient.")
        if age < 21: reasons.append("Regulatory age restrictions apply.")
    
    return " | ".join(reasons) if reasons else "Standard neural evaluation complete."

# --- PDF Report Generation ---
def generate_pdf_report(result, license_key):
    pdf = FPDF()
    pdf.add_page()
    
    # Auto-detecting logo
    for logo in ["logo.png", "logo.png.png"]:
        if os.path.exists(logo):
            pdf.image(logo, 10, 8, 33)
            pdf.ln(20)
            break
    
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "VisionPro AI Assessment Report", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"License ID: {license_key}", ln=True)
    pdf.cell(0, 10, f"Decision: {result['decision']}", ln=True)
    pdf.cell(0, 10, f"Confidence: {result['confidence']:.2f}%", ln=True)
    pdf.ln(5)
    
    pdf.multi_cell(0, 10, f"AI Reasoning: {result.get('reasoning', 'Evaluation based on 100K data points.')}")
    pdf.ln(10)
    pdf.cell(0, 10, f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    
    return bytes(pdf.output(dest='S'))

# --- Sidebar Dashboard ---
with st.sidebar:
    st.title("📊 Terminal Analytics")
    st.write(f"Logged as: **{st.session_state.license_key}**")
    st.divider()
    
    try:
        # Load activity for this specific license
        logs = supabase.table("audit_logs").select("*").eq("license_key", st.session_state.license_key).execute()
        if logs.data:
            df = pd.DataFrame(logs.data)
            st.metric("Total Usage", len(df))
            fig = px.pie(df, names="decision", hole=0.6, color="decision",
                         color_discrete_map={'ELIGIBLE':'#2ecc71', 'NOT ELIGIBLE':'#e74c3c'})
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Analytics engine syncing...")

    if st.button("Log out from Terminal", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

# --- Main Assessment UI ---
st.title("🚀 Customer Assessment Terminal")
st.markdown("Automated diagnostic powered by VisionPro V2 (100K Training Set).")

if model and scaler:
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Customer Age", 18, 95, 45)
        with col2:
            balance = st.number_input("Yearly Balance ($)", 0, 2000000, 250000)
        with col3:
            tenure = st.number_input("Tenure (Years)", 0, 50, 10)

    if st.button("Run Diagnostic", use_container_width=True, type="primary"):
        # Processing features
        features = np.array([[age, balance, tenure]])
        scaled = scaler.transform(features)
        
        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0]
        conf = float(max(prob) * 100)
        
        decision = "ELIGIBLE" if pred == 1 else "NOT ELIGIBLE"
        reasoning = get_decision_reasoning(age, balance, tenure, decision)
        
        st.session_state.last_result = {
            "decision": decision, "confidence": conf, "reasoning": reasoning
        }
        
        # Log to Supabase audit_logs
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
        <div style="border: 2px solid {color}; padding:25px; border-radius:15px; background:white; text-align:center;">
            <h1 style="color:{color}; margin:0;">{res['decision']}</h1>
            <p style="font-size:1.2em; color:#666;">Model Confidence: {res['confidence']:.2f}%</p>
            <div style="text-align:left; border-left:5px solid {color}; padding:15px; background:#f9f9f9; color:#333; margin-top:15px;">
                <strong>Analytical Reasoning:</strong><br>{res['reasoning']}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    report = generate_pdf_report(res, st.session_state.license_key)
    st.download_button(
        label="📥 Download Certified PDF Report",
        data=report,
        file_name=f"VisionPro_Report_{res['decision']}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

st.divider()
st.caption("VisionPro AI Enterprise Solutions © 2026 | Powered by Neural V2 Engine")
