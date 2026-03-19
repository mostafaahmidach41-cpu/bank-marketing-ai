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
st.set_page_config(page_title="Enterprise AI Terminal", layout="wide", page_icon="🚀")

# --- Custom CSS for Modern UI ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .reasoning-box { padding: 15px; border-radius: 8px; margin-top: 10px; font-size: 0.9em; line-height: 1.4; }
    </style>
    """, unsafe_allow_html=True)

# --- Database Connection ---
try:
    # Using Streamlit Secrets for secure Supabase initialization
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception:
    st.error("Connection Error: Please check your Streamlit Secrets for Supabase credentials.")
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
    license_input = st.text_input("Enter License Key", placeholder="PREMIUM-BANK-2026")
    
    if st.button("Activate System", use_container_width=True, type="primary"):
        try:
            res = supabase.table("licenses").select("key_value").eq("key_value", license_input).eq("is_active", True).execute()
            if res.data:
                st.session_state.authenticated = True
                st.session_state.license_key = license_input
                st.rerun()
            else:
                st.error("Invalid or inactive license key.")
        except Exception as e:
            st.error(f"Authentication Error: {e}")
    st.stop()

# --- AI Asset Loading ---
@st.cache_resource
def load_assets():
    try:
        # These assets represent the core of your VisionPro AI model
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Asset Error: {e}")
        return None, None

model, scaler = load_assets()

# --- AI Reasoning Logic ---
def get_decision_reasoning(age, balance, tenure, decision):
    reasons = []
    if decision == "ELIGIBLE":
        if balance > 100000: reasons.append("✅ High capital liquidity detected.")
        if tenure > 5: reasons.append("✅ Strong long-term relationship history.")
        if 25 <= age <= 60: reasons.append("✅ Profile falls within prime demographic bracket.")
    else:
        if balance < 50000: reasons.append("❌ Insufficient yearly balance for premium tier.")
        if tenure < 2: reasons.append("❌ Relationship duration below minimum threshold.")
        if age < 21: reasons.append("❌ Client does not meet minimum age requirements.")
    
    return " ".join(reasons) if reasons else "Profile analyzed based on standard risk parameters."

# --- PDF Generation (Enhanced with Reasoning) ---
def generate_pdf_report(result, license_key):
    pdf = FPDF()
    pdf.add_page()
    
    possible_logos = ["logo.png (2).png", "logo.png.png", "logo.png"]
    for logo in possible_logos:
        if os.path.exists(logo):
            pdf.image(logo, 10, 8, 33)
            pdf.ln(20)
            break
    
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Customer AI Assessment Report", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"License ID: {license_key}", ln=True)
    pdf.cell(0, 10, f"AI Decision: {result['decision']}", ln=True)
    pdf.cell(0, 10, f"Confidence: {result['confidence']:.2f}%", ln=True)
    pdf.cell(0, 10, f"Reasoning: {result['reasoning']}", ln=True)
    pdf.cell(0, 10, f"Assessment Date: {datetime.datetime.now().strftime('%Y-%m-%d')}", ln=True)
    
    # Compatibility fix for Streamlit download
    return bytes(pdf.output(dest='S'))

# --- Sidebar Analytics Section ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=80)
    st.success(f"Verified: {st.session_state.license_key}")
    st.divider()
    st.subheader("📊 Performance Analytics")
    
    try:
        analytics_res = supabase.table("audit_logs").select("*").eq("license_key", st.session_state.license_key).execute()
        if analytics_res.data:
            df = pd.DataFrame(analytics_res.data)
            fig_pie = px.pie(df, names="decision", hole=0.5, 
                             color="decision", color_discrete_map={'ELIGIBLE':'#2ecc71', 'NOT ELIGIBLE':'#e74c3c'})
            fig_pie.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)
            st.metric("Total Assessments", len(df))
    except:
        st.info("Syncing with Supabase...")

    st.divider()
    if st.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

# --- Main Assessment Engine ---
st.title("🚀 Customer AI Assessment Terminal")
st.markdown("Enter client metrics below for real-time risk assessment and eligibility verification.")

if model and scaler:
    with st.container():
        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            age = st.slider("Customer Age", 18, 95, 49)
        with col_b:
            balance = st.number_input("Yearly Balance ($)", 0, 1000000, 250000)
        with col_c:
            tenure = st.number_input("Relationship Tenure (Years)", 0, 50, 9)

    if st.button("Generate AI Decision", use_container_width=True, type="primary"):
        features = np.array([[age, balance, tenure]])
        scaled_features = scaler.transform(features)
        pred = model.predict(scaled_features)[0]
        conf = float(max(model.predict_proba(scaled_features)[0]) * 100)
        decision = "ELIGIBLE" if pred == 1 else "NOT ELIGIBLE"
        
        # New Reasoning Engine Integration
        reasoning = get_decision_reasoning(age, balance, tenure, decision)
        
        st.session_state.last_result = {
            "age": age, "balance": balance, "tenure": tenure, 
            "decision": decision, "confidence": conf, "reasoning": reasoning
        }
        
        try:
            supabase.table("audit_logs").insert({
                "license_key": st.session_state.license_key,
                "customer_age": age,
                "balance": float(balance),
                "tenure": tenure,
                "decision": decision,
                "confidence": conf
            }).execute()
        except Exception as e:
            st.warning(f"Database logging failed: {e}")
        st.rerun()

# --- Result Rendering ---
if st.session_state.last_result:
    res = st.session_state.last_result
    st.divider()
    
    res_color = "#2ecc71" if res["decision"] == "ELIGIBLE" else "#e74c3c"
    bg_color = "rgba(46, 204, 113, 0.1)" if res["decision"] == "ELIGIBLE" else "rgba(231, 76, 60, 0.1)"
    
    st.markdown(f"""
        <div style="border: 2px solid {res_color}; background-color: {bg_color}; padding:25px; border-radius:12px; text-align:center;">
            <h1 style="color:{res_color}; margin:0;">{res['decision']}</h1>
            <h3 style="color:#555;">Confidence Level: {res['confidence']:.2f}%</h3>
            <div class="reasoning-box" style="background-color: white; border-left: 5px solid {res_color}; text-align: left; padding: 15px;">
                <strong>AI Reasoning:</strong><br>{res['reasoning']}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.ln(1)
    report_data = generate_pdf_report(res, st.session_state.license_key)
    st.download_button(
        label="📥 Download Official Assessment (PDF)",
        data=report_data,
        file_name=f"AI_Report_{res['decision']}_{datetime.date.today()}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

# --- Activity Log Table ---
st.divider()
st.subheader("📜 Recent Activity Log")
try:
    recent_logs = supabase.table("audit_logs").select("*").eq("license_key", st.session_state.license_key).order("created_at", desc=True).limit(5).execute()
    if recent_logs.data:
        log_df = pd.DataFrame(recent_logs.data)[["customer_age", "balance", "tenure", "decision", "confidence"]]
        st.table(log_df)
except:
    st.info("Ready for new assessments.")
