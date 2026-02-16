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
    # Using Streamlit Secrets for secure connection
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception:
    st.error("Connection Error: Check Streamlit Secrets.")
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
    
    if st.button("Activate System", use_container_width=True):
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
        st.error(f"Asset Error: {e}")
        return None, None

model, scaler = load_assets()

# --- PDF Generation (Bytes-Safe Version) ---
def generate_pdf_report(result, license_key):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Customer AI Assessment Report", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"License ID: {license_key}", ln=True)
    pdf.cell(0, 10, f"AI Decision: {result['decision']}", ln=True)
    pdf.cell(0, 10, f"Confidence: {result['confidence']:.2f}%", ln=True)
    pdf.cell(0, 10, f"Assessment Date: {datetime.datetime.now().strftime('%Y-%m-%d')}", ln=True)
    
    # Ensuring byte conversion for Streamlit compatibility
    pdf_output = pdf.output()
    if isinstance(pdf_output, str):
        return pdf_output.encode('latin-1')
    return bytes(pdf_output)

# --- Enhanced Sidebar Analytics ---
with st.sidebar:
    st.success(f"Identity: {st.session_state.license_key}")
    st.divider()
    st.subheader("📊 Org Analytics")
    
    try:
        # Fetch data for visualization
        analytics_res = supabase.table("audit_logs").select("*").eq("license_key", st.session_state.license_key).execute()
        if analytics_res.data:
            df = pd.DataFrame(analytics_res.data)
            
            # Eligibility Distribution
            fig_pie = px.pie(df, names="decision", hole=0.4, 
                             color="decision", color_discrete_map={'ELIGIBLE':'#2ecc71', 'NOT ELIGIBLE':'#e74c3c'},
                             title="Eligibility Rate")
            fig_pie.update_layout(showlegend=False, margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Average Balance Comparison
            avg_bal = df.groupby("decision")["balance"].mean().reset_index()
            fig_bar = px.bar(avg_bal, x="decision", y="balance", color="decision",
                             color_discrete_map={'ELIGIBLE':'#2ecc71', 'NOT ELIGIBLE':'#e74c3c'},
                             title="Avg Balance ($)")
            fig_bar.update_layout(showlegend=False, margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.metric("Total Assessments", len(df))
    except:
        st.info("Analytics data loading...")

    st.divider()
    if st.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

# --- Main Assessment Engine ---
st.title("🚀 Customer AI Assessment Terminal")

if model and scaler:
    col_a, col_b = st.columns(2)
    with col_a:
        age = st.slider("Customer Age", 18, 95, 35)
        balance = st.number_input("Yearly Balance ($)", 0, 1_000_000, 50000)
    with col_b:
        tenure = st.number_input("Relationship Tenure (Years)", 0, 50, 10)

    if st.button("Execute Neural Analysis", use_container_width=True, type="primary"):
        # Prediction
        features = np.array([[age, balance, tenure]])
        scaled_features = scaler.transform(features)
        pred = model.predict(scaled_features)[0]
        conf = float(max(model.predict_proba(scaled_features)[0]) * 100)
        decision = "ELIGIBLE" if pred == 1 else "NOT ELIGIBLE"
        
        st.session_state.last_result = {"age": age, "balance": balance, "tenure": tenure, "decision": decision, "confidence": conf}
        
        # Log to Database
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
    st.divider()
    
    res_color = "#2ecc71" if res["decision"] == "ELIGIBLE" else "#e74c3c"
    st.markdown(f"""
        <div style="border: 2px solid {res_color}; padding:25px; border-radius:15px; text-align:center; background-color: rgba(0,0,0,0.05);">
            <h2 style="color:{res_color}; margin:0;">Result: {res['decision']}</h2>
            <h4 style="margin:5px 0;">Confidence Score: {res['confidence']:.2f}%</h4>
        </div>
    """, unsafe_allow_html=True)
    
    # Download Button logic
    report_data = generate_pdf_report(res, st.session_state.license_key)
    st.download_button(
        label="📥 Download Official Assessment (PDF)",
        data=report_data,
        file_name=f"AI_Assessment_{res['decision']}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

# --- Activity Log Table ---
st.divider()
st.subheader("📜 Recent Activity Log")
try:
    recent_logs = supabase.table("audit_logs").select("*").eq("license_key", st.session_state.license_key).order("created_at", desc=True).limit(5).execute()
    if recent_logs.data:
        # Columns must match database keys
        st.dataframe(pd.DataFrame(recent_logs.data)[["customer_age", "balance", "tenure", "decision", "confidence"]], use_container_width=True)
except:
    st.info("Activity log currently empty.")
