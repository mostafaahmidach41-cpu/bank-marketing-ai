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
# Fetching from Streamlit Secrets for production safety
URL = "https://ixwvplxnfndjbmdsvdpu.supabase.co"
try:
    KEY = st.secrets["SUPABASE_KEY"]
except:
    st.error("Missing SUPABASE_KEY in Secrets.")
    st.stop()

supabase: Client = create_client(URL, KEY)

# --- SaaS Session Management ---
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# --- SaaS Authentication Portal ---
if not st.session_state.auth_user:
    st.set_page_config(page_title="SaaS Banking AI Login", layout="centered")
    st.title("🔐 Enterprise AI Portal")
    
    # Tabs for modern SaaS Login/Signup flow
    tab1, tab2 = st.tabs(["Sign In", "Register"])
    
    with tab1:
        email = st.text_input("Email", key="login_email")
        pw = st.text_input("Password", type="password", key="login_pw")
        if st.button("Access System", use_container_width=True):
            try:
                # Supabase Auth integration as configured in your dashboard
                res = supabase.auth.sign_in_with_password({"email": email, "password": pw})
                st.session_state.auth_user = res.user
                st.rerun()
            except Exception:
                st.error("Authentication failed. Check credentials or email verification.")
    
    with tab2:
        new_email = st.text_input("Corporate Email", key="reg_email")
        new_pw = st.text_input("Create Password", type="password", key="reg_pw")
        if st.button("Create SaaS Account", use_container_width=True):
            try:
                # Signup logic linked to your Auth Providers
                supabase.auth.sign_up({"email": new_email, "password": new_pw})
                st.success("Account created! Verify your email to continue.")
            except Exception as e:
                st.error(f"Registration failed: {e}")
    st.stop()

# --- Helper Functions ---
def create_assessment_report(age, balance, tenure, result, confidence, user_email):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Customer AI Assessment Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Account: {user_email}", ln=True)
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
    current_user_email = st.session_state.auth_user.email
    
    # --- Sidebar & Data Isolation ---
    st.sidebar.info(f"Active Session: {current_user_email}")
    if st.sidebar.button("Logout", use_container_width=True):
        supabase.auth.sign_out()
        st.session_state.auth_user = None
        st.rerun()

    # Query strictly isolated to the logged-in user's email
    response = supabase.table("audit_logs").select("*").eq("email", current_user_email).order("created_at", desc=True).execute()
    user_logs_df = pd.DataFrame(response.data) if response.data else pd.DataFrame()

    if not user_logs_df.empty:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Personal Analytics")
        fig_pie = px.pie(user_logs_df, names="decision", color="decision", 
                         color_discrete_map={"ELIGIBLE": "#2ecc71", "NOT ELIGIBLE": "#e74c3c"})
        fig_pie.update_layout(showlegend=False, height=200, margin=dict(t=30, b=0, l=0, r=0))
        st.sidebar.plotly_chart(fig_pie, use_container_width=True)
        st.sidebar.metric("Your Total Assessments", len(user_logs_df))

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

        try: importances = model.feature_importances_
        except: importances = [0.4, 0.4, 0.2]

        st.session_state.last_result = {"age": age, "balance": balance, "tenure": tenure, 
                                        "decision": decision, "confidence": conf, "importances": importances}

        # Logging with current email for multi-tenant isolation
        supabase.table("audit_logs").insert({
            "email": current_user_email, "customer_age": age, "balance": float(balance),
            "tenure": tenure, "decision": decision, "confidence": conf
        }).execute()
        st.rerun()

    # --- Display Results ---
    if st.session_state.last_result:
        res = st.session_state.last_result
        st.markdown("---")
        col_res, col_imp = st.columns([1, 1])
        
        with col_res:
            bg, txt, border = ("#d4edda", "#155724", "#c3e6cb") if res['decision'] == "ELIGIBLE" else ("#f8d7da", "#721c24", "#f5c6cb")
            icon = "✅" if res['decision'] == "ELIGIBLE" else "❌"
            st.markdown(f"""<div style="background-color: {bg}; color: {txt}; padding: 20px; border-radius: 10px; border: 2px solid {border}; text-align: center;">
                        <h2 style="margin: 0;">{icon} Result: {res['decision']}</h2>
                        <h4 style="margin: 5px 0 0 0;">Confidence Score: {res['confidence']}%</h4></div>""", unsafe_allow_html=True)
            st.progress(res["confidence"] / 100)

        with col_imp:
            st.write("🔍 **Why this decision? (Feature Impact)**")
            imp_df = pd.DataFrame({"Feature": ["Age", "Balance", "Tenure"], "Impact": res["importances"]}).sort_values(by="Impact")
            fig_imp = px.bar(imp_df, x="Impact", y="Feature", orientation="h", color="Impact", color_continuous_scale="Viridis")
            fig_imp.update_layout(height=180, margin=dict(t=0, b=0, l=0, r=0), coloraxis_showscale=False)
            st.plotly_chart(fig_imp, use_container_width=True)

    # --- Activity Log & PDF ---
    st.markdown("---")
    st.subheader("📜 Recent Activity Log")
    if not user_logs_df.empty:
        log_view = user_logs_df[["customer_age", "balance", "tenure", "decision", "confidence"]].head(5)
        log_view.columns = ["Age", "Balance ($)", "Tenure (Y)", "Decision", "Confidence (%)"]
        
        # Color coding for ELIGIBLE (green) and NOT ELIGIBLE (red)
        st.table(log_view.style.applymap(lambda x: 'color: #2ecc71; font-weight: bold' if x == 'ELIGIBLE' else 'color: #e74c3c; font-weight: bold', subset=['Decision']))

        if st.session_state.last_result:
            pdf_bytes = create_assessment_report(res["age"], res["balance"], res["tenure"], res["decision"], res["confidence"], current_user_email)
            st.download_button("Download Assessment PDF", pdf_bytes, f"Report_{current_user_email}.pdf", "application/pdf", use_container_width=True)

else:
    st.error("System Failure: AI Assets Missing.")
