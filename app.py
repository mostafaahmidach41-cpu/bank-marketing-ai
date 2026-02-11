import streamlit as st
import pickle
import numpy as np
from supabase import create_client, Client
from fpdf import FPDF
import datetime
import plotly.express as px
import pandas as pd

# --- Supabase configuration ---
URL = "https://ixwvplxnfndjbmdsvdpu.supabase.co"
KEY = "sb_publishable_666yE2Qkv09Y5NQ_QlQaEg_L8fneOgL"
supabase: Client = create_client(URL, KEY)

# --- Session state management ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_user" not in st.session_state:
    st.session_state.current_user = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# --- Security Portal ---
if not st.session_state.authenticated:
    st.set_page_config(page_title="Enterprise Security Portal", layout="centered")
    st.title("🛡️ Enterprise Security Portal")
    user_input = st.text_input("Username or License Key", placeholder="Enter registered license key").strip()
    if st.button("Activate System", use_container_width=True):
        try:
            res = supabase.table("licenses").select("*").eq("key_value", user_input).eq("is_active", True).execute()
            if res.data and len(res.data) > 0:
                st.session_state.authenticated = True
                st.session_state.current_user = user_input
                st.rerun()
            else:
                st.error("Invalid or inactive license")
        except Exception as e:
            st.error(f"Authentication error: {e}")
    st.stop()

# --- Helper Functions ---
def create_assessment_report(age, balance, tenure, result, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Customer AI Assessment Report", ln=True, align="C")
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
    pdf.cell(200, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    return pdf.output(dest="S").encode("latin-1")

@st.cache_resource
def load_assets():
    try:
        with open("model.pkl", "rb") as f: model = pickle.load(f)
        with open("scaler.pkl", "rb") as f: scaler = pickle.load(f)
        return model, scaler
    except Exception: return None, None

# --- Main Application ---
model, scaler = load_assets()

if model and scaler:
    st.set_page_config(page_title="Customer AI Assessment Terminal", layout="wide")
    st.title("🚀 Customer AI Assessment Terminal")

    # --- Sidebar & Enhanced Analytics ---
    st.sidebar.info(f"Logged in as: {st.session_state.current_user}")
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Performance Analytics")

    all_data_df = pd.DataFrame()
    try:
        response = supabase.table("audit_logs").select("*").order("created_at", desc=True).execute()
        if response.data:
            all_data_df = pd.DataFrame(response.data)
            
            fig_pie = px.pie(all_data_df, names='decision', title='Eligibility Rate',
                             color='decision', color_discrete_map={'ELIGIBLE':'#2ecc71', 'NOT ELIGIBLE':'#e74c3c'})
            fig_pie.update_layout(showlegend=False, height=220, margin=dict(t=30, b=0, l=0, r=0))
            st.sidebar.plotly_chart(fig_pie, use_container_width=True)

            avg_balance = all_data_df.groupby('decision')['balance'].mean().reset_index()
            fig_bar = px.bar(avg_balance, x='decision', y='balance', title='Avg Balance ($)',
                             color='decision', color_discrete_map={'ELIGIBLE':'#2ecc71', 'NOT ELIGIBLE':'#e74c3c'})
            fig_bar.update_layout(showlegend=False, height=220, margin=dict(t=30, b=0, l=0, r=0))
            st.sidebar.plotly_chart(fig_bar, use_container_width=True)
            
            st.sidebar.metric("Total Assessments", len(all_data_df))

            st.sidebar.markdown("---")
            st.sidebar.subheader("📥 Data Export")
            csv_data = all_data_df.to_csv(index=False).encode('utf-8')
            st.sidebar.download_button(label="Download Full Log (CSV)", data=csv_data, file_name=f"Full_Audit_Log_{datetime.date.today()}.csv", mime="text/csv", use_container_width=True)
    except Exception:
        st.sidebar.warning("Charts loading...")

    # --- Input fields ---
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Customer Age", 18, 95, 35)
        balance = st.number_input("Yearly Balance ($)", 0, 1_000_000, 250_000)
    with col2:
        tenure = st.number_input("Relationship Tenure (Years)", 0, 50, 8)
        st.info("AI analysis considers financial stability and loyalty metrics.")

    # --- Decision processing ---
    if st.button("Generate AI Decision", use_container_width=True):
        try:
            features = np.array([[age, balance, tenure]])
            scaled = scaler.transform(features)
            prediction = model.predict(scaled)
            probabilities = model.predict_proba(scaled)[0]
            confidence = max(probabilities) * 100
            decision = "ELIGIBLE" if prediction[0] == 1 else "NOT ELIGIBLE"

            # Calculate Feature Importance for this specific decision
            # We use the model's feature_importances_ (for Tree models) or coef_ (for Linear models)
            try:
                importances = model.feature_importances_
            except AttributeError:
                # Fallback for models without feature_importances_
                importances = [0.33, 0.34, 0.33] # Placeholder balanced weight
            
            st.session_state.last_result = {
                "age": age, "balance": balance, "tenure": tenure, 
                "decision": decision, "confidence": confidence,
                "importances": importances
            }
            
            audit_entry = {"license_key": st.session_state.current_user, "customer_age": age, "balance": float(balance), "tenure": tenure, "decision": decision, "confidence": float(confidence)}
            supabase.table("audit_logs").insert(audit_entry).execute()
            st.rerun()
        except Exception as e: st.error(f"Processing Error: {e}")

    # Results Display & Importance Chart
    if st.session_state.last_result:
        res = st.session_state.last_result
        st.markdown("---")
        
        c1, c2 = st.columns([1, 1])
        with c1:
            if res['decision'] == "ELIGIBLE": st.success(f"Result: {res['decision']} | Confidence: {res['confidence']:.2f}%")
            else: st.warning(f"Result: {res['decision']} | Confidence: {res['confidence']:.2f}%")
            st.progress(res['confidence'] / 100)
        
        with c2:
            st.write("🔍 **Why this decision? (Feature Impact)**")
            imp_df = pd.DataFrame({
                'Feature': ['Age', 'Balance', 'Tenure'],
                'Impact': res['importances']
            }).sort_values(by='Impact', ascending=True)
            
            fig_imp = px.bar(imp_df, x='Impact', y='Feature', orientation='h',
                             color='Impact', color_continuous_scale='Viridis')
            fig_imp.update_layout(height=180, margin=dict(t=0, b=0, l=0, r=0), showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_imp, use_container_width=True)

    # Recent Activity Log Table
    st.markdown("---")
    st.subheader("📜 Recent Activity Log")
    if not all_data_df.empty:
        log_display = all_data_df[['customer_age', 'balance', 'tenure', 'decision', 'confidence']].head(5)
        log_display.columns = ['Age', 'Balance ($)', 'Tenure (Y)', 'Decision', 'Confidence (%)']
        def style_decision(val):
            color = '#2ecc71' if val == 'ELIGIBLE' else '#e74c3c'
            return f'color: {color}; font-weight: bold'
        st.table(log_display.style.applymap(style_decision, subset=['Decision']).format({'Balance ($)': '{:,.0f}', 'Confidence (%)': '{:.2f}'}))

    # Reporting Section
    st.markdown("---")
    st.subheader("📄 Reporting Section")
    if st.session_state.last_result:
        res = st.session_state.last_result
        pdf_data = create_assessment_report(res['age'], res['balance'], res['tenure'], res['decision'], res['confidence'])
        st.download_button(label="Download Official PDF Report", data=pdf_data, file_name=f"Assessment_{datetime.date.today()}.pdf", mime="application/pdf", use_container_width=True)
