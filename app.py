import streamlit as st
import pickle
import numpy as np
from supabase import create_client, Client
from fpdf import FPDF
import datetime

# --- Supabase configuration ---
URL = "https://ixwvplxnfndjbmdsvdpu.supabase.co"
KEY = "sb_publishable_666yE2Qkv09Y5NQ_QlQaEg_L8fneOgL"
supabase: Client = create_client(URL, KEY)

# --- Session state management ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_user" not in st.session_state:
    st.session_state.current_user = ""
# Initialize last assessment results to keep the button functional
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# --- Security Portal ---
if not st.session_state.authenticated:
    st.set_page_config(page_title="Enterprise Security Portal", layout="centered")
    st.title("🛡️ Enterprise Security Portal")

    user_input = st.text_input(
        "Username or License Key",
        placeholder="Enter registered license key"
    ).strip()

    if st.button("Activate System", use_container_width=True):
        try:
            res = (
                supabase
                .table("licenses")
                .select("*")
                .eq("key_value", user_input)
                .eq("is_active", True)
                .execute()
            )

            if res.data and len(res.data) > 0:
                st.session_state.authenticated = True
                st.session_state.current_user = user_input
                st.success("Access granted")
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
    pdf.cell(
        200,
        10,
        f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ln=True
    )

    return pdf.output(dest="S").encode("latin-1")


@st.cache_resource
def load_assets():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception:
        return None, None


# --- Main Application ---
model, scaler = load_assets()

if model and scaler:
    st.set_page_config(page_title="Customer AI Assessment Terminal", layout="wide")
    st.title("🚀 Customer AI Assessment Terminal")

    # --- Sidebar & Analytics Dashboard ---
    st.sidebar.info(f"Logged in as: {st.session_state.current_user}")
    
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Live Analytics Dashboard")

    try:
        stats_res = supabase.table("audit_logs").select("id", count="exact").execute()
        total_checks = stats_res.count if stats_res.count else 0
        
        eligible_res = supabase.table("audit_logs").select("id", count="exact").eq("decision", "ELIGIBLE").execute()
        eligible_checks = eligible_res.count if eligible_res.count else 0

        st.sidebar.metric("Total Assessments", total_checks)
        st.sidebar.metric("Total Eligible", eligible_checks)
        
        if total_checks > 0:
            rate = (eligible_checks / total_checks) * 100
            st.sidebar.write(f"Acceptance Rate: {rate:.1f}%")
            st.sidebar.progress(rate / 100)
    except Exception:
        st.sidebar.warning("Analytics temporarily unavailable")

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

            # Save result to session state for PDF generation
            st.session_state.last_result = {
                "age": age,
                "balance": balance,
                "tenure": tenure,
                "decision": decision,
                "confidence": confidence
            }

            audit_entry = {
                "license_key": st.session_state.current_user,
                "customer_age": age,
                "balance": float(balance),
                "tenure": tenure,
                "decision": decision,
                "confidence": float(confidence)
            }
            supabase.table("audit_logs").insert(audit_entry).execute()

            st.markdown("---")
            if prediction[0] == 1:
                st.success(f"Result: {decision} | Confidence: {confidence:.2f}%")
            else:
                st.warning(f"Result: {decision} | Confidence: {confidence:.2f}%")

            st.progress(confidence / 100)
            
            st.rerun()

        except Exception as e:
            st.error(f"System Error: {e}")

    # --- Permanent PDF Download Section ---
    st.markdown("---")
    st.subheader("Reporting Section")
    
    if st.session_state.last_result:
        # Generate data from the last assessment stored in session state
        res = st.session_state.last_result
        pdf_data = create_assessment_report(res['age'], res['balance'], res['tenure'], res['decision'], res['confidence'])
        
        st.download_button(
            label="Download Official PDF Report",
            data=pdf_data,
            file_name=f"Assessment_{st.session_state.current_user}_{datetime.date.today()}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    else:
        # Show a disabled-looking button or info if no assessment has been done yet
        st.info("Please generate an AI Decision first to enable the PDF report download.")
        st.button("Download Official PDF Report (Disabled)", disabled=True, use_container_width=True)

else:
    st.error("Critical Failure: AI Assets (Model/Scaler) not found.")
