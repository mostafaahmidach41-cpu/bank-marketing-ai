import streamlit as st
import pickle
import numpy as np
from supabase import create_client, Client
from fpdf import FPDF
import datetime
import plotly.express as px
import pandas as pd
import os

# --- Secure Configuration (Production Standard) ---
URL = "https://ixwvplxnfndjbmdsvdpu.supabase.co"
KEY = os.getenv("SUPABASE_KEY")

if not KEY:
    st.error("Missing SUPABASE_KEY. Please set it in Streamlit Secrets.")
    st.stop()

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
    st.title("Enterprise Security Portal")

    user_input = st.text_input(
        "Username or License Key",
        type="password",
        placeholder="Enter license key"
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
    st.set_page_config(page_title="AI Assessment Terminal", layout="wide")
    st.title("Customer AI Assessment Terminal")

    # --- Multi-tenant Analytics (Data Isolation) ---
    st.sidebar.info(f"Identity: {st.session_state.current_user}")

    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

    all_data_df = pd.DataFrame()

    try:
        response = (
            supabase
            .table("audit_logs")
            .select("*")
            .eq("license_key", st.session_state.current_user)
            .order("created_at", desc=True)
            .execute()
        )

        if response.data:
            all_data_df = pd.DataFrame(response.data)

            st.sidebar.markdown("---")
            st.sidebar.subheader("Your Org Analytics")

            fig_pie = px.pie(
                all_data_df,
                names="decision",
                title="Eligibility Distribution",
                color="decision",
                color_discrete_map={
                    "ELIGIBLE": "#2ecc71",
                    "NOT ELIGIBLE": "#e74c3c"
                }
            )

            fig_pie.update_layout(
                showlegend=False,
                height=200,
                margin=dict(t=30, b=0, l=0, r=0)
            )

            st.sidebar.plotly_chart(fig_pie, use_container_width=True)
            st.sidebar.metric("Your Total Checks", len(all_data_df))

            csv_data = all_data_df.to_csv(index=False).encode("utf-8")

            st.sidebar.download_button(
                "Export My Logs (CSV)",
                csv_data,
                f"My_Logs_{datetime.date.today()}.csv",
                "text/csv",
                use_container_width=True
            )

    except Exception:
        pass

    # --- Input Section ---
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Customer Age", 18, 95, 35)
        balance = st.number_input("Yearly Balance ($)", 0, 1_000_000, 25000)

    with col2:
        tenure = st.number_input("Relationship Tenure (Years)", 0, 50, 9)
        st.info("System isolated for: " + st.session_state.current_user)

    # --- Decision Engine ---
    if st.button("Execute AI Analysis", use_container_width=True):
        try:
            features = np.array([[age, balance, tenure]])
            scaled = scaler.transform(features)

            prediction = model.predict(scaled)
            probabilities = model.predict_proba(scaled)[0]

            confidence = float(np.max(probabilities) * 100)
            confidence = round(confidence, 2)

            decision = "ELIGIBLE" if prediction[0] == 1 else "NOT ELIGIBLE"

            try:
                importances = model.feature_importances_
            except AttributeError:
                try:
                    importances = np.abs(model.coef_[0])
                except Exception:
                    importances = np.array([0.33, 0.34, 0.33])

            st.session_state.last_result = {
                "age": age,
                "balance": balance,
                "tenure": tenure,
                "decision": decision,
                "confidence": confidence,
                "importances": importances
            }

            audit_entry = {
                "license_key": st.session_state.current_user,
                "customer_age": age,
                "balance": float(balance),
                "tenure": tenure,
                "decision": decision,
                "confidence": confidence
            }

            supabase.table("audit_logs").insert(audit_entry).execute()
            st.rerun()

        except Exception as e:
            st.error(f"Core Error: {e}")

    # --- Results & Explainability ---
    if st.session_state.last_result:
        res = st.session_state.last_result

        st.markdown("---")

        c1, c2 = st.columns([1, 1])

        with c1:
            if res["decision"] == "ELIGIBLE":
                st.success(f"Final Decision: {res['decision']}")
            else:
                st.warning(f"Final Decision: {res['decision']}")

            st.metric("Confidence Score", f"{res['confidence']}%")

        with c2:
            st.write("Feature Impact Analysis")

            imp_df = pd.DataFrame({
                "Feature": ["Age", "Balance", "Tenure"],
                "Impact": res["importances"]
            }).sort_values(by="Impact")

            fig_imp = px.bar(
                imp_df,
                x="Impact",
                y="Feature",
                orientation="h",
                color="Impact",
                color_continuous_scale="Blues"
            )

            fig_imp.update_layout(
                height=160,
                margin=dict(t=0, b=0, l=0, r=0),
                coloraxis_showscale=False
            )

            st.plotly_chart(fig_imp, use_container_width=True)

    # --- Activity Log ---
    st.markdown("---")
    st.subheader("Recent Records")

    if not all_data_df.empty:
        st.dataframe(
            all_data_df[
                ["customer_age", "balance", "tenure", "decision", "confidence"]
            ].head(5),
            use_container_width=True
        )

    # --- PDF Reporting ---
    if st.session_state.last_result:
        res = st.session_state.last_result

        pdf_data = create_assessment_report(
            res["age"],
            res["balance"],
            res["tenure"],
            res["decision"],
            res["confidence"]
        )

        st.download_button(
            "Download Secure PDF",
            pdf_data,
            f"Report_{st.session_state.current_user}.pdf",
            "application/pdf",
            use_container_width=True
        )

else:
    st.error("System Failure: Assets Missing.")
