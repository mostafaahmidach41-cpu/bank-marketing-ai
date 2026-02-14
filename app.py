import streamlit as st
import pickle
import numpy as np
import pandas as pd
import datetime
from supabase import create_client, Client
import plotly.express as px
from fpdf import FPDF

# --------------------------------------------------
# CONFIG (RUN ONCE ONLY)
# --------------------------------------------------
st.set_page_config(page_title="Enterprise AI Terminal", layout="wide")

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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
# AUTHENTICATION LAYER
# --------------------------------------------------
def authenticate_license(key):
    response = (
        supabase.table("licenses")
        .select("key_value")
        .eq("key_value", key)
        .eq("is_active", True)
        .execute()
    )
    return bool(response.data)


if not st.session_state.authenticated:
    st.title("Enterprise Security Portal")

    license_input = st.text_input("Enter License Key")

    if st.button("Activate System"):
        if authenticate_license(license_input):
            st.session_state.authenticated = True
            st.session_state.license_key = license_input
            st.rerun()
        else:
            st.error("Invalid or inactive license")

    st.stop()

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


model, scaler = load_model()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.success(f"Active License: {st.session_state.license_key}")

if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.license_key = None
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Analytics Dashboard")

# --------------------------------------------------
# LOAD USER DATA (ISOLATED PER TENANT)
# --------------------------------------------------
@st.cache_data(ttl=60)
def load_user_logs(license_key):
    response = (
        supabase.table("audit_logs")
        .select("*")
        .eq("license_key", license_key)
        .order("created_at", desc=True)
        .execute()
    )
    return pd.DataFrame(response.data) if response.data else pd.DataFrame()


logs_df = load_user_logs(st.session_state.license_key)

if not logs_df.empty:
    fig = px.pie(
        logs_df,
        names="decision",
        title="Eligibility Rate"
    )
    st.sidebar.plotly_chart(fig, use_container_width=True)

    st.sidebar.metric("Total Assessments", len(logs_df))


# --------------------------------------------------
# MAIN INPUT
# --------------------------------------------------
st.title("Customer AI Assessment Terminal")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Customer Age", 18, 95, 35)
    balance = st.number_input("Yearly Balance ($)", 0, 1_000_000, 200_000)

with col2:
    tenure = st.number_input("Relationship Tenure (Years)", 0, 50, 5)

# --------------------------------------------------
# AI DECISION ENGINE
# --------------------------------------------------
if st.button("Generate AI Decision", use_container_width=True):

    features = np.array([[age, balance, tenure]])
    scaled = scaler.transform(features)

    prediction = model.predict(scaled)[0]
    probabilities = model.predict_proba(scaled)[0]

    confidence = float(max(probabilities) * 100)
    decision = "ELIGIBLE" if prediction == 1 else "NOT ELIGIBLE"

    # Safe Feature Importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = abs(model.coef_[0])
    else:
        importances = [0.33, 0.33, 0.34]

    st.session_state.last_result = {
        "age": age,
        "balance": balance,
        "tenure": tenure,
        "decision": decision,
        "confidence": confidence,
        "importances": importances
    }

    # Insert Secure Log
    supabase.table("audit_logs").insert({
        "license_key": st.session_state.license_key,
        "customer_age": age,
        "balance": float(balance),
        "tenure": tenure,
        "decision": decision,
        "confidence": confidence
    }).execute()

    st.rerun()

# --------------------------------------------------
# RESULTS DISPLAY
# --------------------------------------------------
if st.session_state.last_result:
    result = st.session_state.last_result

    st.markdown("---")

    if result["decision"] == "ELIGIBLE":
        st.success(f"Result: ELIGIBLE | Confidence: {result['confidence']:.2f}%")
    else:
        st.error(f"Result: NOT ELIGIBLE | Confidence: {result['confidence']:.2f}%")

    st.progress(result["confidence"] / 100)

    # Feature Impact Chart
    impact_df = pd.DataFrame({
        "Feature": ["Age", "Balance", "Tenure"],
        "Impact": result["importances"]
    }).sort_values("Impact")

    fig_imp = px.bar(
        impact_df,
        x="Impact",
        y="Feature",
        orientation="h"
    )
    st.plotly_chart(fig_imp, use_container_width=True)

# --------------------------------------------------
# RECENT ACTIVITY
# --------------------------------------------------
st.markdown("---")
st.subheader("Recent Activity Log")

if not logs_df.empty:
    display_df = logs_df[[
        "customer_age",
        "balance",
        "tenure",
        "decision",
        "confidence"
    ]].head(10)

    st.dataframe(display_df, use_container_width=True)
