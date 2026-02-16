import streamlit as st
import pandas as pd
import numpy as np
import datetime
from fpdf import FPDF

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="Customer AI Assessment Terminal",
    layout="wide"
)

# ==============================
# Session State Initialization
# ==============================
if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "license_key" not in st.session_state:
    st.session_state.license_key = "PREMIUM-BANK-2026"

# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.success(f"Active License: {st.session_state.license_key}")
    st.button("Logout")

# ==============================
# Main Title
# ==============================
st.title("Customer AI Assessment Terminal")

# ==============================
# Input Section
# ==============================
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Customer Age", 18, 80, 35)
    balance = st.number_input("Yearly Balance ($)", 0, 1000000, 20000)

with col2:
    tenure = st.number_input("Relationship Tenure (Years)", 0, 50, 5)

# ==============================
# AI Decision Logic
# ==============================
def ai_decision(age, balance, tenure):
    score = (balance / 10000) + (tenure * 2) - (age / 50)

    if score > 5:
        return "ELIGIBLE", round(min(score / 10, 0.99), 4)
    else:
        return "NOT ELIGIBLE", round(max(score / 10, 0), 4)

# ==============================
# PDF Generator
# ==============================
def generate_pdf_report(result, confidence, license_key):

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Customer AI Assessment Report", ln=True, align="C")
    pdf.ln(10)

    pdf.cell(200, 10, txt=f"Decision: {result}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence Score: {confidence}", ln=True)
    pdf.cell(200, 10, txt=f"License: {license_key}", ln=True)
    pdf.cell(200, 10, txt=f"Date: {datetime.date.today()}", ln=True)

    return pdf.output(dest="S")

# ==============================
# Generate Decision Button
# ==============================
if st.button("Generate AI Decision"):

    result, confidence = ai_decision(age, balance, tenure)

    st.session_state.last_result = {
        "age": age,
        "balance": balance,
        "tenure": tenure,
        "decision": result,
        "confidence": confidence
    }

    st.session_state.history.append(st.session_state.last_result)

# ==============================
# Display Result
# ==============================
if st.session_state.last_result is not None:

    res = st.session_state.last_result

    st.divider()
    st.subheader("AI Decision Result")

    if res["decision"] == "ELIGIBLE":
        st.success(f"Decision: {res['decision']}")
    else:
        st.error(f"Decision: {res['decision']}")

    st.write(f"Confidence Score: {res['confidence']}")

    pdf_bytes = generate_pdf_report(
        res["decision"],
        res["confidence"],
        st.session_state.license_key
    )

    st.download_button(
        label="Download Official PDF Report",
        data=pdf_bytes,
        file_name=f"Assessment_{datetime.date.today()}.pdf",
        mime="application/pdf"
    )

# ==============================
# Activity Log
# ==============================
st.divider()
st.subheader("Recent Activity Log")

if len(st.session_state.history) > 0:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)
else:
    st.info("No activity yet.")
