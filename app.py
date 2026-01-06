import streamlit as st
import pickle
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="Bank Customer Eligibility Assessment",
    layout="wide"
)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.markdown("""
<style>
.main-title {
    font-size: 32px;
    font-weight: 700;
    color: #0A2540;
}
.sub-title {
    font-size: 16px;
    color: #4A5568;
}
.card {
    background-color: #F7FAFC;
    padding: 20px;
    border-radius: 12px;
}
.decision-yes {
    color: #0F766E;
    font-size: 26px;
    font-weight: 700;
}
.decision-no {
    color: #991B1B;
    font-size: 26px;
    font-weight: 700;
}
.metric {
    font-size: 18px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>Bank Customer Eligibility Assessment</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Decision support system for marketing campaign eligibility</div>", unsafe_allow_html=True)
st.markdown("---")

left, right = st.columns([1, 1.2])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Customer Information")

    age = st.slider("Customer Age (Years)", 18, 80, 35)
    balance = st.number_input("Account Balance (USD)", 0.0, 500000.0, 25000.0, step=500.0)
    duration = st.slider("Contact Duration (Days)", 1, 3650, 180)

    submit = st.button("Assess Eligibility")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Decision Summary")

    if submit:
        X = np.array([[age, balance, duration]])
        X_scaled = scaler.transform(X)

        prob = model.predict_proba(X_scaled)[0][1]
        prediction = 1 if prob >= 0.5 else 0

        if prediction == 1:
            st.markdown("<div class='decision-yes'>Eligible for Campaign</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='decision-no'>Not Eligible for Campaign</div>", unsafe_allow_html=True)

        st.markdown("---")

        confidence = round(prob, 2)
        st.markdown(f"<div class='metric'>Decision Confidence: {confidence}</div>", unsafe_allow_html=True)

        if prob >= 0.7:
            risk = "Low Risk"
        elif prob >= 0.4:
            risk = "Medium Risk"
        else:
            risk = "High Risk"

        st.markdown(f"<div class='metric'>Risk Level: {risk}</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("Decision influenced mainly by customer balance and engagement duration.")

        st.markdown("---")
        st.caption(f"Assessment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    else:
        st.info("Enter customer data and click 'Assess Eligibility'")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption(
    "This system provides decision support only and does not replace human judgment or official credit approval processes."
)

