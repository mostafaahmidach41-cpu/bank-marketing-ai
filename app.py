import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# ---------------------------
# Simple Login Session
# ---------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("Enterprise AI Portal 🔐")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if email and password:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Please enter your credentials")

    st.stop()

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.success("Identity: PREMIUM-BANK-2026")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Org Analytics")

# Dummy Data
np.random.seed(1)
data = pd.DataFrame({
    "Age": np.random.randint(25, 60, 40),
    "Balance": np.random.randint(20000, 120000, 40),
    "Tenure": np.random.randint(1, 20, 40)
})

data["Decision"] = np.where(
    (data["Balance"] > 70000) & (data["Tenure"] > 5),
    "ELIGIBLE",
    "NOT ELIGIBLE"
)

data["Confidence"] = np.random.uniform(50, 80, 40).round(2)

# ---------------------------
# Sidebar Charts
# ---------------------------
eligible_count = (data["Decision"] == "ELIGIBLE").sum()
not_eligible_count = (data["Decision"] == "NOT ELIGIBLE").sum()

fig1, ax1 = plt.subplots()
ax1.pie(
    [eligible_count, not_eligible_count],
    labels=["Eligible", "Not Eligible"],
    autopct="%1.0f%%"
)

st.sidebar.pyplot(fig1)

st.sidebar.markdown("**Total Assessments**")
st.sidebar.write(len(data))

# ---------------------------
# Main Layout
# ---------------------------
col1, col2 = st.columns([2, 1])

latest = data.iloc[-1]

with col1:
    if latest["Decision"] == "ELIGIBLE":
        st.success(f"Result: {latest['Decision']}")
    else:
        st.error(f"Result: {latest['Decision']}")

    st.write(f"### Confidence Score: {latest['Confidence']}%")
    st.progress(int(latest["Confidence"]))

with col2:
    st.subheader("Why this decision? (Feature Impact)")

    impact_data = pd.DataFrame({
        "Feature": ["Age", "Tenure", "Balance"],
        "Impact": [0.85, 0.75, 0.65]
    })

    st.bar_chart(impact_data.set_index("Feature"))

# ---------------------------
# Activity Log
# ---------------------------
st.markdown("---")
st.subheader("Recent Activity Log")

st.dataframe(
    data[["Age", "Balance", "Tenure", "Decision", "Confidence"]],
    use_container_width=True
)
