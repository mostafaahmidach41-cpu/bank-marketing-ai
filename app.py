import streamlit as st
import pickle
import numpy as np
from fpdf import FPDF

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Bank AI - Confidence Analytics",
    layout="wide"
)

# --- 2. PDF Report Function (with confidence metric) ---
def generate_pdf_report(data_summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "AI Assessment Report with Confidence Metrics", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)

    for key, value in data_summary.items():
        pdf.cell(200, 10, f"{key}: {value}", ln=True)

    return pdf.output(dest='S').encode('latin-1')

# --- 3. Load Model and Scaler ---
@st.cache_resource
def load_system_assets():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("Model files are missing.")
        return None, None

model, scaler = load_system_assets()

# --- 4. Interface and Analysis ---
if model and scaler:
    st.sidebar.success("System Status: Online")
    st.title("Customer Eligibility Assessment with Confidence Analysis")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Customer Age", 18, 95, 35)
        balance = st.number_input("Annual Balance ($)", 0, 500000, 2500)

    with col2:
        duration = st.number_input("Relationship Duration (Years)", 0, 50, 5)
        reference_day = st.slider("Reference Day", 1, 31, 15)

    if st.button("Run AI Assessment"):
        try:
            # Prepare input features
            input_features = np.array([[age, balance, duration]])
            scaled_input = scaler.transform(input_features)

            # Prediction and confidence calculation
            prediction = model.predict(scaled_input)
            probability = model.predict_proba(scaled_input)[0]

            confidence = max(probability) * 100
            result_label = "Eligible" if prediction[0] == 1 else "Not Eligible"

            # Display results
            st.markdown(f"### Decision: **{result_label}**")
            st.write(f"**AI Confidence Level:** {confidence:.2f}%")
            st.progress(confidence / 100)

            if prediction[0] == 1:
                st.success(f"The system is {confidence:.2f}% confident the customer will accept the offer.")
            else:
                st.warning(f"The system is {confidence:.2f}% confident the customer will not accept the offer at this time.")

            # Generate PDF Report
            st.markdown("---")
            pdf_bytes = generate_pdf_report({
                "Age": age,
                "Balance": balance,
                "Duration": duration,
                "Decision": result_label,
                "Confidence": f"{confidence:.2f}%"
            })

            st.download_button(
                label="Download Final Report",
                data=pdf_bytes,
                file_name="AI_Assessment_Report.pdf",
                mime="application/pdf"
            )

        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.info("Note: If probability error occurs, ensure the model was trained with probability=True.")

