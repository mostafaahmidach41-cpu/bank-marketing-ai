  import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import io
import plotly.express as px

class MarketingModel(nn.Module):
    def __init__(self):
        super(MarketingModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x)

def load_model():
    model = MarketingModel()
    model.load_state_dict(torch.load('marketing_model.pth'))
    model.eval()
    return model

st.set_page_config(page_title="Bank AI SaaS", layout="wide")

if 'predictions' not in st.session_state:
    st.session_state.predictions = pd.DataFrame(columns=['Email', 'Age', 'Balance', 'Duration', 'Result', 'Confidence'])

st.title("Bank Marketing AI Predictor")

email = st.text_input("Enter your Email to login:")

if email:
    st.success(f"Welcome: {email}")
    
    st.sidebar.header("Inputs")
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
    balance = st.sidebar.number_input("Balance", value=1000.0)
    duration = st.sidebar.number_input("Duration", value=10.0)

    if st.sidebar.button("Predict"):
        model = load_model()
        input_data = torch.tensor([[age, balance, duration]], dtype=torch.float32)
        with torch.no_grad():
            output = model(input_data).item()
        
        result = "YES" if output > 0.5 else "NO"
        confidence = output if output > 0.5 else 1 - output

        new_data = {
            'Email': email,
            'Age': age,
            'Balance': balance,
            'Duration': duration,
            'Result': result,
            'Confidence': f"{confidence:.4f}"
        }
        st.session_state.predictions = pd.concat([st.session_state.predictions, pd.DataFrame([new_data])], ignore_index=True)

        st.write(f"### Result: {result}")
        st.write(f"### Confidence: {confidence:.4f}")

    if not st.session_state.predictions.empty:
        st.write("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Prediction History")
            st.dataframe(st.session_state.predictions)
            
            output_excel = io.BytesIO()
            with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                st.session_state.predictions.to_excel(writer, index=False)
            
            st.download_button(
                label="Download Report",
                data=output_excel.getvalue(),
                file_name="marketing_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        with col2:
            st.subheader("Analytics Chart")
            fig = px.pie(st.session_state.predictions, names='Result', title='Success vs Failure Distribution')
            st.plotly_chart(fig)
