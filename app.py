import streamlit as st
import torch
import torch.nn as nn
import os
import re
import pandas as pd
from datetime import datetime

class BankModel(nn.Module):
    def __init__(self):
        super(BankModel, self).__init__()
        self.layer1 = nn.Linear(3, 8)
        self.layer2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x

st.set_page_config(page_title="Bank AI SaaS", layout="centered")

def is_valid_gmail(email):
    pattern = r'^[a-z0-9](\.?[a-z0-9]){5,}@gmail\.com$'
    return re.match(pattern, email)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("Login")
    email_input = st.text_input("Gmail Address:", placeholder="user@gmail.com")
    
    if st.button("Sign In"):
        if is_valid_gmail(email_input):
            st.session_state.authenticated = True
            st.session_state.user_email = email_input
            st.rerun()
        else:
            st.error("Invalid Gmail format")
    st.stop()

st.title(f"Welcome: {st.session_state.user_email}")

if os.path.exists("marketing_model.pth"):
    try:
        model = BankModel()
        model.load_state_dict(torch.load("marketing_model.pth", weights_only=True))
        model.eval()
        
        st.sidebar.header("Inputs")
        age = st.sidebar.number_input("Age", 18, 95, 30)
        balance = st.sidebar.number_input("Balance", 0.0, 100000.0, 1000.0)
        duration = st.sidebar.number_input("Duration", 0.0, 1000.0, 10.0)

        if st.sidebar.button("Predict"):
            inp = torch.tensor([[float(age), float(balance), float(duration)]], dtype=torch.float32)
            with torch.no_grad():
                prediction = model(inp).item()
            
            label = "NO" if prediction >= 0.5 else "YES"
            st.metric("Result", label)
            st.metric("Confidence", f"{prediction:.4f}")
            
            new_data = {
                "Email": [st.session_state.user_email],
                "Age": [age],
                "Balance": [balance],
                "Duration": [duration],
                "Prediction": [label],
                "Confidence": [prediction],
                "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            }
            df = pd.DataFrame(new_data)
            file_path = "report.xlsx"
            
            if os.path.exists(file_path):
                old_df = pd.read_excel(file_path)
                df = pd.concat([old_df, df], ignore_index=True)
            
            df.to_excel(file_path, index=False)
            st.success(f"Prediction saved for {st.session_state.user_email}")

        if os.path.exists("report.xlsx"):
            with open("report.xlsx", "rb") as f:
                st.download_button(
                    label="Download Report",
                    data=f,
                    file_name="bank_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.error("Model file missing")