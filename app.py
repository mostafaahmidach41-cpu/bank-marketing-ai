import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
import os

# Model architecture (MUST match train.py exactly)
class BankModel(nn.Module):
    def __init__(self):
        super(BankModel, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Load model
model = BankModel()
model.load_state_dict(torch.load("marketing_model.pth", map_location="cpu"))
model.eval()

# Load scaler
scaler = joblib.load("scaler.pkl")

# UI
st.set_page_config(page_title="Bank Marketing AI", layout="centered")
st.title("Bank Marketing Prediction System")

age = st.number_input("Age", min_value=18, max_value=95, value=60)
balance = st.number_input("Balance", min_value=0.0, max_value=200000.0, value=30000.0)
duration = st.number_input("Duration", min_value=0.0, max_value=5000.0, value=2000.0)

if st.button("Predict"):
    # Prepare input
    input_array = np.array([[age, balance, duration]])
    input_scaled = scaler.transform(input_array)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    with torch.no_grad():
        prob = model(input_tensor).item()

    result = "YES" if prob >= 0.5 else "NO"
    confidence = prob if result == "YES" else 1 - prob

    color = "green" if result == "YES" else "red"

    st.markdown(f"<h1 style='color:{color}; text-align:center;'>{result}</h1>", unsafe_allow_html=True)
    st.metric("Confidence", f"{confidence:.4f}")
    st.success("Prediction completed successfully.")

