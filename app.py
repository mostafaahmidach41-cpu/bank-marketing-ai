import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
import os

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

st.set_page_config(page_title="Bank Marketing AI", layout="centered")
st.title("Bank Marketing Prediction System")

@st.cache_resource
def load_assets():
    model = BankModel()
    if os.path.exists("marketing_model.pth"):
        model.load_state_dict(torch.load("marketing_model.pth", map_location="cpu"))
    model.eval()
    scaler = joblib.load("scaler.pkl") if os.path.exists("scaler.pkl") else None
    return model, scaler

model, scaler = load_assets()

# Updated UI with correct banking duration logic
age = st.number_input("Age (Years)", min_value=18, max_value=100, value=60)
balance = st.number_input("Account Balance", min_value=0.0, max_value=500000.0, value=30000.0)
duration = st.number_input("Duration (Days)", min_value=0.0, max_value=10000.0, value=2000.0)

if st.button("Predict"):
    if scaler is not None:
        input_array = np.array([[age, balance, duration]])
        input_scaled = scaler.transform(input_array)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        with torch.no_grad():
            prob = model(input_tensor).item()
    else:
        # Emergency logic if scaler is missing
        prob = (balance / 100000.0) * 0.4 + (duration / 5000.0) * 0.6

    # Logical threshold for YES result
    result = "YES" if prob >= 0.4 else "NO"
    confidence = prob if result == "YES" else 1 - prob
    color = "green" if result == "YES" else "red"

    st.markdown(f"<h1 style='color:{color}; text-align:center;'>{result}</h1>", unsafe_allow_html=True)
    st.metric("Confidence Score", f"{confidence:.4f}")
    st.success(f"System updated: Analyzing duration of {duration} days.")
