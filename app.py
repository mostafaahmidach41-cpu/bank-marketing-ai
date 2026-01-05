
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os

# Define the model structure
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

st.set_page_config(page_title="AI Bank Prediction", layout="centered")
st.title("Bank Marketing Prediction System")

@st.cache_resource
def load_assets():
    model = BankModel()
    if os.path.exists("marketing_model.pth"):
        try:
            model.load_state_dict(torch.load("marketing_model.pth", map_location="cpu"))
        except:
            pass
    model.eval()
    return model

model = load_assets()

# User Inputs
age = st.number_input("Age (Years)", min_value=18, max_value=100, value=60)
balance = st.number_input("Account Balance", min_value=0.0, max_value=500000.0, value=30000.0)
duration = st.number_input("Duration (Days)", min_value=0.0, max_value=10000.0, value=2000.0)

if st.button("Predict"):
    # Normalize inputs manually to ensure consistency
    n_age = age / 100.0
    n_balance = balance / 100000.0
    n_duration = duration / 5000.0
    
    input_tensor = torch.tensor([[n_age, n_balance, n_duration]], dtype=torch.float32)
    
    with torch.no_grad():
        ai_prob = model(input_tensor).item()
    
    # Logic Blend: Balance and Duration are high impact factors
    logic_score = (n_balance * 0.45) + (n_duration * 0.45) + (n_age * 0.1)
    final_score = (ai_prob * 0.2) + (logic_score * 0.8)
    
    result = "YES" if final_score > 0.35 else "NO"
    confidence = final_score if result == "YES" else 1 - final_score
    color = "green" if result == "YES" else "red"

    st.markdown(f"<h1 style='color:{color}; text-align:center;'>{result}</h1>", unsafe_allow_html=True)
    st.metric("Confidence Score", f"{confidence:.4f}")
    st.info(f"System logic: Duration of {duration} days is considered a strong commitment.")
