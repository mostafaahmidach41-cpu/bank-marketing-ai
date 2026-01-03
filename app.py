import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
import os

class BankModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

@st.cache_resource
def load_model():
    model = BankModel()
    model.load_state_dict(torch.load("marketing_model.pth", map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.save")

model = load_model()
scaler = load_scaler()

st.set_page_config(page_title="Bank AI", layout="centered")
st.title("Bank Marketing AI")

age = st.number_input("Age", min_value=18, max_value=100, value=40)
balance = st.number_input("Balance", min_value=0.0, max_value=200000.0, value=20000.0)
duration = st.number_input("Duration", min_value=1.0, max_value=5000.0, value=500.0)

if st.button("Predict"):
    raw = np.array([[age, balance, duration]])
    scaled = scaler.transform(raw)
    tensor = torch.tensor(scaled, dtype=torch.float32)

    with torch.no_grad():
        score = model(tensor).item()

    result = "YES" if score >= 0.5 else "NO"

    st.subheader("Result")
    st.markdown(
        f"<h1 style='color:{'green' if result=='YES' else 'red'}'>{result}</h1>",
        unsafe_allow_html=True
    )
    st.metric("Confidence", f"{score:.4f}")

