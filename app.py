import streamlit as st
import torch
import torch.nn as nn
import numpy as np
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

st.title("AI Bank Marketing Prediction")

@st.cache_resource
def load_model():
    model = BankModel()
    if os.path.exists('marketing_model.pth'):
        model.load_state_dict(torch.load('marketing_model.pth'))
    model.eval()
    return model

model = load_model()

age = st.number_input("Age", value=60)
balance = st.number_input("Balance", value=30000.0)
duration = st.number_input("Duration", value=2000.0)

if st.button("Predict"):
    # Manual normalization to match your training logic
    n_age = (age - 47.0) / 19.0 
    n_bal = (balance - 29010.0) / 33000.0
    n_dur = (n_age + n_bal) / 2 # Simple logical blending
    
    input_data = torch.tensor([[n_age, n_bal, (duration/4000)]], dtype=torch.float32)
    
    with torch.no_grad():
        prediction = model(input_data).item()
    
    result = "YES" if prediction > 0.5 else "NO"
    st.header(f"Result: {result}")
    st.write(f"Confidence: {prediction:.4f}")
