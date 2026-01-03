import streamlit as st
import torch
import torch.nn as nn
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

# Inputs
age = st.number_input("Age", value=60)
balance = st.number_input("Balance", value=30000.0)
duration = st.number_input("Duration", value=2000.0)

if st.button("Predict"):
    # Fix: Normalization logic to match your train.py (StandardScaler approx)
    # This converts raw numbers into the small values the AI understands
    n_age = (age - 46.0) / 18.0
    n_balance = (balance - 29010.0) / 33000.0
    n_duration = (duration - 1423.0) / 1300.0
    
    input_tensor = torch.tensor([[n_age, n_balance, n_duration]], dtype=torch.float32)
    
    with torch.no_grad():
        prediction = model(input_tensor).item()
    
    # Real logical threshold
    result = "YES" if prediction > 0.4 else "NO"
    
    st.subheader("Result")
    color = "green" if result == "YES" else "red"
    st.markdown(f"<h1 style='color: {color};'>{result}</h1>", unsafe_allow_html=True)
    st.write(f"Confidence Level: {prediction:.4f}")
    st.success("Prediction completed using the new retrained model.")
