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

st.title("Bank Marketing Prediction System")

@st.cache_resource
def load_model():
    model = BankModel()
    if os.path.exists('marketing_model.pth'):
        model.load_state_dict(torch.load('marketing_model.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

age = st.number_input("Age", value=60)
balance = st.number_input("Balance", value=30000.0)
duration = st.number_input("Duration (Days)", value=2000.0)

if st.button("Predict"):
    # Normalization matching train.py
    n_age, n_bal, n_dur = age/100.0, balance/100000.0, duration/5000.0
    input_data = torch.tensor([[n_age, n_bal, n_dur]], dtype=torch.float32)
    
    with torch.no_grad():
        prob = model(input_data).item()
    
    # Final adjustment to ensure logic wins over bias
    result = "YES" if prob > 0.4 else "NO"
    color = "green" if result == "YES" else "red"
    
    st.markdown(f"<h1 style='color:{color};'>{result}</h1>", unsafe_allow_html=True)
    st.write(f"Confidence: {prob:.4f}")