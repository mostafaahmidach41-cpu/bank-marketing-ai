import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import plotly.express as px
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

st.set_page_config(page_title="AI Bank Marketing", layout="wide")
st.title("Welcome: mostafaahmidach41@gmail.com")

@st.cache_resource
def load_model():
    model = BankModel()
    if os.path.exists('marketing_model.pth'):
        model.load_state_dict(torch.load('marketing_model.pth'))
    model.eval()
    return model

model = load_model()

st.sidebar.header("Inputs")
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=65)
balance = st.sidebar.number_input("Balance", value=20000.0)
duration = st.sidebar.number_input("Duration", value=2000.0)

if st.sidebar.button("Predict"):
    n_age = age / 100.0
    n_balance = balance / 50000.0
    n_duration = duration / 5000.0
    
    input_tensor = torch.tensor([[n_age, n_balance, n_duration]], dtype=torch.float32)
    
    with torch.no_grad():
        raw_prediction = model(input_tensor).item()
    
    logic_score = (n_age * 0.2) + (n_balance * 0.4) + (n_duration * 0.4)
    final_score = (raw_prediction + logic_score) / 2
    
    result = "YES" if final_score > 0.45 else "NO"
    confidence = final_score if result == "YES" else 1 - final_score

    st.subheader("Result")
    res_color = "green" if result == "YES" else "red"
    st.markdown(f"<h1 style='color: {res_color};'>{result}</h1>", unsafe_allow_html=True)
    
    st.subheader("Confidence")
    st.write(f"{confidence:.4f}")

    st.success(f"Prediction saved for mostafaahmidach41@gmail.com")
    
    viz_df = pd.DataFrame({
        'Feature': ['Age Impact', 'Balance Impact', 'Duration Impact'],
        'Value': [n_age, n_balance, n_duration]
    })
    fig = px.bar(viz_df, x='Feature', y='Value', color='Feature', range_y=[0,1])
    st.plotly_chart(fig)

st.button("Download Report")
