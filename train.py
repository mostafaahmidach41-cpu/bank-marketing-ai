
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

np.random.seed(42)

rows = 1000
age = np.random.randint(18, 80, rows)
balance = np.random.randint(0, 120000, rows)
duration = np.random.randint(10, 4000, rows)

y = (
    (age > 45).astype(int) +
    (balance > 30000).astype(int) +
    (duration > 800).astype(int)
)
y = (y >= 2).astype(int)

df = pd.DataFrame({
    "age": age,
    "balance": balance,
    "duration": duration,
    "y": y
})

X = df[["age", "balance", "duration"]].values
y = df["y"].values.reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

joblib.dump(scaler, "scaler.save")

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

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

model = BankModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

for epoch in range(200):
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "marketing_model.pth")
print("Model trained successfully")
