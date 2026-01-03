import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# Logic to create balanced training data
data = {
    'age': [25, 60, 30, 65, 40, 70, 20, 55, 35, 80],
    'balance': [500, 50000, 1000, 30000, 2000, 80000, 100, 25000, 1500, 100000],
    'duration': [50, 2000, 100, 1500, 200, 3000, 30, 1200, 150, 4000],
    'y': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 0 = NO, 1 = YES
}

df = pd.DataFrame(data)
X = df[['age', 'balance', 'duration']].values
y = df['y'].values.reshape(-1, 1)

# Normalizing the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

model = BankModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training Loop
for epoch in range(500):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), 'marketing_model.pth')
print("Model retrained and saved successfully!")
