import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

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

# Balanced Logical Training Data
data = {
    'age': [25, 60, 30, 65, 40, 70, 20, 80],
    'balance': [500, 50000, 1000, 30000, 2000, 100000, 100, 80000],
    'duration': [50, 2000, 100, 1500, 200, 4000, 30, 3000],
    'y': [0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)
# Normalization: divide by max expected values
X = torch.tensor(df[['age', 'balance', 'duration']].values / [100, 100000, 5000], dtype=torch.float32)
y = torch.tensor(df['y'].values.reshape(-1, 1), dtype=torch.float32)

model = BankModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

for epoch in range(1000):
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), 'marketing_model.pth')
print("Logical Model Trained Successfully")