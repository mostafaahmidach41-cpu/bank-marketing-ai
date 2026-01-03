import torch
import torch.nn as nn
import torch.optim as optim

class BankModel(nn.Module):
    def __init__(self):
        super(BankModel, self).__init__()
        self.layer1 = nn.Linear(3, 8)
        self.layer2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x

model = BankModel()
X_train = torch.randn(100, 3)
y_train = torch.randint(0, 2, (100, 1)).float()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

for epoch in range(10):
    optimizer.zero_grad()
    loss = criterion(model(X_train), y_train)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "marketing_model.pth")
print("Successfully created marketing_model.pth")