import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Balanced training data for logical bank marketing predictions
# 0 = Not Eligible, 1 = Eligible
data = {
    'age': [25, 60, 30, 65, 40, 70, 20, 80, 35, 75],
    'balance': [500, 50000, 1000, 30000, 2000, 100000, 100, 80000, 1500, 90000],
    'duration': [10, 2000, 50, 1500, 100, 3000, 20, 2500, 80, 3500],
    'y': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)
X = df[['age', 'balance', 'duration']]
y = df['y']

# Data Normalization using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Training Logistic Regression Model
# This model supports the predict_proba function used in your new UI
model = LogisticRegression()
model.fit(X_scaled, y)

# Saving the artifacts in pickle format (.pkl)
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Success! 'model.pkl' and 'scaler.pkl' have been generated successfully.")