import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# Generate dummy dataset
np.random.seed(42)
data_size = 500

df = pd.DataFrame({
    "Temperature": np.random.uniform(20, 60, size=data_size),
    "Humidity": np.random.uniform(20, 90, size=data_size),
    "Rain": np.random.uniform(0, 100, size=data_size),
    "Flame": np.random.uniform(0, 1023, size=data_size),
    "Water Level": np.random.uniform(0, 100, size=data_size),
    "Thunder Sound": np.random.uniform(0, 1023, size=data_size),
    "Voltage Drop": np.random.uniform(210, 240, size=data_size),
    "Current Drop": np.random.uniform(0, 10, size=data_size),
})

# Define a dummy target based on flame and rain thresholds
df["RiskLevel"] = np.where((df["Flame"] > 700) | (df["Rain"] > 80), 1, 0)  # 1 = High risk, 0 = Normal

# Train a model
X = df[["Temperature", "Humidity", "Rain", "Flame", "Water Level", "Thunder Sound", "Voltage Drop", "Current Drop"]]
y = df["RiskLevel"]

model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
