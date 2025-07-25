import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv("student_data.csv")

# Features & Target
X = df[["Hours_Studied", "Attendance", "Sleep_Hours"]]
y = df["Grade"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save trained model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
