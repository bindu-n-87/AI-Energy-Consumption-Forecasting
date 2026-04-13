import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import mean_squared_error, r2_score
from feature_engineering import create_features


df = pd.read_csv("../data/energy.csv")

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')


df = create_features(df)

features = ['hour', 'day', 'month', 'lag_1', 'lag_2', 'rolling_mean_3']
target = 'energy'

X = df[features]
y = df[target]

model = joblib.load("../models/energy_model.pkl")

y_pred = model.predict(X)

rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print("\n====================")
print("FINAL MODEL EVALUATION")
print("====================")
print("RMSE:", rmse)
print("R2 Score:", r2)


plt.figure(figsize=(10,5))
plt.plot(y.values, label="Actual Energy", linewidth=2)
plt.plot(y_pred, label="Predicted Energy", linewidth=2)
plt.title("Energy Consumption Forecasting: Actual vs Predicted")
plt.xlabel("Time Steps")
plt.ylabel("Energy Usage")
plt.legend()

# Save plot
plt.savefig("../images/actual_vs_predicted.png")
plt.show()