import pandas as pd
import matplotlib.pyplot as plt
import joblib

from feature_engineering import create_features

# ---------------------------
# 1. LOAD DATA
# ---------------------------
df = pd.read_csv("../data/energy.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# ---------------------------
# 2. LOAD MODEL
# ---------------------------
model = joblib.load("../models/energy_model.pkl")

# ---------------------------
# 3. FEATURE ENGINEERING
# ---------------------------
df = create_features(df)

features = ['hour', 'day', 'month', 'lag_1', 'lag_2', 'rolling_mean_3']

X = df[features]
df['predicted'] = model.predict(X)

# ---------------------------
# 4. LOAD FUTURE FORECAST
# ---------------------------
future_df = pd.read_csv("../outputs/future_forecast.csv")
future_df['date'] = pd.to_datetime(future_df['date'])
future_df = future_df.sort_values('date')

# ---------------------------
# 5. CLEAN DASHBOARD PLOT (FIXED)
# ---------------------------
plt.figure(figsize=(14,6))

# Show only last 50 points for clarity
plt.plot(
    df['date'][-50:],
    df['energy'][-50:],
    label="Actual Energy (Last 50 points)",
    linewidth=2
)

plt.plot(
    df['date'][-50:],
    df['predicted'][-50:],
    label="Model Prediction (Last 50 points)",
    linewidth=2
)

# Future forecast (24 hours)
plt.plot(
    future_df['date'],
    future_df['predicted_energy'],
    label="Future Forecast (Next 24h)",
    linewidth=3
)

# ---------------------------
# 6. STYLING FIXES
# ---------------------------
plt.title("AI Energy Consumption Forecasting Dashboard")
plt.xlabel("Time")
plt.ylabel("Energy Usage (kWh)")
plt.legend()
plt.grid(True)

plt.xticks(rotation=45)
plt.tight_layout()

# High-quality save for GitHub
plt.savefig("../images/final_dashboard.png", dpi=300)

plt.show()

print("\nDashboard generated successfully!")