import pandas as pd
import numpy as np
import joblib

from feature_engineering import create_features

df = pd.read_csv("../data/energy.csv")

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

model = joblib.load("../models/energy_model.pkl")

df = create_features(df)

features = ['hour', 'day', 'month', 'lag_1', 'lag_2', 'rolling_mean_3']

last_row = df.iloc[-1].copy()

future_predictions = []

for i in range(24):

    # Create next timestamp
    next_time = df['date'].iloc[-1] + pd.Timedelta(hours=i+1)

    # Build input row
    input_data = pd.DataFrame([last_row[features]])

    # Predict energy
    pred = model.predict(input_data)[0]

    # Save prediction
    future_predictions.append({
        "date": next_time,
        "predicted_energy": pred
    })

    # Update lag values (simulate real system)
    last_row['lag_2'] = last_row['lag_1']
    last_row['lag_1'] = pred
    last_row['rolling_mean_3'] = pred


forecast_df = pd.DataFrame(future_predictions)

print("\nFuture Energy Forecast:")
print(forecast_df)


forecast_df.to_csv("../outputs/future_forecast.csv", index=False)

print("\nForecast saved to outputs/future_forecast.csv")
