import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
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

# 4. TRAIN-TEST SPLIT (TIME SERIES STYLE)
split_index = int(len(df) * 0.8)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]


# 5. MODEL TRAINING
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


# 7. EVALUATION

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n====================")
print("MODEL PERFORMANCE")
print("====================")
print("RMSE:", rmse)
print("R2 Score:", r2)


# 8. SAVE MODEL

joblib.dump(model, "../models/energy_model.pkl")
print("\nModel saved successfully!")