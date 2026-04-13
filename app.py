import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from src.feature_engineering import create_features

st.set_option('client.showErrorDetails', True)

# ---------------------------
# TITLE
# ---------------------------
st.title("⚡ AI-Powered Energy Forecasting System")
st.write("Smart Energy Consumption Prediction Dashboard")

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv("data/energy.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# ---------------------------
# LOAD MODEL
# ---------------------------
model = joblib.load("models/energy_model.pkl")

# ---------------------------
# FEATURE ENGINEERING
# ---------------------------
df = create_features(df)

features = ['hour', 'day', 'month', 'lag_1', 'lag_2', 'rolling_mean_3']

df['prediction'] = model.predict(df[features])

# ---------------------------
# SIDEBAR OPTIONS
# ---------------------------
st.sidebar.header("Controls")

show_data = st.sidebar.checkbox("Show Raw Data")
show_graph = st.sidebar.checkbox("Show Forecast Graph")

# ---------------------------
# RAW DATA
# ---------------------------
if show_data:
    st.subheader("📊 Energy Dataset")
    st.write(df.tail(20))

# ---------------------------
# GRAPH
# ---------------------------
if show_graph:
    st.subheader("📈 Actual vs Predicted Energy")

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(df['date'], df['energy'], label="Actual")
    ax.plot(df['date'], df['prediction'], label="Predicted")

    ax.set_xlabel("Time")
    ax.set_ylabel("Energy Consumption")
    ax.legend()

    st.pyplot(fig)

# ---------------------------
# FUTURE FORECAST BUTTON
# ---------------------------
if st.button("🔮 Predict Next 24 Hours"):

    last = df.iloc[-1].copy()
    future = []

    for i in range(24):
        next_time = df['date'].iloc[-1] + pd.Timedelta(hours=i+1)

        input_data = pd.DataFrame([last[features]])
        pred = model.predict(input_data)[0]

        future.append([next_time, pred])

        last['lag_2'] = last['lag_1']
        last['lag_1'] = pred
        last['rolling_mean_3'] = pred

    future_df = pd.DataFrame(future, columns=['date', 'predicted_energy'])

    st.subheader("🔮 Future Forecast (24 Hours)")
    st.write(future_df)

    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.plot(future_df['date'], future_df['predicted_energy'], color='red')

    st.pyplot(fig2)