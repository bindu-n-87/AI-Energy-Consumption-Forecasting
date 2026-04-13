import pandas as pd
import numpy as np
import os

# ---------------------------
# CREATE DATA FOLDER SAFELY
# ---------------------------
os.makedirs("data", exist_ok=True)

# ---------------------------
# GENERATE TIME SERIES
# ---------------------------
dates = pd.date_range(start="2025-01-01", periods=720, freq="h")

np.random.seed(42)

energy = (
    30
    + 10 * np.sin(np.arange(720) * 2 * np.pi / 24)
    + np.random.normal(0, 2, 720)
)

df = pd.DataFrame({
    "date": dates,
    "energy": energy
})

# ---------------------------
# SAVE FILE SAFELY
# ---------------------------
file_path = os.path.join("data", "energy.csv")
df.to_csv(file_path, index=False)

print("Dataset created successfully at:", file_path)