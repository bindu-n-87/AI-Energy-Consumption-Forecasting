import pandas as pd

# Load dataset
df = pd.read_csv("../data/energy.csv")

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Show data
print("Dataset Preview:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())