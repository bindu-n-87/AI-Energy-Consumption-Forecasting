import pandas as pd

def create_features(df):
    """
    Feature engineering for energy forecasting
    """

    # Time-based features
    df['hour'] = df['date'].dt.hour
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month

    # Lag features (VERY IMPORTANT)
    df['lag_1'] = df['energy'].shift(1)
    df['lag_2'] = df['energy'].shift(2)

    # Rolling mean (trend detection)
    df['rolling_mean_3'] = df['energy'].rolling(window=3).mean()

    # Drop NaN values created by lag/rolling
    df = df.dropna()

    print("\nFeature Engineering Completed!")
    print(df.head())

    return df


# Test run
if __name__ == "__main__":
    df = pd.read_csv("../data/energy.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    df_features = create_features(df)