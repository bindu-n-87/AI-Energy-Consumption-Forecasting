import pandas as pd

def load_and_clean_data(file_path):

    df = pd.read_csv(file_path)

    print("Raw Data Preview:")
    print(df.head())

    # Convert datetime
    df['date'] = pd.to_datetime(df['date'])

    # Sort time series
    df = df.sort_values('date')

    print("\nMissing Values Before Cleaning:")
    print(df.isnull().sum())

    # FIXED forward fill
    df['energy'] = df['energy'].ffill()

    print("\nData Cleaning Completed!")
    print(df.head())

    return df

if __name__ == "__main__":
    data_path = "../data/energy.csv"   # IMPORTANT FIX
    cleaned_df = load_and_clean_data(data_path)
