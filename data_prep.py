import pandas as pd

def load_and_clean(path):
    df = pd.read_csv(path)
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    print("✅ Data loaded and cleaned successfully!")
    print(f"Shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print("\nSample data:")
    print(df.head())
    return df

if __name__ == "__main__":
    data_path = "./data/Telco-Customer-Churn.csv"
    df = load_and_clean(data_path)
