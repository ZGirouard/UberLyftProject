import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_explore_data(filepath):
    print(f"Loading data from {filepath}...")

    df = pd.read_csv(filepath)

    print(f"\nDataset shape: {df.shape}")
    print(f"Number of rows: {df.shape[0]:,}")
    print(f"Number of columns: {df.shape[1]}")

    print("-" * 50)
    print("COLUMN NAMES AND TYPES")
    print(df.dtypes)
    print("-" * 50)
    print("FIRST FEW ROWS")
    print(df.head())
    print("-" * 50)
    print("BASIC STATISTICS")
    print(df.describe())
    print("-" * 50)
    print("MISSING VALUES")
    print(df.isnull().sum())
    print("-" * 50)
    print("TARGET VARIABLE (PRICE) ANALYSIS")
    if "price" in df.columns:
        print(f"Price summary:")
        print(f"Mean: {df['price'].mean():.2f}")
        print(f"Median: {df['price'].median():.2f}")
        print(f"Standard Deviation: {df['price'].std():.2f}")
    else:
        print("No price column found in the dataset.")

    print("-" * 50)
    print("CATEGORICAL VARIABLES")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\n{col}: {df[col].nunique()} unique values")
        print(df[col].value_counts().head(10))
    
    return df

if __name__ == "__main__":
    data_path = "data/cab_rides.csv"
    df = load_and_explore_data(data_path)
    