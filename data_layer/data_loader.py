import pandas as pd
import numpy as np
from pathlib import Path

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Load and clean taxi trip data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with relevant columns
    """
    # 1.1 Load specific columns for speed
    cols = ['tpep_pickup_datetime', 'pickup_longitude', 'pickup_latitude']
    df = pd.read_csv(file_path, usecols=cols)
    
    # 1.2 Convert time
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    
    # 1.3 Clean geographical outliers
    # NYC approximately: long [-74.05, -73.75], lat [40.63, 40.85]
    df = df[
        (df['pickup_longitude'] > -74.05) & (df['pickup_longitude'] < -73.75) &
        (df['pickup_latitude'] > 40.63) & (df['pickup_latitude'] < 40.85)
    ]
    
    return df

def main():
    # Example usage with the correct path
    data_file = Path("data/raw/yellow_tripdata_2015-01.csv")
    
    if not data_file.exists():
        print(f"Error: {data_file} not found!")
        print("Please make sure the file is in the data/raw directory.")
        return
    
    # Load and clean data
    df = load_and_clean_data(str(data_file))
    
    # 1.4 Quick look at the data
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nDataFrame Info:")
    print(df.info())
    
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Save processed data
    output_file = Path("data/processed/cleaned_taxi_data.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to: {output_file}")

if __name__ == "__main__":
    main() 