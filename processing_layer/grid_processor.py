import pandas as pd
import numpy as np
from pathlib import Path

def apply_spatial_grid(df: pd.DataFrame, cell_size: float = 0.005) -> pd.DataFrame:
    """
    Assign grid coordinates to GPS points based on fixed cell size.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame with GPS points
        cell_size (float): Grid cell size in degrees (default 0.005)
        
    Returns:
        pd.DataFrame: Original DataFrame with added 'grid_x', 'grid_y', and 'grid_id'
    """
    # 1. Define grid bounds
    min_lon, max_lon = -74.05, -73.75
    min_lat, max_lat = 40.63, 40.85

    # 2. Compute grid_x and grid_y as integer indices
    df['grid_x'] = ((df['pickup_longitude'] - min_lon) / cell_size).astype(int)
    df['grid_y'] = ((df['pickup_latitude'] - min_lat) / cell_size).astype(int)
    
    # 3. Validate grid coordinates
    assert (df['grid_x'] >= 0).all() and (df['grid_y'] >= 0).all(), "Negative grid coordinates detected!"
    
    # 4. Create grid_id for easier reference
    df['grid_id'] = df['grid_x'].astype(str) + "_" + df['grid_y'].astype(str)
    
    return df

def main():
    # Load the cleaned data
    input_file = Path("data/processed/cleaned_taxi_data.csv")
    
    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        print("Please run data_loader.py first to generate the cleaned data.")
        return
    
    # Read the cleaned data
    df = pd.read_csv(input_file)
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    
    # Apply spatial gridding
    df = apply_spatial_grid(df, cell_size=0.005)
    
    # Display sample of the gridded data
    print("\nSample of gridded data:")
    print(df[['pickup_longitude', 'pickup_latitude', 'grid_x', 'grid_y', 'grid_id']].head())
    
    # Save the gridded data
    output_file = Path("data/processed/gridded_taxi_data.csv")
    df.to_csv(output_file, index=False)
    print(f"\nGridded data saved to: {output_file}")
    
    # Print grid statistics
    print("\nGrid Statistics:")
    print(f"Number of unique grid cells: {df[['grid_x', 'grid_y']].drop_duplicates().shape[0]}")
    print("\nGrid dimensions:")
    print(f"X range: {df['grid_x'].min()} to {df['grid_x'].max()}")
    print(f"Y range: {df['grid_y'].min()} to {df['grid_y'].max()}")
    
    # Print some grid_id statistics
    print("\nGrid ID Statistics:")
    print(f"Total unique grid IDs: {df['grid_id'].nunique()}")
    print("\nMost common grid cells (top 5):")
    print(df['grid_id'].value_counts().head())

if __name__ == "__main__":
    main()
