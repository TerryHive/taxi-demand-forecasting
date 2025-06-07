import pandas as pd
import numpy as np
from pathlib import Path

def aggregate_pickups(df: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
    """
    Aggregate the number of pickups per grid_id and time interval.
    
    Args:
        df (pd.DataFrame): Input DataFrame with pickup datetime and grid_id
        freq (str): Time frequency for aggregation (default '15min')
        
    Returns:
        pd.DataFrame: Aggregated pickup counts per grid_id per time slot
    """
    # 1. Round timestamps to nearest 15-minute slot
    df['time_slot'] = df['tpep_pickup_datetime'].dt.floor(freq)

    # 2. Group by time_slot and grid_id
    grouped = df.groupby(['time_slot', 'grid_id']).size().reset_index(name='pickup_count')

    return grouped

def main():
    # Load the gridded data
    input_file = Path("data/processed/gridded_taxi_data.csv")
    
    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        print("Please run grid_processor.py first to generate the gridded data.")
        return
    
    # Read the gridded data
    df = pd.read_csv(input_file)
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    
    # Apply time aggregation
    aggregated_df = aggregate_pickups(df, freq="15min")
    
    # Display sample of the aggregated data
    print("\nSample of aggregated data:")
    print(aggregated_df.head())
    
    # Save the aggregated data
    output_file = Path("data/processed/aggregated_pickups.csv")
    aggregated_df.to_csv(output_file, index=False)
    print(f"\nAggregated data saved to: {output_file}")
    
    # Print aggregation statistics
    print("\nAggregation Statistics:")
    print(f"Total time slots: {aggregated_df['time_slot'].nunique()}")
    print(f"Total grid cells: {aggregated_df['grid_id'].nunique()}")
    print(f"Total records: {len(aggregated_df)}")
    print("\nTime range:")
    print(f"From: {aggregated_df['time_slot'].min()}")
    print(f"To: {aggregated_df['time_slot'].max()}")
    print("\nPickup count statistics:")
    print(aggregated_df['pickup_count'].describe())

if __name__ == "__main__":
    main() 