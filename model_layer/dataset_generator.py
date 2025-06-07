import pandas as pd
import numpy as np
from pathlib import Path

def generate_sliding_window(df: pd.DataFrame, window_size: int = 4) -> tuple:
    """
    Generate sliding window sequences for supervised ML.
    
    Args:
        df (pd.DataFrame): Aggregated pickup data with time_slot, grid_id, pickup_count
        window_size (int): Number of time steps to use as input
    
    Returns:
        tuple: (X, y) arrays ready for training
    """
    sequences = []
    labels = []

    # Sort by grid and time
    df = df.sort_values(by=['grid_id', 'time_slot'])

    # Group by grid_id
    for grid, group in df.groupby('grid_id'):
        counts = group['pickup_count'].values
        
        if len(counts) > window_size:
            for i in range(len(counts) - window_size):
                x = counts[i:i+window_size]
                y = counts[i+window_size]
                sequences.append(x)
                labels.append(y)

    X = np.array(sequences)
    y = np.array(labels)
    
    return X, y

def main():
    # Create model input directory if it doesn't exist
    model_input_dir = Path("data/model_input")
    model_input_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the aggregated data
    input_file = Path("data/processed/aggregated_pickups.csv")
    
    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        print("Please run time_aggregator.py first to generate the aggregated data.")
        return
    
    # Read the aggregated data
    df = pd.read_csv(input_file)
    df['time_slot'] = pd.to_datetime(df['time_slot'])
    
    # Generate sliding window sequences
    X, y = generate_sliding_window(df, window_size=4)
    
    # Print dataset information
    print("\nDataset Information:")
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
    print("\nExample sequences:")
    print("X[0]:", X[0])
    print("y[0]:", y[0])
    
    # Save the training data
    np.save(model_input_dir / "X.npy", X)
    np.save(model_input_dir / "y.npy", y)
    print(f"\nTraining data saved to: {model_input_dir}")
    
    # Print some statistics
    print("\nData Statistics:")
    print("X mean:", X.mean())
    print("X std:", X.std())
    print("y mean:", y.mean())
    print("y std:", y.std())
    
    # Print grid statistics
    print("\nGrid Statistics:")
    print(f"Number of grids with sequences: {df['grid_id'].nunique()}")
    print(f"Average sequences per grid: {len(X) / df['grid_id'].nunique():.2f}")

if __name__ == "__main__":
    main() 