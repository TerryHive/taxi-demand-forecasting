import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def create_sliding_windows(data, window_size=6, horizon=1):
    """
    Create sliding windows for time series prediction.
    
    Args:
        data: DataFrame with 'grid_id', 'grid_code' and 'pickup_count' columns
        window_size: Number of time steps to use as input
        horizon: Number of time steps to predict ahead
        
    Returns:
        X: Input sequences (n_samples, window_size, n_features)
        y: Target values (n_samples,)
        grid_codes: Grid codes for each sequence (n_samples,)
    """
    # Group by grid_id to create sequences for each location
    sequences = []
    targets = []
    grid_codes = []
    
    for grid_id in data['grid_id'].unique():
        grid_data = data[data['grid_id'] == grid_id]
        grid_code = grid_data['grid_code'].iloc[0]
        pickup_counts = grid_data['pickup_count'].values
        
        # Create sequences
        for i in range(len(pickup_counts) - window_size - horizon + 1):
            sequences.append(pickup_counts[i:i + window_size])
            targets.append(pickup_counts[i + window_size + horizon - 1])
            grid_codes.append(grid_code)
    
    return np.array(sequences), np.array(targets), np.array(grid_codes)

def normalize_data(X, y):
    """
    Normalize data using Z-score normalization.
    
    Args:
        X: Input sequences
        y: Target values
        
    Returns:
        X_norm: Normalized input sequences
        y_norm: Normalized target values
        norm_params: Dictionary containing mean and std for inverse normalization
    """
    # Calculate mean and std for X (using all values)
    X_mean = X.mean()
    X_std = X.std()
    
    # Calculate mean and std for y
    y_mean = y.mean()
    y_std = y.std()
    
    # Normalize
    X_norm = (X - X_mean) / X_std
    y_norm = (y - y_mean) / y_std
    
    # Store normalization parameters
    norm_params = {
        'X_mean': X_mean,
        'X_std': X_std,
        'y_mean': y_mean,
        'y_std': y_std
    }
    
    return X_norm, y_norm, norm_params

def main():
    # Create output directory
    output_dir = Path("data/model_input")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load aggregated data
    print("Loading aggregated data...")
    data = pd.read_csv("data/processed/aggregated_pickups.csv")
    
    # Encode grid_ids
    print("Encoding grid IDs...")
    encoder = LabelEncoder()
    data['grid_code'] = encoder.fit_transform(data['grid_id'])
    
    # Save grid encoder classes
    np.save(output_dir / "grid_encoder_classes.npy", encoder.classes_)
    
    # Create sequences
    print("Creating sequences...")
    X, y, grid_codes = create_sliding_windows(data)
    
    # Normalize data
    print("Normalizing data...")
    X_norm, y_norm, norm_params = normalize_data(X, y)
    
    # Save normalized data
    print("Saving normalized data...")
    np.save(output_dir / "X.npy", X_norm)
    np.save(output_dir / "y.npy", y_norm)
    np.save(output_dir / "grid_codes.npy", grid_codes)
    
    # Save normalization parameters
    print("Saving normalization parameters...")
    np.save(output_dir / "norm_params.npy", norm_params)
    
    print("\nDataset generation complete!")
    print(f"Input shape: {X_norm.shape}")
    print(f"Target shape: {y_norm.shape}")
    print(f"Grid codes shape: {grid_codes.shape}")
    print(f"\nNormalization parameters:")
    print(f"X mean: {norm_params['X_mean']:.2f}")
    print(f"X std: {norm_params['X_std']:.2f}")
    print(f"y mean: {norm_params['y_mean']:.2f}")
    print(f"y std: {norm_params['y_std']:.2f}")

if __name__ == "__main__":
    main() 