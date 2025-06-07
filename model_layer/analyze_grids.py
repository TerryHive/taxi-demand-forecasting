import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def plot_predictions(y_true, y_pred, grid_id, save_path=None):
    """Plot actual vs predicted values for a specific grid."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', marker='o', markersize=2)
    plt.plot(y_pred, label='Predicted', marker='x', markersize=2)
    plt.title(f'Actual vs Predicted Taxi Demand - Grid {grid_id}')
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Pickups')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# Load data
y_true = np.load("data/processed/y_true.npy")
y_pred = np.load("data/processed/y_pred.npy")
grid_ids = np.load("data/model_input/grid_codes.npy")

# Reshape arrays
y_true = y_true.reshape(-1)
y_pred = y_pred.reshape(-1)
grid_ids = grid_ids.reshape(-1)

# Split grid_ids to match test set
_, grid_ids_test = train_test_split(grid_ids, test_size=0.2, random_state=42)

# Load grid metrics
grid_metrics = pd.read_csv("visualizations/grid_metrics.csv")

# Find grids with high activity (high MAE/RMSE)
print("\nGrids with high activity (sorted by MAE):")
high_activity_grids = grid_metrics.sort_values('MAE', ascending=False)
print(high_activity_grids.head(10))

# Select grids to analyze based on high activity
target_grids = [79, 1120, 78, 202]  # Grids with high MAE

# Initialize scaler and fit on all true values
scaler = MinMaxScaler()
scaler.fit(y_true.reshape(-1, 1))

for grid_id in target_grids:
    mask = grid_ids_test == grid_id
    if np.any(mask):
        yt = y_true[mask]
        yp = y_pred[mask]
        
        # Reverse scaling to get actual pickup values
        yt = scaler.inverse_transform(yt.reshape(-1, 1)).reshape(-1)
        yp = scaler.inverse_transform(yp.reshape(-1, 1)).reshape(-1)
        
        # Check if we have meaningful data
        if np.max(yt) < 20:  # Skip if max pickups < 20
            print(f"\nSkipping Grid {grid_id}: Max pickups = {np.max(yt):.2f}")
            continue
        
        # Get metrics for this grid
        grid_metric = grid_metrics[grid_metrics['Grid ID'] == grid_id].iloc[0]
        print(f"\nMetrics for Grid {grid_id}:")
        print(f"MAE: {grid_metric['MAE']:.2f}")
        print(f"RMSE: {grid_metric['RMSE']:.2f}")
        print(f"MAPE: {grid_metric['MAPE']:.2f}%")
        print(f"Max pickups: {np.max(yt):.2f}")
        print(f"Min pickups: {np.min(yt):.2f}")
        print(f"Mean pickups: {np.mean(yt):.2f}")
        
        # Plot predictions
        plot_predictions(
            yt, yp, 
            grid_id=grid_id,
            save_path=f"visualizations/grid_{grid_id}_predictions.png"
        ) 