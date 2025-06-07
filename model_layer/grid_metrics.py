import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Load predictions and corresponding grid ids
y_true = np.load("data/processed/y_true.npy")
y_pred = np.load("data/processed/y_pred.npy")
grid_ids = np.load("data/model_input/grid_codes.npy")

# Print shapes for debugging
print("Initial shapes:")
print(f"y_true shape: {y_true.shape}")
print(f"y_pred shape: {y_pred.shape}")
print(f"grid_ids shape: {grid_ids.shape}")

# Ensure all arrays have the same shape
y_true = y_true.reshape(-1)
y_pred = y_pred.reshape(-1)
grid_ids = grid_ids.reshape(-1)

# Split grid_ids to match test set
_, grid_ids_test = train_test_split(grid_ids, test_size=0.2, random_state=42)

print("\nAfter reshaping and splitting:")
print(f"y_true shape: {y_true.shape}")
print(f"y_pred shape: {y_pred.shape}")
print(f"grid_ids_test shape: {grid_ids_test.shape}")

# Unique grids
unique_grids = np.unique(grid_ids_test)
print(f"\nNumber of unique grids: {len(unique_grids)}")

# Prepare list to collect metrics
results = []

for grid in unique_grids:
    mask = grid_ids_test == grid
    yt = y_true[mask]
    yp = y_pred[mask]

    if len(yt) == 0:
        continue

    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    # MAPE only if yt > 1 (to avoid % explosion)
    valid = yt > 1
    mape = np.mean(np.abs((yt[valid] - yp[valid]) / yt[valid])) * 100 if np.any(valid) else np.nan

    results.append({
        "Grid ID": int(grid),
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    })

# Convert to DataFrame and sort
df = pd.DataFrame(results)
df_sorted = df.sort_values(by="MAE")

# Save to CSV and print top 10
df_sorted.to_csv("visualizations/grid_metrics.csv", index=False)
print("\nTop 10 grids with lowest MAE:")
print(df_sorted.head(10)) 