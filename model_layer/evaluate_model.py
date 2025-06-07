import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load predictions
y_true = np.load("data/processed/y_true.npy")
y_pred = np.load("data/processed/y_pred.npy")

# Reshape arrays to 1D if they are not already
y_true = y_true.reshape(-1)
y_pred = y_pred.reshape(-1)

# Calculate MAE and RMSE
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# Calculate MAPE only for points where y_true > 1
mask = y_true > 1  # ignore points with almost zero pickups
mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Display results
print(f"Evaluation Metrics (Overall):")
print(f"MAE  = {mae:.2f}")
print(f"RMSE = {rmse:.2f}")
print(f"MAPE = {mape:.2f}% (calculated on {np.sum(mask)} points where true value > 1)")

# Plot predictions for grid 15_26
plt.figure(figsize=(12, 6))
plt.plot(y_true, label='Actual', marker='o', markersize=2)
plt.plot(y_pred, label='Predicted', marker='x', markersize=2)
plt.title('Actual vs Predicted Taxi Demand - Grid 15_26')
plt.xlabel('Time Steps')
plt.ylabel('Number of Pickups')
plt.legend()
plt.grid(True)
plt.savefig('visualizations/grid_15_26_predictions.png')
plt.close() 