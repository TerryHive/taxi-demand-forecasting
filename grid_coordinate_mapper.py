import json
import numpy as np
from pathlib import Path

# NYC bounding box coordinates
NYC_BOUNDS = {
    'min_lat': 40.4961,  # Southernmost latitude
    'max_lat': 40.9176,  # Northernmost latitude
    'min_lon': -74.2556, # Westernmost longitude
    'max_lon': -73.7004  # Easternmost longitude
}

def calculate_grid_center(grid_id, grid_size=(30, 30)):
    """
    Calculate the center coordinates of a grid cell
    
    Args:
        grid_id: String in format "x_y"
        grid_size: Tuple of (num_x_cells, num_y_cells)
        
    Returns:
        Tuple of (latitude, longitude) for the center of the grid cell
    """
    # Parse grid coordinates
    x, y = map(int, grid_id.split('_'))
    
    # Calculate cell size
    lat_step = (NYC_BOUNDS['max_lat'] - NYC_BOUNDS['min_lat']) / grid_size[1]
    lon_step = (NYC_BOUNDS['max_lon'] - NYC_BOUNDS['min_lon']) / grid_size[0]
    
    # Calculate cell center
    lat = NYC_BOUNDS['min_lat'] + (y + 0.5) * lat_step
    lon = NYC_BOUNDS['min_lon'] + (x + 0.5) * lon_step
    
    return lat, lon

def main():
    # Create output directory if it doesn't exist
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load aggregated data to get all grid IDs
    try:
        import pandas as pd
        data = pd.read_csv("data/processed/aggregated_pickups.csv")
        grid_ids = data['grid_id'].unique()
    except FileNotFoundError:
        # If file doesn't exist, create a 30x30 grid
        grid_ids = [f"{x}_{y}" for x in range(30) for y in range(30)]
    
    # Calculate coordinates for each grid cell
    grid_coordinates = {}
    for grid_id in grid_ids:
        lat, lon = calculate_grid_center(grid_id)
        grid_coordinates[grid_id] = [float(lat), float(lon)]
    
    # Save to JSON file
    output_path = output_dir / "grid_coordinates.json"
    with open(output_path, 'w') as f:
        json.dump(grid_coordinates, f, indent=2)
    
    print(f"Grid coordinates saved to {output_path}")
    print(f"Total grid cells mapped: {len(grid_coordinates)}")

if __name__ == "__main__":
    main() 