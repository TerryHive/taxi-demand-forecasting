import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation, FFMpegWriter
import folium
from folium import plugins
from pathlib import Path
import json

def parse_grid_id(grid_id):
    """Convert grid_id string to x,y coordinates"""
    x, y = map(int, grid_id.split('_'))
    return x, y

def create_heatmap(data, timestamp, output_path=None):
    """
    Create a heatmap of taxi demand for a specific timestamp
    
    Args:
        data: DataFrame with 'grid_id', 'time_slot', and 'pickup_count'
        timestamp: Specific timestamp to visualize
        output_path: Optional path to save the plot
    """
    # Filter data for specific timestamp
    time_data = data[data['time_slot'] == timestamp].copy()
    
    # Convert grid_ids to coordinates
    time_data['x'] = time_data['grid_id'].apply(lambda x: parse_grid_id(x)[0])
    time_data['y'] = time_data['grid_id'].apply(lambda x: parse_grid_id(x)[1])
    
    # Create pivot table for heatmap
    heatmap_data = time_data.pivot_table(
        values='pickup_count',
        index='y',
        columns='x',
        fill_value=0
    )
    
    # Create plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt='.0f')
    plt.title(f'Taxi Demand Heatmap - {timestamp}')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_time_series(data, grid_id, output_path=None):
    """
    Create a time series plot for a specific grid
    
    Args:
        data: DataFrame with 'grid_id', 'time_slot', and 'pickup_count'
        grid_id: Specific grid to visualize
        output_path: Optional path to save the plot
    """
    # Filter data for specific grid
    grid_data = data[data['grid_id'] == grid_id].copy()
    grid_data['time_slot'] = pd.to_datetime(grid_data['time_slot'])
    grid_data = grid_data.sort_values('time_slot')
    
    # Create plot
    plt.figure(figsize=(15, 6))
    plt.plot(grid_data['time_slot'], grid_data['pickup_count'])
    plt.title(f'Taxi Demand Time Series - Grid {grid_id}')
    plt.xlabel('Time')
    plt.ylabel('Number of Pickups')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def create_animation(data, output_path, fps=4, max_frames=100, show_colorbar=False):
    """
    Create an animation of taxi demand heatmaps over time
    
    Args:
        data: DataFrame with 'grid_id', 'time_slot', and 'pickup_count'
        output_path: Path to save the animation
        fps: Frames per second
        max_frames: Maximum number of frames to include in animation
        show_colorbar: Whether to show a single colorbar for the entire animation
    """
    # Get unique timestamps and limit them
    timestamps = sorted(data['time_slot'].unique())[:max_frames]
    
    # Create figure with tight layout to prevent colorbar issues
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a single colorbar for the entire animation if requested
    if show_colorbar:
        # Get the first frame's data to set up colorbar
        first_data = data[data['time_slot'] == timestamps[0]].copy()
        first_data['x'] = first_data['grid_id'].apply(lambda x: parse_grid_id(x)[0])
        first_data['y'] = first_data['grid_id'].apply(lambda x: parse_grid_id(x)[1])
        first_heatmap = first_data.pivot_table(
            values='pickup_count',
            index='y',
            columns='x',
            fill_value=0
        )
        # Create initial heatmap with colorbar
        sns.heatmap(first_heatmap, cmap='YlOrRd', ax=ax, cbar=True)
        ax.clear()  # Clear it after getting the colorbar
    
    fig.tight_layout()
    
    def update(frame):
        # Clear the axis completely
        ax.clear()
        
        timestamp = timestamps[frame]
        time_data = data[data['time_slot'] == timestamp].copy()
        
        # Convert grid_ids to coordinates
        time_data['x'] = time_data['grid_id'].apply(lambda x: parse_grid_id(x)[0])
        time_data['y'] = time_data['grid_id'].apply(lambda x: parse_grid_id(x)[1])
        
        # Create pivot table
        heatmap_data = time_data.pivot_table(
            values='pickup_count',
            index='y',
            columns='x',
            fill_value=0
        )
        
        # Create heatmap without colorbar to avoid recursion issues
        sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax, cbar=False)
        
        # Add proper labels
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")
        ax.set_title(f'Taxi Demand - {timestamp}')
        
        # Ensure tight layout after each update
        fig.tight_layout()
    
    # Create animation
    anim = FuncAnimation(
        fig, update, frames=len(timestamps),
        interval=1000/fps, repeat=False
    )
    
    # Save animation using Pillow writer (which is built into matplotlib)
    anim.save(str(output_path), writer='pillow', fps=fps)
    plt.close()

def create_interactive_map(data, timestamp, output_path):
    """
    Create an interactive map with taxi demand heatmap
    
    Args:
        data: DataFrame with 'grid_id', 'time_slot', and 'pickup_count'
        timestamp: Specific timestamp to visualize
        output_path: Path to save the HTML map
    """
    # Load NYC coordinates mapping
    with open('data/processed/grid_coordinates.json', 'r') as f:
        grid_coords = json.load(f)
    
    # Filter data for specific timestamp
    time_data = data[data['time_slot'] == timestamp].copy()
    
    # Create base map centered on NYC
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=12)
    
    # Prepare heatmap data
    heat_data = []
    for _, row in time_data.iterrows():
        grid_id = row['grid_id']
        if grid_id in grid_coords:
            lat, lon = grid_coords[grid_id]
            heat_data.append([lat, lon, row['pickup_count']])
    
    # Add heatmap layer
    plugins.HeatMap(heat_data).add_to(m)
    
    # Save map
    m.save(output_path)

def plot_predictions(actual, predicted, grid_id, output_path=None):
    """
    Plot actual vs predicted values for a specific grid
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        grid_id: Grid ID for the plot title
        output_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual', marker='o')
    plt.plot(predicted, label='Predicted', marker='x')
    plt.title(f'Actual vs Predicted Taxi Demand - Grid {grid_id}')
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Pickups')
    plt.legend()
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def main():
    # Create output directory
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    data = pd.read_csv("data/processed/aggregated_pickups.csv")
    
    # Example timestamp
    timestamp = "2015-01-01 08:00:00"
    
    # Create visualizations
    create_heatmap(data, timestamp, output_dir / "heatmap.png")
    plot_time_series(data, "15_26", output_dir / "time_series.png")
    create_animation(data, output_dir / "demand_animation.gif")
    create_interactive_map(data, timestamp, output_dir / "interactive_map.html")

if __name__ == "__main__":
    main() 