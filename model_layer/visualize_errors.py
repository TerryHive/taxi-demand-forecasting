import numpy as np
import pandas as pd
import folium
from folium import plugins
import json

def get_color_from_mae(mae_value, max_mae):
    """Get color based on MAE value (red for high error, green for low)."""
    # Normalize MAE to 0-1 range
    normalized = mae_value / max_mae
    
    # Convert to RGB (green to red)
    r = int(255 * normalized)
    g = int(255 * (1 - normalized))
    b = 0
    
    return f'#{r:02x}{g:02x}{b:02x}'

# Load grid metrics
grid_metrics = pd.read_csv("visualizations/grid_metrics.csv")

# Load grid coordinates
with open('data/processed/grid_coordinates.json', 'r') as f:
    grid_coords = json.load(f)

# Create base map centered on NYC
m = folium.Map(location=[40.7128, -74.0060], zoom_start=12)

# Get max MAE for color scaling
max_mae = grid_metrics['MAE'].max()

# Add markers for each grid
for _, row in grid_metrics.iterrows():
    grid_id = str(row['Grid ID'])
    if grid_id in grid_coords:
        lat, lon = grid_coords[grid_id]
        mae = row['MAE']
        
        # Create circle marker
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            popup=f"Grid {grid_id}<br>MAE: {mae:.2f}<br>MAPE: {row['MAPE']:.2f}%",
            color=get_color_from_mae(mae, max_mae),
            fill=True,
            fill_color=get_color_from_mae(mae, max_mae),
            fill_opacity=0.7
        ).add_to(m)

# Add a color scale legend
legend_html = '''
<div style="position: fixed; 
            bottom: 50px; left: 50px; width: 150px; height: 90px; 
            border:2px solid grey; z-index:9999; font-size:14px;
            background-color:white;
            padding: 10px;
            ">
    <p><strong>MAE Scale</strong></p>
    <p style="color:green;">Low Error</p>
    <p style="color:red;">High Error</p>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Save map
m.save("visualizations/error_heatmap.html")
print("Map saved to visualizations/error_heatmap.html") 