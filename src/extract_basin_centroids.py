#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# Extract Basin Centroids

This script extracts centroid coordinates from basin geometry files and finds 
the nearest NOAA GEFS grid points for use in forecast data extraction.
"""

# Import required libraries
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from shapely.geometry import Point
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', message='The specified chunks')


def load_basin_boundaries(dir_path):
    """Load basin boundary geometries from all shapefiles in a directory.
    
    Args:
        dir_path (str): Path to the directory containing shapefiles with basin boundaries
        
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame containing basin geometries from all shapefiles
    """
    try:
        # Check if directory exists
        if not os.path.isdir(dir_path):
            print(f"Error: Directory {dir_path} not found.")
            return None
        
        # Get all shapefile paths in the directory
        shapefile_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) 
                          if f.lower().endswith('.shp')]
        
        if not shapefile_paths:
            print(f"Error: No shapefiles found in {dir_path}")
            return None
        
        print(f"Found {len(shapefile_paths)} shapefiles in directory")
        
        # List to store GeoDataFrames from each shapefile
        all_basins = []
        
        # Process each shapefile
        for file_path in shapefile_paths:
            print(f"Loading {os.path.basename(file_path)}...")
            # Load the shapefile with geopandas
            basins = gpd.read_file(file_path)
            
            # Check if the data was loaded correctly
            if basins.empty:
                print(f"  Warning: Shapefile {os.path.basename(file_path)} contains no features")
                continue
                
            # Ensure we have a basin name column
            # Look for common basin name field variations
            name_fields = ['NAME', 'Name', 'name', 'BASIN_NAME', 'BasinName', 'ID', 'basin_id']
            basin_name_field = None
            
            for field in name_fields:
                if field in basins.columns:
                    basin_name_field = field
                    break
            
            # If no basin name field is found, create a default one
            if basin_name_field is None:
                print(f"  No basin name field found in {os.path.basename(file_path)}. Using index as basin ID.")
                # Add filename prefix to make basin IDs unique across files
                file_prefix = os.path.splitext(os.path.basename(file_path))[0]
                basins['basin_name'] = [f"{file_prefix}_Basin_{i}" for i in range(len(basins))]
            else:
                # Rename the field to a standard name
                basins = basins.rename(columns={basin_name_field: 'basin_name'})
                # Add filename prefix to basin names to ensure uniqueness across files
                file_prefix = os.path.splitext(os.path.basename(file_path))[0]
                basins['basin_name'] = file_prefix + '_' + basins['basin_name'].astype(str)
            
            print(f"  Added {len(basins)} basins from {os.path.basename(file_path)}")
            all_basins.append(basins)
        
        if not all_basins:
            print("No valid basin data found in any shapefile.")
            return None
            
        # Combine all GeoDataFrames into one
        combined_basins = gpd.GeoDataFrame(pd.concat(all_basins, ignore_index=True))
        
        # Ensure combined basin names are unique
        if combined_basins['basin_name'].duplicated().any():
            print("Warning: Duplicate basin names found after combining. Appending index to make them unique.")
            mask = combined_basins['basin_name'].duplicated(keep=False)
            dup_indices = [str(i) for i in range(sum(mask))]
            combined_basins.loc[mask, 'basin_name'] = combined_basins.loc[mask, 'basin_name'] + '_' + dup_indices

        print(f"Loaded a total of {len(combined_basins)} basin boundaries from all shapefiles")
        return combined_basins
    
    except Exception as e:
        print(f"Error loading basin boundaries: {e}")
        return None


def calculate_basin_centroids(basins):
    """Calculate centroids for all basin geometries.
    
    Args:
        basins (geopandas.GeoDataFrame): GeoDataFrame containing basin geometries
        
    Returns:
        pandas.DataFrame: DataFrame containing basin names and centroid coordinates
    """
    if basins is None:
        return None
    
    # Create a DataFrame to store basin centroids
    centroids = pd.DataFrame({
        'basin_name': basins['basin_name'],
        'centroid_lat': [geom.centroid.y for geom in basins.geometry],
        'centroid_lon': [geom.centroid.x for geom in basins.geometry]
    })
    
    print(f"Calculated centroids for {len(centroids)} basins")
    return centroids


def plot_basins_and_centroids(basins, centroids):
    """Plot basin boundaries and their centroids."""
    if centroids is None:
        return
        
    # Display the centroids
    print(centroids.head())
    
    # Plot the basins and their centroids
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot basins
    if basins is not None:
        basins.plot(ax=ax, color='lightblue', edgecolor='gray', alpha=0.5)
    
    # Plot centroids
    ax.scatter(centroids['centroid_lon'], centroids['centroid_lat'], 
               color='red', marker='o', s=20, label='Centroids')
    
    # Add labels for a few centroids (not all to avoid clutter)
    for idx, row in centroids.sample(min(5, len(centroids))).iterrows():
        ax.annotate(row['basin_name'], 
                    xy=(row['centroid_lon'], row['centroid_lat']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title("Basin Boundaries and Centroids")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def connect_to_gefs_dataset(email="optional@email.com"):
    """Connect to the NOAA GEFS dataset and return the grid coordinates.
    
    Args:
        email (str, optional): Email address for tracking dataset usage.
        
    Returns:
        tuple: (lat_array, lon_array) arrays of grid point coordinates
    """
    zarr_url = f"https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr?email={email}"
    
    try:
        # Open the dataset with just coordinates to minimize data transfer
        print("Connecting to NOAA GEFS dataset to get grid coordinates...")
        ds = xr.open_zarr(zarr_url, decode_timedelta=True, consolidated=True, chunks={})
        
        # Extract just the coordinate variables
        lats = ds.latitude.values
        lons = ds.longitude.values
        
        print(f"GEFS dataset grid: {len(lats)} latitudes × {len(lons)} longitudes")
        print(f"Latitude range: {lats.min()} to {lats.max()}")
        print(f"Longitude range: {lons.min()} to {lons.max()}")
        
        return lats, lons
    except Exception as e:
        print(f"Error connecting to NOAA GEFS dataset: {e}")
        return None, None


def find_nearest_grid_points(centroids, lats, lons):
    """Find the nearest GEFS grid points to each basin centroid.
    
    Args:
        centroids (pandas.DataFrame): DataFrame with basin centroid coordinates
        lats (numpy.ndarray): Array of GEFS grid latitudes
        lons (numpy.ndarray): Array of GEFS grid longitudes
        
    Returns:
        pandas.DataFrame: DataFrame with basin centroids and nearest grid points
    """
    if centroids is None or lats is None or lons is None:
        return centroids
    
    # Create columns for the nearest grid points
    centroids['nearest_lat'] = None
    centroids['nearest_lon'] = None
    centroids['grid_distance_km'] = None
    
    # Function to calculate the great circle distance between two points
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate the great circle distance between two points in kilometers."""
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        return c * r
    
    # For each centroid, find the nearest grid point
    for idx, row in centroids.iterrows():
        centroid_lat = row['centroid_lat']
        centroid_lon = row['centroid_lon']
        
        # Find the nearest latitude and longitude indices
        lat_idx = np.abs(lats - centroid_lat).argmin()
        lon_idx = np.abs(lons - centroid_lon).argmin()
        
        # Get the actual latitude and longitude values at these indices
        nearest_lat = lats[lat_idx]
        nearest_lon = lons[lon_idx]
        
        # Calculate the distance between the centroid and the nearest grid point
        distance = haversine_distance(centroid_lat, centroid_lon, nearest_lat, nearest_lon)
        
        # Store the results
        centroids.at[idx, 'nearest_lat'] = nearest_lat
        centroids.at[idx, 'nearest_lon'] = nearest_lon
        centroids.at[idx, 'grid_distance_km'] = distance
    
    print(f"Found nearest grid points for {len(centroids)} basin centroids")
    print(f"Average distance to nearest grid point: {centroids['grid_distance_km'].mean():.2f} km")
    print(f"Maximum distance to nearest grid point: {centroids['grid_distance_km'].max():.2f} km")
    
    return centroids


def plot_grid_points_analysis(centroids_with_grid):
    """Plot analysis of grid points and their distances."""
    if centroids_with_grid is None:
        return
        
    # Display the results
    print(centroids_with_grid.head())
    
    # Plot a histogram of distances to nearest grid point
    plt.figure(figsize=(10, 6))
    plt.hist(centroids_with_grid['grid_distance_km'], bins=20, alpha=0.7, color='blue')
    plt.axvline(centroids_with_grid['grid_distance_km'].mean(), color='red', linestyle='dashed', linewidth=2, 
                label=f'Mean: {centroids_with_grid["grid_distance_km"].mean():.2f} km')
    plt.title('Distribution of Distances to Nearest GEFS Grid Point')
    plt.xlabel('Distance (km)')
    plt.ylabel('Number of Basin Centroids')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Plot the centroids and their nearest grid points
    plt.figure(figsize=(12, 8))
    
    # Plot centroids
    plt.scatter(centroids_with_grid['centroid_lon'], centroids_with_grid['centroid_lat'], 
                color='blue', marker='o', s=30, alpha=0.7, label='Basin Centroids')
    
    # Plot nearest grid points
    plt.scatter(centroids_with_grid['nearest_lon'], centroids_with_grid['nearest_lat'], 
                color='red', marker='x', s=30, alpha=0.7, label='Nearest GEFS Grid Points')
    
    # Draw lines between centroids and nearest grid points
    for idx, row in centroids_with_grid.sample(min(20, len(centroids_with_grid))).iterrows():
        plt.plot([row['centroid_lon'], row['nearest_lon']], 
                 [row['centroid_lat'], row['nearest_lat']], 
                 'k-', alpha=0.3, linewidth=0.5)
    
    plt.title('Basin Centroids and Their Nearest GEFS Grid Points')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def save_to_csv(centroids_with_grid, output_file):
    """Save the basin centroids with grid points to a CSV file.
    
    Args:
        centroids_with_grid (pandas.DataFrame): DataFrame with basin centroid and grid point data
        output_file (str): Path to the output CSV file
    """
    if centroids_with_grid is None:
        print("No data to save.")
        return
    
    try:
        # Save the DataFrame to CSV
        centroids_with_grid.to_csv(output_file, index=False)
        print(f"Saved {len(centroids_with_grid)} basin centroids with grid points to {output_file}")
    except Exception as e:
        print(f"Error saving to CSV: {e}")


def main():
    """Main function to run the basin centroid extraction process."""
    # Define paths for input and output data
    basin_dir_path = "../data/basin_shapefiles"  # Directory containing basin boundary shapefiles
    output_dir = "../data"
    output_file = os.path.join(output_dir, "basin_centroids_with_grid_points.csv")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Basin directory path: {basin_dir_path}")
    print(f"Output will be saved to: {output_file}")
    
    # Load basin boundary data from all shapefiles in the directory
    basins = load_basin_boundaries(basin_dir_path)
    
    if basins is not None:
        # Calculate basin centroids
        centroids = calculate_basin_centroids(basins)
        
        # Plot basins and centroids
        plot_basins_and_centroids(basins, centroids)
        
        # Connect to the GEFS dataset to get grid coordinates
        lats, lons = connect_to_gefs_dataset("your.email@example.com")
        
        # Find the nearest GEFS grid points to each basin centroid
        centroids_with_grid = find_nearest_grid_points(centroids, lats, lons)
        
        # Plot analysis of grid points
        plot_grid_points_analysis(centroids_with_grid)
        
        # Save the basin centroids with grid points to CSV
        save_to_csv(centroids_with_grid, output_file)
        
        print("\nConclusion:")
        print("This script has:")
        print("1. Loaded basin boundary geometries from shapefiles in a directory")
        print("2. Calculated centroid coordinates for each basin")
        print("3. Connected to the NOAA GEFS dataset to get grid coordinates")
        print("4. Found the nearest GEFS grid point to each basin centroid")
        print("5. Saved the basin centroids with grid points to a CSV file")
        print(f"\nThe output file {output_file} will be used by fetch_basin_forecasts.py to retrieve forecast data for each basin.")
    else:
        print("Failed to load basin data. Please check the input directory and try again.")


if __name__ == "__main__":
    main()
