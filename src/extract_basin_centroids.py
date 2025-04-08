#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# Extract Basin Centroids

This script extracts centroid coordinates from basin geometry files and saves
them for later use in forecast data processing.
"""

# Import required libraries
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


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
        'latitude': [geom.centroid.y for geom in basins.geometry],
        'longitude': [geom.centroid.x for geom in basins.geometry]
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
    ax.scatter(centroids['longitude'], centroids['latitude'], 
               color='red', marker='o', s=20, label='Centroids')
    
    # Add labels for a few centroids (not all to avoid clutter)
    for idx, row in centroids.sample(min(5, len(centroids))).iterrows():
        ax.annotate(row['basin_name'], 
                    xy=(row['longitude'], row['latitude']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title("Basin Boundaries and Centroids")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def save_to_csv(centroids, output_file):
    """Save the basin centroids to a CSV file.
    
    Args:
        centroids (pandas.DataFrame): DataFrame with basin centroid data
        output_file (str): Path to the output CSV file
    """
    if centroids is None:
        print("No data to save.")
        return
    
    try:
        # Save the DataFrame to CSV
        centroids.to_csv(output_file, index=False)
        print(f"Saved {len(centroids)} basin centroids to {output_file}")
    except Exception as e:
        print(f"Error saving to CSV: {e}")


def main():
    """Main function to run the basin centroid extraction process."""
    # Define paths for input and output data
    basin_dir_path = "../data/basin_shapefiles"  # Directory containing basin boundary shapefiles
    output_dir = "../data"
    output_file = os.path.join(output_dir, "basin_centroids.csv")
    
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
        
        # Save the basin centroids to CSV
        save_to_csv(centroids, output_file)
        
        print("\nConclusion:")
        print("This script has:")
        print("1. Loaded basin boundary geometries from shapefiles in a directory")
        print("2. Calculated centroid coordinates for each basin")
        print("3. Saved the basin centroids to a CSV file")
        print(f"\nThe output file {output_file} can be used for further processing.")
    else:
        print("Failed to load basin data. Please check the input directory and try again.")


if __name__ == "__main__":
    main()
