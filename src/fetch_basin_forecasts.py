#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# Fetch NOAA GEFS Forecasts for Basin Points

This script loads basin centroid coordinates from a CSV file and fetches 
the corresponding NOAA GEFS forecast data for each point.
"""

# Import required libraries
import os
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import json

# Suppress warnings about chunking
warnings.filterwarnings('ignore', message='The specified chunks')


def connect_to_gefs_dataset(variables=None, email="optional@email.com"):
    """Connect to the NOAA GEFS dataset and return a dataset object.
    
    Args:
        variables (list, optional): List of variable names to load. If None, all variables are loaded.
        email (str, optional): Email address for tracking dataset usage.
        
    Returns:
        xarray.Dataset: The loaded dataset
    """
    zarr_url = f"https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr?email={email}"
    
    try:
        if variables:
            # Load only the specified variables to reduce memory usage
            ds = xr.open_zarr(zarr_url, decode_timedelta=True, consolidated=True)
            # Select just the requested variables and coordinates
            ds = ds[variables]
        else:
            # Load all variables
            ds = xr.open_zarr(zarr_url, decode_timedelta=True, consolidated=True)
            
        print(f"Connected to NOAA GEFS dataset with dimensions: {dict(ds.dims)}")
        return ds
    except Exception as e:
        print(f"Error connecting to NOAA GEFS dataset: {e}")
        return None


def extract_basin_forecasts(ds, basin_points, output_dir, latest_forecast_only=True):
    """
    Extract forecast data for each basin point and combine into a single dataset
    with basin as a new dimension.
    
    Args:
        ds (xarray.Dataset): The NOAA GEFS dataset
        basin_points (pandas.DataFrame): DataFrame containing basin point information
        output_dir (str): Directory to save the extracted forecasts
        latest_forecast_only (bool): If True, extract only the latest forecast initialization
        
    Returns:
        xarray.Dataset: Combined dataset with basin as an additional coordinate
    """
    # Use only the latest forecast if specified
    if latest_forecast_only:
        init_times = pd.to_datetime(ds.init_time.values)
        latest_init = init_times.max()
        ds = ds.sel(init_time=latest_init)
        print(f"Using only the latest forecast from {latest_init}")
    
    # List to store individual basin datasets
    basin_ds_list = []
    basin_names = []
    
    # Process each basin point
    for idx, basin in basin_points.iterrows():
        basin_name = basin['basin_name']
        print(f"\nExtracting forecast for {basin_name}...")
        
        # Select the nearest grid point to the basin centroid
        nearest_lat = basin['nearest_lat']
        nearest_lon = basin['nearest_lon']
        
        try:
            # Extract data for the specific location
            basin_ds = ds.sel(latitude=nearest_lat, longitude=nearest_lon, method='nearest')
            
            # Add basin metadata
            basin_ds = basin_ds.assign_coords(basin=basin_name)
            basin_ds_list.append(basin_ds)
            basin_names.append(basin_name)
            
        except Exception as e:
            print(f"Error extracting forecast for {basin_name}: {e}")
    
    # Combine all basin datasets along a new 'basin' dimension
    if basin_ds_list:
        combined_ds = xr.concat(basin_ds_list, dim='basin')
        combined_ds = combined_ds.assign_coords(basin=basin_names)
        
        # Add basin metadata to the dataset
        basin_info = basin_points.set_index('basin_name')
        for col in ['centroid_lat', 'centroid_lon', 'grid_distance_km']:
            combined_ds[f'basin_{col}'] = ('basin', basin_info[col].loc[basin_names].values)
        
        print(f"\nCombined forecasts for {len(basin_names)} basins into a single dataset")
        return combined_ds
    else:
        print("No basin forecasts were successfully extracted")
        return None


def save_forecasts(combined_ds, output_dir):
    """Save the combined forecast dataset.
    
    Args:
        combined_ds (xarray.Dataset): Combined forecast dataset with basin as a dimension
        output_dir (str): Directory to save the extracted forecasts
    """
    if combined_ds is None:
        print("No forecast data to save.")
        return
    
    try:
        # Get the latest forecast initialization time
        init_time = pd.to_datetime(combined_ds.init_time.values[0])
        init_time_str = init_time.strftime("%Y%m%d_%H")
        
        # Create output filename
        output_file = os.path.join(output_dir, f"basin_forecasts_{init_time_str}.nc")
        
        # Save to NetCDF
        combined_ds.to_netcdf(output_file)
        print(f"Saved combined basin forecasts to {output_file}")
        
        # Also save a metadata file with basin information
        basin_info = {
            'init_time': init_time.strftime("%Y-%m-%d %H:%M:%S"),
            'basins': []
        }
        
        for basin in combined_ds.basin.values:
            basin_info['basins'].append({
                'name': basin,
                'centroid_lat': float(combined_ds.basin_centroid_lat.sel(basin=basin).values),
                'centroid_lon': float(combined_ds.basin_centroid_lon.sel(basin=basin).values),
                'grid_distance_km': float(combined_ds.basin_grid_distance_km.sel(basin=basin).values)
            })
        
        metadata_file = os.path.join(output_dir, f"basin_forecasts_{init_time_str}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(basin_info, f, indent=2)
        
        print(f"Saved metadata to {metadata_file}")
        
    except Exception as e:
        print(f"Error saving forecast data: {e}")


def plot_forecast_sample(combined_ds, variable='temperature_2m'):
    """Plot a sample forecast for visualization.
    
    Args:
        combined_ds (xarray.Dataset): Combined forecast dataset
        variable (str): Variable to plot
    """
    if combined_ds is None:
        return
    
    try:
        # Select a sample of basins (up to 5) to avoid cluttered plot
        if len(combined_ds.basin) > 5:
            sample_basins = np.random.choice(combined_ds.basin.values, 5, replace=False)
            plot_ds = combined_ds.sel(basin=sample_basins)
        else:
            plot_ds = combined_ds
        
        # Convert to pandas for easier plotting
        print(f"Plotting {variable} forecasts for sample basins...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get the lead time in a more readable format
        lead_times = pd.to_datetime(plot_ds.lead_time.values)
        
        # Plot forecast data for each basin
        for basin in plot_ds.basin.values:
            basin_data = plot_ds[variable].sel(basin=basin)
            
            # If there are multiple ensemble members, plot the mean and range
            if 'ens' in basin_data.dims:
                mean_data = basin_data.mean(dim='ens')
                min_data = basin_data.min(dim='ens')
                max_data = basin_data.max(dim='ens')
                
                ax.plot(lead_times, mean_data, label=basin)
                ax.fill_between(lead_times, min_data, max_data, alpha=0.2)
            else:
                ax.plot(lead_times, basin_data, label=basin)
        
        # Format the plot
        ax.set_title(f"{variable} Forecast ({plot_ds.attrs.get('units', 'unknown units')})")
        ax.set_xlabel("Lead Time")
        ax.set_ylabel(f"{variable} ({plot_ds[variable].attrs.get('units', 'unknown units')})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error plotting forecast sample: {e}")


def main():
    """Main function to run the basin forecast extraction process."""
    # Define paths for input and output data
    basin_centroids_path = "../data/basin_centroids_with_grid_points.csv"
    output_dir = "../data/basin_forecasts"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load basin centroids with grid points
    try:
        basin_points = pd.read_csv(basin_centroids_path)
        print(f"Loaded {len(basin_points)} basin points from {basin_centroids_path}")
        print(basin_points.head())  # Display first few rows
    except FileNotFoundError:
        print(f"Error: File {basin_centroids_path} not found.")
        print("Run the extract_basin_centroids.py script first to generate this file.")
        return
    
    # Connect to the dataset with just the variables we need
    # For hydrological modeling, we typically need temperature, precipitation, and sometimes wind/humidity
    variables = [
        'temperature_2m',                        # 2m air temperature
        'precipitation_surface',                 # Total precipitation
        'relative_humidity_2m',                  # Relative humidity
        'downward_long_wave_radiation_flux_surface',  # Surface downward long-wave radiation
        'downward_short_wave_radiation_flux_surface', # Surface downward short-wave radiation
        'precipitable_water_atmosphere',         # Precipitable water
        'pressure_surface',                      # Surface pressure
        'maximum_temperature_2m',                # Maximum temperature
        'minimum_temperature_2m',                # Minimum temperature
        'total_cloud_cover_atmosphere',          # Total cloud cover
        'wind_u_10m',                            # 10m U wind component
        'wind_v_10m'                             # 10m V wind component
    ]
    
    # Get the latest GEFS forecast dataset
    email = "your.email@example.com"  # Replace with your email
    ds = connect_to_gefs_dataset(variables, email)
    
    # Display basic information about the dataset
    if ds is not None:
        print("\nDataset Variables:")
        for var_name in ds.data_vars:
            var = ds[var_name]
            print(f"  - {var_name}: {var.dims} ({var.attrs.get('units', 'no units')})")
            
        # Get information about forecast dates
        init_times = pd.to_datetime(ds.init_time.values)
        latest_init = init_times.max()
        print(f"\nLatest forecast initialization time: {latest_init}")
        
        # Extract forecasts for all basin points
        combined_ds = extract_basin_forecasts(ds, basin_points, output_dir)
        
        # Plot a sample forecast for visualization
        if combined_ds is not None:
            # Plot temperature forecast
            plot_forecast_sample(combined_ds, 'temperature_2m')
            
            # Plot precipitation forecast
            plot_forecast_sample(combined_ds, 'precipitation_surface')
            
            # Save the forecasts
            save_forecasts(combined_ds, output_dir)
    else:
        print("Failed to connect to NOAA GEFS dataset. Please check your internet connection and try again.")


if __name__ == "__main__":
    main()
