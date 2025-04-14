#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# Fetch Basin Forecasts

This script extracts forecast data for basin centroids from the NOAA GEFS 35-day forecast
dataset and combines them into a single dataset with a unified structure.
"""

import os
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


def load_basin_centroids(file_path):
    """Load basin centroid coordinates from CSV.
    
    Args:
        file_path (str): Path to the basin centroids CSV file
        
    Returns:
        pandas.DataFrame: DataFrame containing basin names and coordinates
    """
    try:
        centroids = pd.read_csv(file_path)
        print(f"Loaded {len(centroids)} basin centroids from {file_path}")
        return centroids
    except Exception as e:
        print(f"Error loading basin centroids: {e}")
        return None


def extract_forecast_for_basin(ds, basin_name, latitude, longitude, init_time=None):
    """Extract forecast data for a specific basin point.
    
    Args:
        ds (xarray.Dataset): NOAA GEFS forecast dataset
        basin_name (str): Name of the basin
        latitude (float): Latitude of the basin centroid
        longitude (float): Longitude of the basin centroid
        init_time (str, optional): Initial forecast time. If None, use all available times.
        
    Returns:
        xarray.Dataset: Forecast data for the basin
    """
    # Select the init time if provided
    if init_time is not None:
        basin_ds = ds.sel(init_time=init_time)
    else:
        # Use all available initialization times
        basin_ds = ds
    
    # Extract forecast for the basin's location
    basin_ds = basin_ds.sel(latitude=latitude, longitude=longitude, method="nearest")
    
    # Add basin name as a coordinate
    basin_ds = basin_ds.assign_coords(basin=basin_name)
    
    return basin_ds


def fetch_forecasts_for_basins(ds, centroids, init_time=None):
    """Extract forecasts for all basin centroids while preserving original dimensions.
    
    Args:
        ds (xarray.Dataset): NOAA GEFS forecast dataset
        centroids (pandas.DataFrame): DataFrame with basin centroid coordinates
        init_time (str, optional): Initial forecast time. If None, use all available times.
        
    Returns:
        xarray.Dataset: Combined dataset with basin dimension replacing lat/lon
    """
    # Select the init time if provided, otherwise use all times
    if init_time is not None:
        ds = ds.sel(init_time=init_time)
    
    # Create a list to store data for each basin
    basin_data_list = []
    
    print(f"Extracting forecasts for {len(centroids)} basins...")
    for idx, row in centroids.iterrows():
        basin_name = row['basin_name']
        lat = row['latitude']
        lon = row['longitude']
        
        print(f"Processing basin: {basin_name} (lat: {lat:.4f}, lon: {lon:.4f})")
        
        # Extract forecast for this basin's location using nearest neighbor interpolation
        basin_ds = ds.sel(latitude=lat, longitude=lon, method="nearest")
        
        # Add basin name as a coordinate
        basin_ds = basin_ds.assign_coords(basin=basin_name)
        basin_data_list.append(basin_ds)
    
    # Combine all basin datasets along a new 'basin' dimension
    combined_ds = xr.concat(basin_data_list, dim="basin")
    
    print(f"Successfully extracted forecasts for {len(centroids)} basins")
    print(f"Dataset dimensions: {dict(combined_ds.dims)}")
    
    return combined_ds


def interpolate_to_hourly(ds, max_hours=240):
    """Interpolate forecast data to hourly resolution for the first 10 days.
    
    This function performs two main operations:
    1. Shortens the forecast to a specified maximum time horizon (default: 240 hours / 10 days)
    2. Interpolates the 3-hourly data to hourly resolution using linear interpolation
    3. Converts the lead_time dimension to integer hours (1-240)
    
    Args:
        ds (xarray.Dataset): The forecast dataset containing the lead_time dimension
        max_hours (int, optional): Maximum forecast horizon in hours. Default is 240 (10 days).
        
    Returns:
        xarray.Dataset: Dataset with hourly resolution and truncated forecast horizon,
                        with lead_time dimension as int64 ranging from 1 to max_hours
    """
    # Check if lead_time dimension exists
    if 'lead_time' not in ds.dims:
        raise ValueError("Dataset must contain a 'lead_time' dimension")
    
    # First, limit the forecast to the specified time horizon (default: 10 days)
    # Convert lead_time to hours if it's a timedelta
    if isinstance(ds.lead_time.values[0], np.timedelta64):
        lead_hours = ds.lead_time.dt.total_seconds().values / 3600
    else:
        # Assume lead_time is already in hours
        lead_hours = ds.lead_time.values
    
    # Select only lead times up to max_hours
    ds_shortened = ds.sel(lead_time=ds.lead_time[lead_hours <= max_hours])
    
    # Create a new array of hourly lead times as temporary values for interpolation
    if isinstance(ds.lead_time.values[0], np.timedelta64):
        # If lead_time is timedelta, create array of hourly timedeltas
        hourly_lead_times = np.array([np.timedelta64(int(hour), 'h') 
                                      for hour in range(int(max_hours) + 1)])
    else:
        # If lead_time is numeric, create array of hourly values
        hourly_lead_times = np.arange(0, max_hours + 1, 1)
    
    # Create a new dataset with the hourly lead times
    ds_hourly = ds_shortened.interp(lead_time=hourly_lead_times, method='linear')
    
    # Create a new lead_time coordinate with integer hours from 1 to max_hours
    # We exclude hour 0 as requested, starting at hour 1
    ds_hourly = ds_hourly.isel(lead_time=slice(1, None))
    
    # Create the new lead_time values (integers from 1 to max_hours)
    new_lead_time = np.arange(1, max_hours + 1)
    
    # Now assign the new coordinate
    ds_hourly = ds_hourly.assign_coords(lead_time=new_lead_time)
    
    # Update the lead_time attributes to indicate it's now in hours
    ds_hourly.lead_time.attrs['units'] = 'hours'
    ds_hourly.lead_time.attrs['long_name'] = 'Lead time in hours'
    
    # Add metadata to indicate that the dataset has been interpolated
    ds_hourly.attrs['interpolation'] = 'Linear interpolation to hourly resolution'
    ds_hourly.attrs['original_resolution'] = '3-hourly for first 10 days'
    ds_hourly.attrs['max_forecast_hours'] = max_hours
    ds_hourly.attrs['lead_time_format'] = 'Integer hours from 1 to 240'
    
    return ds_hourly




def main():
    """Main function to run the basin forecast extraction process."""
    # Define paths for input and output data
    basin_centroids_file = "../data/basin_centroids.csv"
    output_dir = "../data/basin_forecasts"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Basin centroids file: {basin_centroids_file}")
    
    # Load basin centroids
    centroids = load_basin_centroids(basin_centroids_file)
    
    if centroids is not None:
        # Open the NOAA GEFS dataset
        print("Loading NOAA GEFS dataset...")
        try:
            ds = xr.open_zarr(
                "https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr?email=optional@email.com", 
                decode_timedelta=True
            )
            
            # Get the latest init_time or use a specific one
            # init_time = "2025-01-01T00"  # Example specific time
            init_time = None  # Use all available init times
            
            # Extract forecasts for all basins while preserving the original dimensions
            combined_ds = fetch_forecasts_for_basins(ds, centroids, init_time)
            
            if combined_ds is not None:                
                print("\nConclusion:")
                print("This script has:")
                print("1. Loaded basin centroid coordinates from a CSV file")
                print("2. Extracted NOAA GEFS forecasts for each basin location")
                print("3. Created a combined dataset with dimensions: basin, init_time, lead_time, ensemble_member")
                
                # Return the combined dataset for inspection and further processing
                print("\nThe combined dataset is available for further processing.")
                print("Example code to access the dataset:")
                print("combined_ds.sel(basin='basin_name', init_time=combined_ds.init_time[-1])")
            else:
                print("Failed to extract basin forecasts.")
        except Exception as e:
            print(f"Error processing NOAA GEFS dataset: {e}")
    else:
        print("Failed to load basin centroids. Please check the input file and try again.")


if __name__ == "__main__":
    main()