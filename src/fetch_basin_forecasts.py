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
from typing import Optional, List, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


def load_basin_centroids(file_path: str) -> Optional[pd.DataFrame]:
    """Load basin centroid coordinates from CSV.
    
    Args:
        file_path (str): Path to the basin centroids CSV file
        
    Returns:
        Optional[pd.DataFrame]: DataFrame containing basin names and coordinates, or None on error.
    """
    try:
        centroids = pd.read_csv(file_path)
        print(f"Loaded {len(centroids)} basin centroids from {file_path}")
        return centroids
    except Exception as e:
        print(f"Error loading basin centroids: {e}")
        return None


def extract_forecast_for_basin(ds: xr.Dataset, basin_name: str, latitude: float, longitude: float, init_time: Optional[str] = None) -> xr.Dataset:
    """Extract forecast data for a specific basin point.
    
    Args:
        ds (xr.Dataset): NOAA GEFS forecast dataset
        basin_name (str): Name of the basin
        latitude (float): Latitude of the basin centroid
        longitude (float): Longitude of the basin centroid
        init_time (Optional[str]): Initial forecast time (e.g., 'YYYY-MM-DDTHH'). If None, use all available times.
        
    Returns:
        xr.Dataset: Forecast data for the basin
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


def fetch_forecasts_for_basins(ds: xr.Dataset, centroids: pd.DataFrame, init_time: Optional[str] = None) -> xr.Dataset:
    """Extract forecasts for all basin centroids while preserving original dimensions.
    
    Args:
        ds (xr.Dataset): NOAA GEFS forecast dataset
        centroids (pd.DataFrame): DataFrame with basin centroid coordinates
        init_time (Optional[str]): Initial forecast time (e.g., 'YYYY-MM-DDTHH'). If None, use all available times.
        
    Returns:
        xr.Dataset: Combined dataset with basin dimension replacing lat/lon
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


def interpolate_to_hourly(ds: xr.Dataset, max_hours: int = 240) -> xr.Dataset:
    """Interpolate forecast data to hourly resolution for the first 10 days.
    
    This function performs two main operations:
    1. Shortens the forecast to a specified maximum time horizon (default: 240 hours / 10 days)
    2. Interpolates the 3-hourly data to hourly resolution using linear interpolation
    3. Converts the lead_time dimension to integer hours (1-240)
    
    Args:
        ds (xr.Dataset): The forecast dataset containing the lead_time dimension
        max_hours (int, optional): Maximum forecast horizon in hours. Default is 240 (10 days).
        
    Returns:
        xr.Dataset: Dataset with hourly resolution and truncated forecast horizon,
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


def plot_basin_forecast(
    ds: xr.Dataset, 
    basin_name: str, 
    init_time: Union[str, np.datetime64, datetime], 
    variable: str, 
    uncertainty_quantiles: Optional[Union[List[float], Tuple[float, float]]] = None, 
    show_members: bool = True
) -> None:
    """Plots the forecast ensemble for a specific basin, init time, and variable.
       Optionally plots median and uncertainty bands instead of all members.

    Args:
        ds (xr.Dataset): The forecast dataset (e.g., basin_forecasts_hourly).
                             Must have dimensions 'basin', 'init_time', 'lead_time', 'ensemble_member'.
        basin_name (str): The name of the basin to plot.
        init_time (Union[str, np.datetime64, datetime]): The initialization time for the forecast.
        variable (str): The name of the variable to plot (e.g., 'temperature_2m').
        uncertainty_quantiles (Optional[Union[List[float], Tuple[float, float]]]): 
                                                        A list/tuple of two floats (0-1)
                                                        representing the lower and upper quantiles
                                                        for the uncertainty band (e.g., [0.05, 0.95]).
                                                        If provided, plots median and shaded band.
                                                        Defaults to None (plots all members).
        show_members (bool): If uncertainty_quantiles is provided, setting this to True
                             will also plot individual members lightly in the background.
                             Defaults to True.
    """
    # --- Input Validation ---
    if 'basin' not in ds.coords or basin_name not in ds['basin'].values:
        print(f"Error: Basin '{basin_name}' not found in dataset coordinates.")
        print(f"Available basins: {list(ds['basin'].values)}")
        return
        
    try:
        # Attempt to convert init_time to datetime64 for consistent comparison
        init_time_dt64 = np.datetime64(init_time)
        if init_time_dt64 not in ds['init_time'].values:
             available_times = pd.to_datetime(ds['init_time'].values)
             print(f"Error: Initialization time '{init_time}' not found in dataset.")
             print(f"Latest available init_time: {available_times.max()}")
             return
    except Exception as e:
        print(f"Error processing init_time '{init_time}': {e}")
        return

    if variable not in ds.data_vars:
        print(f"Error: Variable '{variable}' not found in dataset.")
        print(f"Available variables: {list(ds.data_vars)}")
        return
        
    plot_uncertainty = False
    if uncertainty_quantiles is not None:
        if not isinstance(uncertainty_quantiles, (list, tuple)) or len(uncertainty_quantiles) != 2:
            print("Error: uncertainty_quantiles must be a list or tuple of two floats (e.g., [0.05, 0.95]).")
            return
        q_low, q_high = uncertainty_quantiles
        if not (0 <= q_low < q_high <= 1):
            print("Error: uncertainty_quantiles must be between 0 and 1, with the first value smaller than the second.")
            return
        plot_uncertainty = True
        
    # --- Plotting Logic ---
    try:
        # Select the specific data slice using validated inputs
        forecast_slice = ds.sel(basin=basin_name, init_time=init_time_dt64)[variable]
        lead_time_values = forecast_slice['lead_time'].values

        # Create the plot
        plt.figure(figsize=(12, 6))
        ax = plt.gca() # Get current axes

        if plot_uncertainty:
            # Calculate quantiles
            q_lower = forecast_slice.quantile(uncertainty_quantiles[0], dim="ensemble_member")
            q_median = forecast_slice.quantile(0.5, dim="ensemble_member")
            q_upper = forecast_slice.quantile(uncertainty_quantiles[1], dim="ensemble_member")

            # Plot median
            q_median.plot(ax=ax, color='black', linewidth=2, label='Median')

            # Plot uncertainty band
            ax.fill_between(lead_time_values, q_lower.values, q_upper.values, color='skyblue', alpha=0.4, label=f'{uncertainty_quantiles[0]*100:.0f}-{uncertainty_quantiles[1]*100:.0f}% Quantile Range')

            # Optionally plot members lightly
            if show_members:
                 forecast_slice.plot.line(ax=ax, x='lead_time', hue='ensemble_member', add_legend=False, linewidth=0.5, alpha=0.3, color='grey')

            print(f"Plot generated with median and {uncertainty_quantiles} uncertainty band.")
            ax.legend() # Show legend for median and band

        elif show_members: # Only plot members if show_members is True and not plotting uncertainty band only
            # Default: plot all ensemble members
            forecast_slice.plot.line(ax=ax, x='lead_time', hue='ensemble_member', add_legend=False)
            print("Plot generated showing all ensemble members.")
        else:
             print("Plot generated with no elements (uncertainty band not requested, show_members=False).")


        # Customize the plot
        plt.title(f"Forecast for {variable} in {basin_name}\nInitialization Time: {pd.to_datetime(init_time_dt64).strftime('%Y-%m-%d %H:%M')}")
        plt.xlabel("Lead Time (hours)")
        y_label = f"{variable} ({ds[variable].attrs.get('units', 'N/A')})"
        plt.ylabel(y_label)
        plt.grid(True, linestyle='--', alpha=0.6)

        print(f"Plot generated for: Basin='{basin_name}', Init Time='{init_time}', Variable='{variable}'")


    except Exception as e:
        print(f"An unexpected error occurred during plotting: {e}")


def main() -> None:
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
        # Define Zarr URL (Consider moving to config or args)
        zarr_url = "https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr?email=optional@email.com"
        try:
            ds = xr.open_zarr(zarr_url, decode_timedelta=True)
            
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
            print(f"Error processing NOAA GEFS dataset from {zarr_url}: {e}")
    else:
        print("Failed to load basin centroids. Please check the input file and try again.")


if __name__ == "__main__":
    main()