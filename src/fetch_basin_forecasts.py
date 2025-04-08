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
    """Extract forecasts for all basin centroids.
    
    Args:
        ds (xarray.Dataset): NOAA GEFS forecast dataset
        centroids (pandas.DataFrame): DataFrame with basin centroid coordinates
        init_time (str, optional): Initial forecast time. If None, use the latest.
        
    Returns:
        list: List of forecast datasets for each basin
    """
    basin_forecasts = []
    
    print(f"Extracting forecasts for {len(centroids)} basins...")
    for idx, row in centroids.iterrows():
        basin_name = row['basin_name']
        lat = row['latitude']
        lon = row['longitude']
        
        print(f"Processing basin: {basin_name} (lat: {lat:.4f}, lon: {lon:.4f})")
        
        # Extract forecast for this basin
        basin_ds = extract_forecast_for_basin(ds, basin_name, lat, lon, init_time)
        basin_forecasts.append(basin_ds)
    
    return basin_forecasts


def merge_basin_forecasts(basin_forecasts):
    """Merge individual basin forecasts into a combined dataset.
    
    Args:
        basin_forecasts (list): List of xarray.Dataset objects, one per basin
        
    Returns:
        xarray.Dataset: Combined dataset with all basin forecasts
    """
    if not basin_forecasts:
        print("No basin forecasts to merge.")
        return None
    
    # Create a basin dimension from the list of datasets
    combined_ds = xr.concat(basin_forecasts, dim="basin")
    
    # Initialize a new dataset with the required dimensions
    new_ds = xr.Dataset()
    
    # Add basin dimension
    new_ds.coords['basin'] = combined_ds.basin.values
    
    # Create an array of all unique valid times across all forecasts and init times
    unique_times = sorted(np.unique(combined_ds.valid_time.values))
    new_ds.coords['time'] = pd.DatetimeIndex(unique_times)
    
    # Create lead_time dimension in hours (0 to 840 hours = 35 days)
    # Each lead time represents hours from the forecast initialization
    max_lead_hours = 840  # 35 days in hours
    lead_times = np.arange(0, max_lead_hours + 1, 3)  # 3-hourly steps
    new_ds.coords['lead_time'] = lead_times
    
    # Add ensemble member dimension
    new_ds.coords['ensemble_member'] = combined_ds.ensemble_member.values
    
    print(f"Setting up dataset with dimensions: basin ({len(new_ds.basin)}), time ({len(new_ds.time)}), "
          f"lead_time ({len(new_ds.lead_time)}), ensemble_member ({len(new_ds.ensemble_member)})")
    
    # Get list of variables to process
    var_names = list(combined_ds.data_vars)
    
    # Process each variable
    for var_name in var_names:
        print(f"Processing variable: {var_name}")
        var = combined_ds[var_name]
        
        # Check if the variable has an ensemble_member dimension
        if 'ensemble_member' in var.dims:
            # Create a new array with dimensions: basin, time, lead_time, ensemble_member
            # Initially filled with NaN values
            shape = (len(new_ds.basin), len(new_ds.time), len(new_ds.lead_time), len(new_ds.ensemble_member))
            new_data = np.full(shape, np.nan)
            
            # Map the data into the new array structure
            for b_idx, basin in enumerate(new_ds.basin.values):
                basin_data = var.sel(basin=basin)
                
                # For each initialization time in the dataset
                for init_time in combined_ds.init_time.values:
                    # Convert to datetime for calculations
                    init_datetime = pd.to_datetime(init_time)
                    
                    # Process each lead time for this initialization
                    for lead_time in combined_ds.lead_time.values:
                        # Calculate the valid time for this forecast point
                        lead_td = pd.to_timedelta(lead_time)
                        valid_time = init_datetime + lead_td
                        lead_hours = int(lead_td.total_seconds() / 3600)
                        
                        # Check if this point is within our grid
                        if valid_time in new_ds.time.values and lead_hours in new_ds.lead_time.values:
                            # Find indices in our target array
                            t_idx = np.where(new_ds.time.values == valid_time)[0]
                            lt_idx = np.where(new_ds.lead_time.values == lead_hours)[0]
                            
                            if len(t_idx) > 0 and len(lt_idx) > 0:
                                # Get the forecast values for all ensemble members
                                try:
                                    forecast_values = basin_data.sel(init_time=init_time, lead_time=lead_time).values
                                    # Store each ensemble member in the correct position
                                    for e_idx, _ in enumerate(new_ds.ensemble_member.values):
                                        new_data[b_idx, t_idx[0], lt_idx[0], e_idx] = forecast_values[e_idx]
                                except (KeyError, ValueError, IndexError) as e:
                                    # Skip if the specific combination doesn't exist
                                    continue
            
            # Create the new variable in the dataset
            new_ds[var_name] = xr.DataArray(
                data=new_data,
                dims=['basin', 'time', 'lead_time', 'ensemble_member'],
                coords={
                    'basin': new_ds.basin,
                    'time': new_ds.time,
                    'lead_time': new_ds.lead_time,
                    'ensemble_member': new_ds.ensemble_member
                },
                attrs=var.attrs
            )
        else:
            # For variables without ensemble members, just copy them
            # This handles coordinates and other metadata
            if not any(dim in var.dims for dim in ['init_time', 'ensemble_member']):
                new_ds[var_name] = var
    
    # Add attributes from the original dataset
    for attr_name, attr_value in combined_ds.attrs.items():
        new_ds.attrs[attr_name] = attr_value
    
    print("Successfully merged forecasts and restructured dimensions.")
    return new_ds


def plot_basin_temperature_forecasts(combined_ds, output_dir=None):
    """Create plots for temperature forecasts for each basin.
    
    Args:
        combined_ds (xarray.Dataset): Combined basin forecast dataset
        output_dir (str, optional): Directory to save plots. If None, just display.
    """
    if combined_ds is None:
        print("No dataset to plot.")
        return
    
    # Create output directory if specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get unique basins
    basins = combined_ds.basin.values
    
    # Check if temperature variable exists and has ensemble_member dimension
    if 'temperature_2m' not in combined_ds.data_vars or 'ensemble_member' not in combined_ds['temperature_2m'].dims:
        print("Temperature variable with ensemble_member dimension not found in the dataset.")
        return
    
    print(f"Creating temperature forecast plots for {len(basins)} basins with "
          f"{len(combined_ds.ensemble_member)} ensemble members")
    
    for basin in basins:
        # Extract data for this basin
        basin_ds = combined_ds.sel(basin=basin)
        
        # Plot temperature forecast with ensemble members
        plt.figure(figsize=(14, 8))
        
        # Plot each ensemble member as a separate line
        basin_ds.temperature_2m.plot.line(x='time', hue='ensemble_member', add_legend=False, alpha=0.3)
        
        # Add title and labels
        plt.title(f"Temperature Forecast for {basin}")
        plt.ylabel(f"Temperature ({basin_ds.temperature_2m.attrs.get('units', 'K')})")
        plt.xlabel("Time")
        plt.grid(True, alpha=0.3)
        
        # Customize legend (limit to max 10 entries if there are many)
        if len(combined_ds.ensemble_member) > 10:
            # Just add a note about the ensemble members without showing individual legends
            plt.text(0.98, 0.02, f"Showing all {len(combined_ds.ensemble_member)} ensemble members", 
                     ha='right', va='bottom', transform=plt.gca().transAxes,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            # Create a custom legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='C0', lw=1, alpha=0.7, 
                       label=f'Ensemble {i+1}') for i in range(len(combined_ds.ensemble_member))
            ]
            plt.legend(handles=legend_elements, loc='upper right')
        
        # Save or display the plot
        if output_dir is not None:
            # Create a filename-safe version of the basin name
            basin_filename = str(basin).replace("/", "_").replace(" ", "_")
            filename = os.path.join(output_dir, f"{basin_filename}_temperature_forecast.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved plot to {filename}")
        else:
            plt.tight_layout()
            plt.show()
    
    # Create a visualization showing the ensemble spread
    for basin in basins:
        plt.figure(figsize=(14, 8))
        basin_ds = combined_ds.sel(basin=basin)
        
        # Calculate statistics across ensemble members directly from the DataArray
        ens_mean = basin_ds.temperature_2m.mean(dim='ensemble_member')
        ens_median = basin_ds.temperature_2m.median(dim='ensemble_member')
        ens_min = basin_ds.temperature_2m.min(dim='ensemble_member')
        ens_max = basin_ds.temperature_2m.max(dim='ensemble_member')
        ens_q25 = basin_ds.temperature_2m.quantile(0.25, dim='ensemble_member')
        ens_q75 = basin_ds.temperature_2m.quantile(0.75, dim='ensemble_member')
        
        # Plot median and min/max range
        plt.plot(basin_ds.time, ens_median, 'b-', linewidth=2, label='Median')
        plt.plot(basin_ds.time, ens_mean, 'r-', linewidth=2, label='Mean')
        plt.fill_between(basin_ds.time, ens_q25, ens_q75, color='b', alpha=0.2, label='25-75th Percentile')
        plt.fill_between(basin_ds.time, ens_min, ens_max, color='b', alpha=0.1, label='Min-Max Range')
        
        plt.title(f"Temperature Forecast Distribution for {basin}")
        plt.ylabel(f"Temperature ({basin_ds.temperature_2m.attrs.get('units', 'K')})")
        plt.xlabel("Time")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save or display the plot
        if output_dir is not None:
            basin_filename = str(basin).replace("/", "_").replace(" ", "_")
            filename = os.path.join(output_dir, f"{basin_filename}_temperature_distribution.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved ensemble distribution plot to {filename}")
        else:
            plt.tight_layout()
            plt.show()
    
    # Create a comparative plot of temperature forecasts across all basins
    if len(basins) > 1:
        plt.figure(figsize=(14, 10))
        
        # Use the ensemble mean for comparison across basins
        for basin in basins:
            basin_data = combined_ds.sel(basin=basin).temperature_2m
            
            # Calculate mean across ensemble members and lead times
            basin_mean = basin_data.mean(dim=['ensemble_member', 'lead_time'])
            plt.plot(basin_mean.time, basin_mean, '-', label=str(basin))
        
        plt.title(f"Temperature Forecast Comparison Across Basins (Ensemble Mean)")
        plt.ylabel(f"Temperature ({combined_ds.temperature_2m.attrs.get('units', 'K')})")
        plt.xlabel("Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the comparative plot
        if output_dir is not None:
            filename = os.path.join(output_dir, "basin_temperature_comparison.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved comparative plot to {filename}")
        else:
            plt.show()


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
            init_time = None  # Use the latest
            
            # Extract forecasts for each basin
            basin_forecasts = fetch_forecasts_for_basins(ds, centroids, init_time)
            
            # Merge the basin forecasts into a single dataset
            combined_ds = merge_basin_forecasts(basin_forecasts)
            
            if combined_ds is not None:
                # Create plots for each basin
                plot_dir = os.path.join(output_dir, "plots")
                plot_basin_temperature_forecasts(combined_ds, plot_dir)
                
                print("\nConclusion:")
                print("This script has:")
                print("1. Loaded basin centroid coordinates from a CSV file")
                print("2. Extracted NOAA GEFS forecasts for each basin location")
                print("3. Combined the individual basin forecasts into a single dataset")
                print("4. Generated forecast plots")
                print(f"\nThe plots are available in {os.path.join(output_dir, 'plots')}")
                
                # Return the combined dataset for inspection and further processing
                print("\nThe combined dataset is available for further processing.")
                print("Example code to access the dataset:")
                print("combined_ds.sel(basin='basin_name')")
            else:
                print("Failed to merge basin forecasts.")
        except Exception as e:
            print(f"Error processing NOAA GEFS dataset: {e}")
    else:
        print("Failed to load basin centroids. Please check the input file and try again.")


if __name__ == "__main__":
    main()