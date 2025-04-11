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
    new_lead_time = np.arange(1, max_hours + 1, dtype=np.int64)
    
    # Convert the dataset to use the new integer lead_time dimension
    # First, we need to select all but the first time step (to skip hour 0)
    ds_hourly = ds_hourly.isel(lead_time=slice(1, None))
    
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


def plot_ensemble_members(ds, basin=None, init_time=None, variable='temperature_2m', 
                          output_dir=None, show_plot=True, figsize=(16, 8), dpi=150):
    """Plot all ensemble members for a specific variable.
    
    Args:
        ds (xarray.Dataset): The forecast dataset
        basin (str, optional): Basin name to plot. If None, uses the first basin in the dataset.
        init_time (datetime or str, optional): Initialization time to plot. If None, uses the latest.
        variable (str, optional): Variable to plot. Default is 'temperature_2m'.
        output_dir (str, optional): Directory to save plot. If None, plot is not saved.
        show_plot (bool, optional): Whether to display the plot. Default is True.
        figsize (tuple, optional): Figure size as (width, height). Default is (16, 8).
        dpi (int, optional): Resolution for saved figure. Default is 150.
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Select specific basin if provided
    if basin is not None and 'basin' in ds.dims:
        ds = ds.sel(basin=basin)
    elif 'basin' in ds.dims:
        basin = ds.basin.values[0]  # Use first basin if not specified
        ds = ds.sel(basin=basin)
    else:
        basin = 'Basin'  # Generic name if basin dimension doesn't exist
    
    # Check if the variable exists
    if variable not in ds.data_vars:
        print(f"Variable '{variable}' not found in dataset. Available variables: {list(ds.data_vars)}")
        return None
    
    # Select specific init time if provided, otherwise use the latest
    if init_time is not None:
        if 'init_time' in ds.dims:
            ds = ds.sel(init_time=init_time, method='nearest')
    elif 'init_time' in ds.dims:
        latest_init_time = ds.init_time.values[-1]
        ds = ds.sel(init_time=latest_init_time)
        init_time = latest_init_time
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each ensemble member
    data_var = ds[variable]
    
    # Determine the time dimension to use
    time_dim = 'time' if 'time' in data_var.dims else 'valid_time'
    if time_dim not in data_var.dims and 'lead_time' in data_var.dims:
        # Use lead_time if time dimension is not available
        time_dim = 'lead_time'
    
    # Plot the data
    data_var.plot.line(x=time_dim, hue='ensemble_member', alpha=0.3, add_legend=False, ax=ax)
    
    # Get units from attrs or use default
    units = data_var.attrs.get('units', '')
    
    # Add title and labels
    title = f"{variable.replace('_', ' ').title()} Forecast for {basin}"
    if init_time is not None:
        init_time_str = pd.to_datetime(init_time).strftime('%Y-%m-%d %H:%M')
        title += f" (Init: {init_time_str})"
    
    ax.set_title(title)
    ax.set_ylabel(f"{variable.replace('_', ' ').title()} ({units})")
    ax.set_xlabel("Forecast Time")
    ax.grid(True, alpha=0.3)
    
    # Add a horizontal line at freezing point if temperature
    if 'temperature' in variable.lower() and units:
        if 'c' in units.lower():
            ax.axhline(y=0, color='blue', linestyle='--', alpha=0.7, label='Freezing Point (0°C)')
        elif 'k' in units.lower():
            ax.axhline(y=273.15, color='blue', linestyle='--', alpha=0.7, label='Freezing Point (273.15K)')
    
    # Customize legend
    if len(ds.ensemble_member) > 10:
        # Just add a note about ensemble members
        plt.text(0.98, 0.02, f"Showing all {len(ds.ensemble_member)} ensemble members", 
                 ha='right', va='bottom', transform=ax.transAxes,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        # Create a custom legend with a reasonable number of entries
        legend_elements = [
            Line2D([0], [0], color='C0', lw=1, alpha=0.7, 
                   label=f'Ensemble {i+1}') for i in range(len(ds.ensemble_member))
        ]
        ax.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    
    # Save the plot if output directory is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a filename with basin and init_time if available
        if basin != 'Basin':
            basin_str = str(basin).replace("/", "_").replace(" ", "_")
            filename = f"{basin_str}_{variable}"
        else:
            filename = f"{variable}"
            
        if init_time is not None:
            init_time_str = pd.to_datetime(init_time).strftime('%Y%m%d%H')
            filename += f"_init_{init_time_str}"
            
        filepath = os.path.join(output_dir, f"{filename}_ensemble_members.png")
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Saved ensemble members plot to {filepath}")
    
    # Show or close the plot
    if show_plot:
        plt.show()
    else:
        plt.close()
        
    return fig


def plot_ensemble_statistics(ds, basin=None, init_time=None, variable='temperature_2m',
                            output_dir=None, show_plot=True, figsize=(16, 8), dpi=150):
    """Plot ensemble statistics (mean, median, min/max range, percentiles).
    
    Args:
        ds (xarray.Dataset): The forecast dataset
        basin (str, optional): Basin name to plot. If None, uses the first basin in the dataset.
        init_time (datetime or str, optional): Initialization time to plot. If None, uses the latest.
        variable (str, optional): Variable to plot. Default is 'temperature_2m'.
        output_dir (str, optional): Directory to save plot. If None, plot is not saved.
        show_plot (bool, optional): Whether to display the plot. Default is True.
        figsize (tuple, optional): Figure size as (width, height). Default is (16, 8).
        dpi (int, optional): Resolution for saved figure. Default is 150.
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Check if the variable exists early to avoid unnecessary processing
    if variable not in ds.data_vars:
        print(f"Variable '{variable}' not found in dataset. Available variables: {list(ds.data_vars)}")
        return None
    
    # Use .load() to ensure data is in memory for faster processing
    # Only load the necessary variable and dimensions
    data_subset = ds[variable].load()
    
    # Select specific basin - only select if needed
    if basin is not None and 'basin' in ds.dims:
        data_subset = data_subset.sel(basin=basin)
    elif 'basin' in ds.dims:
        basin = ds.basin.values[0]  # Use first basin if not specified
        data_subset = data_subset.sel(basin=basin)
    else:
        basin = 'Basin'  # Generic name if basin dimension doesn't exist
    
    # Select specific init time - only select if needed
    if 'init_time' in data_subset.dims:
        if init_time is not None:
            data_subset = data_subset.sel(init_time=init_time, method='nearest')
        else:
            latest_init_time = data_subset.init_time.values[-1]
            data_subset = data_subset.sel(init_time=latest_init_time)
            init_time = latest_init_time
    
    # Determine the time dimension to use - do this only once
    time_dim = next((dim for dim in ['time', 'valid_time', 'lead_time'] 
                     if dim in data_subset.dims), None)
    
    if time_dim is None:
        print(f"No time dimension found in the dataset. Available dimensions: {list(data_subset.dims)}")
        return None
    
    # Calculate statistics all at once to minimize loop iterations
    # Use dask optimized functions where possible
    stats = {
        'mean': data_subset.mean(dim='ensemble_member'),
        'median': data_subset.median(dim='ensemble_member'),
        'min': data_subset.min(dim='ensemble_member'),
        'max': data_subset.max(dim='ensemble_member')
    }
    
    # Calculate quantiles in a single operation
    quantiles = data_subset.quantile([0.25, 0.75], dim='ensemble_member')
    stats['q25'] = quantiles.sel(quantile=0.25)
    stats['q75'] = quantiles.sel(quantile=0.75)
    
    # Compute all statistics at once to optimize dask operations
    stats = {k: v.compute() for k, v in stats.items()}
    
    # Get time values once
    time_coord = stats['mean'][time_dim]
    
    if np.issubdtype(time_coord.dtype, np.timedelta64):
        # Optimize timedelta handling
        if init_time is not None:
            base_time = pd.to_datetime(init_time)
            # Vectorized operation to convert timedeltas to timestamps
            time_values = base_time + pd.to_timedelta(time_coord.values)
        else:
            # Fast conversion to days for display
            hours = time_coord.values.astype('timedelta64[h]').astype(float)
            days = hours / 24.0
            time_values = np.arange(len(days))
            # Pre-calculate tick labels
            tick_labels = [f"Day {d:.1f}" for d in days]
    elif np.issubdtype(time_coord.dtype, np.datetime64):
        time_values = time_coord.values  # Keep as numpy array for faster plotting
    else:
        time_values = time_coord.values
    
    # Create figure only after we know we have data
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot with optimized calls
    ax.plot(time_values, stats['median'], 'b-', linewidth=2.5, label='Median')
    ax.plot(time_values, stats['mean'], 'r-', linewidth=2.5, label='Mean')
    ax.fill_between(time_values, stats['q25'], stats['q75'], color='b', alpha=0.2, label='25-75th Percentile')
    ax.fill_between(time_values, stats['min'], stats['max'], color='b', alpha=0.1, label='Min-Max Range')
    
    # Get units from attrs or use default
    units = data_subset.attrs.get('units', '')
    
    # Set title and labels
    title = f"{variable.replace('_', ' ').title()} Forecast Distribution for {basin}"
    if init_time is not None:
        # Format date string once
        init_time_str = pd.to_datetime(init_time).strftime('%Y-%m-%d %H:%M')
        title += f" (Init: {init_time_str})"
    
    ax.set_title(title)
    ax.set_ylabel(f"{variable.replace('_', ' ').title()} ({units})")
    
    # Set appropriate x-axis label
    ax.set_xlabel("Time from Forecast Initialization" if time_dim == 'lead_time' else "Forecast Time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Only format dates if needed
    if np.issubdtype(time_coord.dtype, np.timedelta64) and init_time is None:
        # Set pre-calculated tick labels
        ax.set_xticks(time_values)
        ax.set_xticklabels(tick_labels)
    elif np.issubdtype(time_coord.dtype, np.datetime64):
        fig.autofmt_xdate()
    
    plt.tight_layout()
    
    # Save the plot if output directory is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Build filename components
        if basin != 'Basin':
            basin_str = str(basin).replace("/", "_").replace(" ", "_")
            filename_parts = [basin_str, variable]
        else:
            filename_parts = [variable]
            
        if init_time is not None:
            # Format date string once
            init_time_str = pd.to_datetime(init_time).strftime('%Y%m%d%H')
            filename_parts.append(f"init_{init_time_str}")
            
        filepath = os.path.join(output_dir, f"{'_'.join(filename_parts)}_statistics.png")
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Saved ensemble statistics plot to {filepath}")
    
    # Show or close the plot
    if not show_plot:
        plt.close(fig)
        
    return fig


def plot_quantile_distribution(ds, basin=None, init_time=None, variable='temperature_2m',
                              quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
                              output_dir=None, show_plot=True, figsize=(16, 8), dpi=150):
    """Plot quantile distribution across ensemble members.
    
    Args:
        ds (xarray.Dataset): The forecast dataset
        basin (str, optional): Basin name to plot. If None, uses the first basin in the dataset.
        init_time (datetime or str, optional): Initialization time to plot. If None, uses the latest.
        variable (str, optional): Variable to plot. Default is 'temperature_2m'.
        quantiles (list, optional): List of quantiles to plot. Default is [0.05, 0.25, 0.5, 0.75, 0.95].
        output_dir (str, optional): Directory to save plot. If None, plot is not saved.
        show_plot (bool, optional): Whether to display the plot. Default is True.
        figsize (tuple, optional): Figure size as (width, height). Default is (16, 8).
        dpi (int, optional): Resolution for saved figure. Default is 150.
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Select specific basin if provided
    if basin is not None and 'basin' in ds.dims:
        ds = ds.sel(basin=basin)
    elif 'basin' in ds.dims:
        basin = ds.basin.values[0]  # Use first basin if not specified
        ds = ds.sel(basin=basin)
    else:
        basin = 'Basin'  # Generic name if basin dimension doesn't exist
    
    # Check if the variable exists
    if variable not in ds.data_vars:
        print(f"Variable '{variable}' not found in dataset. Available variables: {list(ds.data_vars)}")
        return None
    
    # Select specific init time if provided, otherwise use the latest
    if init_time is not None:
        if 'init_time' in ds.dims:
            ds = ds.sel(init_time=init_time, method='nearest')
    elif 'init_time' in ds.dims:
        latest_init_time = ds.init_time.values[-1]
        ds = ds.sel(init_time=latest_init_time)
        init_time = latest_init_time
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate quantiles across ensemble members
    quantile_data = ds[variable].quantile(quantiles, dim='ensemble_member')
    
    # Determine the time dimension to use
    time_dim = 'time' if 'time' in quantile_data.dims else 'valid_time'
    if time_dim not in quantile_data.dims and 'lead_time' in quantile_data.dims:
        # Use lead_time if time dimension is not available
        time_dim = 'lead_time'
    
    # Plot quantiles
    quantile_data.plot(x=time_dim, hue='quantile', ax=ax)
    
    # Get units from attrs or use default
    units = ds[variable].attrs.get('units', '')
    
    # Add title and labels
    title = f"{variable.replace('_', ' ').title()} Forecast Quantiles for {basin}"
    if init_time is not None:
        init_time_str = pd.to_datetime(init_time).strftime('%Y-%m-%d %H:%M')
        title += f" (Init: {init_time_str})"
    
    ax.set_title(title)
    ax.set_ylabel(f"{variable.replace('_', ' ').title()} ({units})")
    ax.set_xlabel("Forecast Time")
    ax.grid(True, alpha=0.3)
    
    # Format legend labels to show percentiles
    handles, labels = ax.get_legend_handles_labels()
    percentile_labels = [f"{float(q)*100:.0f}th Percentile" for q in quantiles]
    ax.legend(handles, percentile_labels, title='Forecast Uncertainty', loc='best')
    
    plt.tight_layout()
    
    # Save the plot if output directory is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a filename with basin and init_time if available
        if basin != 'Basin':
            basin_str = str(basin).replace("/", "_").replace(" ", "_")
            filename = f"{basin_str}_{variable}"
        else:
            filename = f"{variable}"
            
        if init_time is not None:
            init_time_str = pd.to_datetime(init_time).strftime('%Y%m%d%H')
            filename += f"_init_{init_time_str}"
            
        filepath = os.path.join(output_dir, f"{filename}_quantiles.png")
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Saved quantile plot to {filepath}")
    
    # Show or close the plot
    if show_plot:
        plt.show()
    else:
        plt.close()
        
    return fig


def plot_basin_temperature_forecasts(combined_ds, output_dir=None, show_plot=True):
    """Create plots for temperature forecasts for each basin.
    
    This function is maintained for backward compatibility.
    It now calls the more comprehensive plot_basin_forecasts function.
    
    Args:
        combined_ds (xarray.Dataset): Combined basin forecast dataset
        output_dir (str, optional): Directory to save plots. If None, plots are not saved.
        show_plot (bool, optional): Whether to display the plots. Default is True.
    """
    # Create a plots directory within the output directory
    if output_dir is not None:
        plots_dir = os.path.join(output_dir, "plots")
    else:
        plots_dir = None
    
    # Call the comprehensive plotting function with default parameters
    return plot_basin_forecasts(
        combined_ds, 
        variables=['temperature_2m', 'precipitation'] if 'precipitation' in combined_ds.data_vars else ['temperature_2m'],
        output_dir=plots_dir,
        show_plot=show_plot
    )


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
                # Create plots for each basin
                plot_dir = os.path.join(output_dir, "plots")
                plot_basin_temperature_forecasts(combined_ds, plot_dir)
                
                print("\nConclusion:")
                print("This script has:")
                print("1. Loaded basin centroid coordinates from a CSV file")
                print("2. Extracted NOAA GEFS forecasts for each basin location")
                print("3. Created a combined dataset with dimensions: basin, init_time, lead_time, ensemble_member")
                print("4. Generated forecast plots")
                print(f"\nThe plots are available in {os.path.join(output_dir, 'plots')}")
                
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