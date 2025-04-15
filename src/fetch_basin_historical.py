#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# Fetch Basin Historical Weather Data

This script fetches historical weather data for given basin centroids using the Open-Meteo API.
"""

import openmeteo_requests
import requests_cache
import pandas as pd
import xarray as xr
from retry_requests import retry
from typing import List, Dict, Optional

def fetch_historical_for_basins(
    centroids: pd.DataFrame, 
    start_date: str, 
    end_date: str, 
    hourly_variables: List[str] = [
        "temperature_2m", "relative_humidity_2m", "precipitation", "rain", "snowfall", 
        "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm", 
        "soil_moisture_100_to_255cm", "et0_fao_evapotranspiration", "surface_pressure", 
        "snow_depth_water_equivalent"
    ]
) -> Optional[xr.Dataset]:
    """Fetches historical hourly weather data for multiple basin centroids.

    Args:
        centroids (pd.DataFrame): DataFrame with 'basin_name', 'latitude', 'longitude'.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        hourly_variables (List[str]): List of hourly variables to fetch.

    Returns:
        Optional[xr.Dataset]: An xarray Dataset containing the historical data 
                              indexed by 'time' and 'basin', or None on error.
    """
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    
    all_basin_data = []

    print(f"Fetching historical data from {start_date} to {end_date} for {len(centroids)} basins...")

    for index, row in centroids.iterrows():
        basin_name = row['basin_name']
        latitude = row['latitude']
        longitude = row['longitude']
        
        print(f"  Fetching data for: {basin_name} (Lat: {latitude:.4f}, Lon: {longitude:.4f})")

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": hourly_variables,
            "models": "best_match" # Uses ERA5 Land / ERA5 depending on availability
        }
        
        try:
            responses = openmeteo.weather_api(url, params=params)
            response = responses[0] # Process first location (should only be one per call)

            # Process hourly data
            hourly = response.Hourly()
            
            # Create time range
            time_range = pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )
            
            hourly_data_dict = {"time": time_range}
            
            # Dynamically extract variables based on the requested list
            for i, var_name in enumerate(hourly_variables):
                hourly_data_dict[var_name] = hourly.Variables(i).ValuesAsNumpy()

            hourly_df = pd.DataFrame(data=hourly_data_dict)
            hourly_df = hourly_df.set_index('time') # Set time as index for xarray conversion
            
            # Convert to xarray Dataset
            basin_ds = xr.Dataset.from_dataframe(hourly_df)
            basin_ds = basin_ds.assign_coords(basin=basin_name) # Add basin coordinate
            basin_ds = basin_ds.expand_dims('basin') # Make basin a dimension
            
            # Add metadata from response
            basin_ds.attrs['latitude'] = response.Latitude()
            basin_ds.attrs['longitude'] = response.Longitude()
            basin_ds.attrs['elevation'] = response.Elevation()
            basin_ds.attrs['timezone'] = response.Timezone()
            basin_ds.attrs['timezone_abbreviation'] = response.TimezoneAbbreviation()
            basin_ds.attrs['utc_offset_seconds'] = response.UtcOffsetSeconds()
            basin_ds.attrs['api_source'] = url
            basin_ds.attrs['api_model'] = params['models']

            all_basin_data.append(basin_ds)

        except Exception as e:
            print(f"    Error fetching or processing data for {basin_name}: {e}")
            # Optionally continue to next basin or return None/raise error
            # continue 

    if not all_basin_data:
        print("No historical data could be fetched for any basin.")
        return None

    # Combine all basin datasets
    try:
        combined_historical_ds = xr.concat(all_basin_data, dim="basin")
        print(f"Successfully fetched and combined historical data for {len(combined_historical_ds.basin)} basins.")
        print(f"Historical Dataset dimensions: {dict(combined_historical_ds.dims)}")
        return combined_historical_ds
    except Exception as e:
        print(f"Error combining historical datasets: {e}")
        return None

# Example usage if run as a script (optional)
if __name__ == "__main__":
    # Define paths and parameters for testing
    basin_centroids_file = "../data/basin_centroids.csv" 
    test_start_date = "2023-01-01"
    test_end_date = "2023-01-07" 
    
    # Need to load centroids - assuming load_basin_centroids exists in another module or here
    # For standalone test, let's create a dummy DataFrame
    try:
        # Attempt to load from the actual file if fetch_basin_forecasts is accessible
        from fetch_basin_forecasts import load_basin_centroids
        centroids_df = load_basin_centroids(basin_centroids_file)
    except (ImportError, FileNotFoundError):
        print("Could not load actual centroids, using dummy data for testing.")
        centroids_df = pd.DataFrame({
            'basin_name': ['test_basin_1', 'test_basin_2'],
            'latitude': [52.52, 51.51],
            'longitude': [13.41, -0.13]
        })

    if centroids_df is not None:
        historical_data = fetch_historical_for_basins(centroids_df, test_start_date, test_end_date)
        
        if historical_data:
            print("Combined Historical Dataset Preview:")
            print(historical_data)
            
            # Example: Access data for a specific basin and time
            # print("Example Access:")
            # print(historical_data.sel(basin='test_basin_1', time='2023-01-01T12:00:00'))
        else:
            print("Failed to retrieve historical data.")
    else:
        print("Failed to load centroids for testing.")

