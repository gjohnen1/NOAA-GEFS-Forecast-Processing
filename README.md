# NOAA-GEFS-Forecast-Processing
This repo fetches NOAA GEFS 35-day forecasts by location, fills missing timesteps (0–240 hours) to create hourly series, structures the data into xarray Datasets (dimensions: basin id, time, lead time) and exports them as NetCDF files.
