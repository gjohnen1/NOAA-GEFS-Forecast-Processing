# NOAA GEFS Forecast Processing

This repository processes NOAA GEFS 35-day forecast data for specific locations. It fetches data from the [NOAA GEFS archive](https://dynamical.org/catalog/noaa-gefs-forecast-35-day/), forward fills missing timesteps, and exports the processed data as NetCDF files.

## Features
- Fetches Zarr datasets for specific locations.
- Forward fills missing timesteps to create an hourly time series.
- Exports processed data as NetCDF files.

## Repository Structure
```
/ ├── README.md ├── LICENSE ├── requirements.txt ├── data/ │ └── basin_shapefiles/ ├── notebooks/ ├── src/ │ ├── init.py │ ├── config.py │ ├── fetch_data.py │ ├── process_data.py │ ├── export_data.py │ └── main.py ├── tests/ └── docs/
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sngrj0hn/NOAA-GEFS-Forecast-Processing.git
   cd NOAA-GEFS-Forecast-Processing
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Configure the `src/config.py` file with your settings. For example:
   - `API_KEY`: Your NOAA API key for accessing the forecast data.
   - `LOCATIONS`: A list of latitude and longitude pairs for the locations you want to process.
   - `OUTPUT_DIR`: The directory where processed NetCDF files will be saved.
2. Run the processing pipeline:
   ```bash
   python src/main.py
   ```

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
