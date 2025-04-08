#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup Project Structure

This script creates the necessary directory structure for the NOAA GEFS Forecast Processing project.
Run this script once to prepare the workspace before using the project.
"""

import os
import sys

def create_directory_structure():
    """Create the project directory structure."""
    # Define the base directory (where this script is located)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define directories to create
    directories = [
        os.path.join(base_dir, "data"),
        os.path.join(base_dir, "data", "basin_shapefiles"),
        os.path.join(base_dir, "data", "basin_forecasts"),
        os.path.join(base_dir, "notebooks"),
        os.path.join(base_dir, "src")
    ]
    
    # Create each directory if it doesn't exist
    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"Created directory: {directory}")
            except Exception as e:
                print(f"Error creating directory {directory}: {e}")
    
    print("\nProject directory structure created successfully!")
    print("\nNext steps:")
    print("1. Place your basin shapefiles in the 'data/basin_shapefiles' directory")
    print("2. Run the 'notebooks/extract_basin_centroids.ipynb' notebook")
    print("3. Run the 'notebooks/fetch_basin_forecasts.ipynb' notebook")

if __name__ == "__main__":
    create_directory_structure()
