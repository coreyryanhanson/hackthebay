import os
import re
import gzip
import numpy as np
import pandas as pd

def standardize_CMC_tidal_strings(value):
    if type(value) is not str:
        return value
    elif re.match(".*outgoing.*ebb", value, flags=re.IGNORECASE):
        return "Outgoing (Ebb)"
    elif re.match(".*incoming.*flood", value, flags=re.IGNORECASE):
        return "Incoming (Flood)"
    elif re.match(".?high", value, flags=re.IGNORECASE):
        return "High"
    elif re.match(".?low", value, flags=re.IGNORECASE):
        return "Low"
    else:
        return value

    
def standardize_CMC_water_surf_strings(value):
    if type(value) is not str:
        return value
    elif re.match(".*white.*caps", value, flags=re.IGNORECASE):
        return "White Caps"
    elif re.match(".?calm", value, flags=re.IGNORECASE):
        return "Calm"
    elif re.match(".?ripple", value, flags=re.IGNORECASE):
        return "Ripple"
    elif re.match(".?waves", value, flags=re.IGNORECASE):
        return "Waves"
    else:
        return value
    
    
def standardize_CMC_wind_strings(value):
    if type(value) is not str:
        return value
    elif re.match("\D?1\D*10\D*knots", value, flags=re.IGNORECASE):
        return "1 To 10 Knots"
    elif re.match("\D?10\D*20\D*knots", value, flags=re.IGNORECASE):
        return "10 To 20 Knots"
    elif re.match(".?Calm", value, flags=re.IGNORECASE):
        return "Calm"
    elif re.match("\D?20\D*30\D*knots", value, flags=re.IGNORECASE):
        return "20 To 30 Knots"
    elif re.match("\D?40\D*knots", value, flags=re.IGNORECASE):
        return "Above 40 Knots"
    else:
        return value


def standardize_CMC_wind_dir_strings(value):
    if type(value) is not str:
        return value
    else:
        return value.upper()
    
def standardize_CMC_weather_strings(value):
    if type(value) is not str:
        return value
    elif re.match(".*partly.*cloudy", value, flags=re.IGNORECASE):
        return "Partly cloudy"
    elif re.match(".*intermittent.*rain", value, flags=re.IGNORECASE):
        return "Intermittent rain"
    elif re.match(".*fog.*haze", value, flags=re.IGNORECASE):
        return "fog/haze"
    elif re.match(".?sunny", value, flags=re.IGNORECASE):
        return "Sunny"
    elif re.match(".?overcast", value, flags=re.IGNORECASE):
        return "Overcast"
    elif re.match(".?rain", value, flags=re.IGNORECASE):
        return "Rain"
    elif re.match(".?drizzle", value, flags=re.IGNORECASE):
        return "Drizzle"
    elif re.match(".?snow", value, flags=re.IGNORECASE):
        return "Snow"
    else:
        return value
    
def extract_timespan(data, col_dt, target_col):
    """Returns a new dataframe with data restricted to the date range in a target column."""
    
    df = data.copy()
    idx = df[df[target_col].isna()==False].set_index(col_dt).index
    
    # Function to remove time precision on date values for the min/max.
    strip = lambda x: x.strftime('%Y-%m-%d')
    
    # Provides the minimum/maximum values of the date range.
    start, end = strip(idx.min()), strip(idx.max())
    
    df = df[(df[col_dt] >= start) & (df[col_dt] <= end)]
    return df

def noaa_gzip_to_raw(path):
    with gzip.open(path, "rb") as f:
        data = f.read().splitlines()
    return data

def select_noaa_files(station_names, dir_path, start_year, end_year):
    all_paths = pd.Index([])
    for year in np.arange(start_year,end_year+1):
        dir_path_year = os.path.join(dir_path, str(year), "")
        all_files = pd.Index(os.listdir(dir_path_year))
        stations = station_names + f"-{year}.gz"
        intersection = stations.intersection(all_files)
        paths = dir_path_year + intersection
        all_paths = all_paths.union(paths, sort=False)
    return all_paths