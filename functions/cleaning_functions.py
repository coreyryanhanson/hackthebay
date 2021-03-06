import os
import re
import math
import gzip
import numpy as np
import pandas as pd


def pandas_series_autocheck(df, column, index, default_value=np.nan):
    if column in df.columns:
        return df[column]
    else:
        return pd.Series(default_value, index=index)


def degrees_to_cos(x):
    """Takes the cosine of an angle given in degrees."""
    return math.cos(math.radians(x))


def degrees_to_sin(x):
    """Takes the sine of an angle given in degrees."""
    return math.sin(math.radians(x))


def convert_to_wind_velocity(data, angle_col, speed_col, new_col_base = "Wind_Vel", drop_speed=True, drop_angle=True):
    """Takes a dataframe and the column names for angle and speed to break them into their x and y component vectors."""
    
    df = data.copy()
    
    # Creates new columns using the sine and cosine function.
    for suffix, function in {"_x":degrees_to_cos, "_y":degrees_to_sin}.items():
        df[new_col_base+suffix] = df[angle_col].map(function) * df[speed_col]
    
    # Drops original columns depending on function arguments.
    for column, drop in {angle_col:drop_angle, speed_col:drop_speed}.items():
        if drop:
            df.drop(columns=column, inplace=True)
    return df


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

def remove_suspect_noaa_data(data, quality_column, value_columns, bad_codes):
    """Iterates through a list of quality codes to remove observations from NOAA data."""
    
    df = data.copy()
    idx = df.index
    conditions = pd.Series(False, index=idx)
    for code in bad_codes:
        conditions = (conditions | (df[quality_column] == code))
    
    for column in [*value_columns, quality_column]:
        df[column] = np.where(conditions, np.nan, df[column])

    return df

def reduce_time_resolution_in_multindex(data, timekey, freq, other_keys):
    """Aggregates a groupby function based on multiple indexes, while resampling and dropping nans of a specified date index."""
    
    df = data.copy()
    
    #Takes the mean of samples if the datatype is a float (All measured values were). Otherwise takes the first item.
    agg_list = generate_agg_list(df, [timekey, *other_keys], ["float"], "mean", "first")
    
    return df.groupby([pd.Grouper(key=timekey, freq=freq), *other_keys]).agg(agg_list).dropna(how="all").reset_index()


def generate_agg_list(df, groupby_keys, main_dtypes, main_func, fallback_func):
    
    # Getting column names for the non index items to be grouped.
    cols = df.columns.difference(groupby_keys, sort=False)
    
    # Functions are chosen between main and fallback ones if the datatype is a match.
    functions = [main_func if df[col].dtype in main_dtypes else fallback_func for col in cols]
    
    #Returns a dictionary to use to aggregate the groupby function.
    return dict(zip(cols,functions))


def accumulate_noaa_observations(data, date_column, period_column, value_column, id_columns, cycles, available_period, time_unit="H"):
    """This function will examine a dataframe with mixed period data and cycle through summing up earlier values in that period if they
    exist. There is a tradeoff with speed, precision, and sparcity and this function rounds the time values to achieve its
    goal as quickly as possible. If any value is incomplete over the interval, it will be considered a nan in the final sum."""
    

    # Saves the original index and creates a copy of the dataframe where index values are reset to prevent side effects
    # and offset bugs being introduced during the merge.
    idx = data.index
    df = data.reset_index().copy()
    
    # Restrict values to ones that match the specified period.
    df[value_column] = np.where(df[period_column]==str(available_period), df[value_column], np.nan)
    
    # Round to hours to prevent differences in minutes hindering the search for subsequent periods.
    df[date_column] = df[date_column].round(time_unit)
    
    # These columns will contain the list of unique identifiers, dates, and period to execute the joins.
    relevant_columns = [*id_columns, date_column, period_column]
    
    # Creates the reference that will be joined on the offset time values.
    reference = df[relevant_columns+[value_column]]
    
    # The values will be aggregated in this defensive copy.
    aggregate = reference.copy()
    
    # The reference needs to remove duplicate rows for a true 1:1 join.
    reference = reference.groupby(relevant_columns).first()
    
    # ITerates through the cycles of the provided time, offsetting the dates and joining available values from earlier observations
    # on those offsetted indexes.
    for c in np.arange(1, cycles):
        aggregate[date_column] = df[date_column] - pd.Timedelta(c * available_period, unit=time_unit)
        aggregate = aggregate.merge(reference, how="left", on=relevant_columns, suffixes=(None, c))
    
    # Ensures that the only values included in the sum are observations.
    aggregate.drop(columns=relevant_columns, inplace=True)
    
    # Returns the index back to its original value.
    aggregate.index = idx
    
    # Returns the sum of the columns. If any value is missing, it will be a nan.
    return aggregate.sum(axis=1, skipna=False)


def aggregate_noaa_quantity(data, output_column, date_column, period_column, value_column, id_columns,
                            instances, desired_period, available_period, time_unit="H"):
    """This function will parse NOAA variable periods in specified columns of a NOAA DataFrame and save them to their own distinct
    column, aggregating values from past data values if needed."""
    
    # Creates a defensive copy of the DataFrame and backs up the index.
    df = data.copy()
    idx = df.index
    
    # User created function to check whether a column exists in the DataFrame. If not a new one is populated by nans.
    df[output_column] = pandas_series_autocheck(df, output_column, idx)
    
    # Periods must be cleanly divisible or results will be inaccurate.
    if desired_period % available_period:
        raise ValueError("Warning time periods are out of sync and will result in incorrect values.")
    cycles = math.ceil(desired_period/available_period)
    
    for i in np.arange(1,instances+1):
        
        # This if statement will save computation time if the period column is not greater than the available period. Replacement
        # values are either taken as is with a perfect match or the sums are calculated with a user driven function offsetting
        # based on the amount of cycles required and conservatively adding if observations exist in all time offsets.
        if cycles == 1:
            replacement_values = df[value_column+str(i)]
        else:
            replacement_values = accumulate_noaa_observations(df, date_column, period_column+str(i), value_column+str(i),
                                                              id_columns, cycles, available_period)
        
        # Requires new values to be only added when existing ones are not available, and the period is a correct match.
        condition = (df[output_column].isna()) & (df[period_column+str(i)] == str(available_period))
        df[output_column] = np.where(condition, replacement_values, df[output_column])
            
    return df


def parse_ghcn_inventory_to_df(filepath):
    columns = ["ID", "Latitude", "Longitude", "Element", "Start_Date", "End_Date"]
    return pd.read_csv(filepath, delim_whitespace=True, names=columns)


def parse_ghcn_daily_by_year_File(filepath):
    columns = ["ID", "Date", "Element", "Data_Value", "M-Flag", "Q-Flag", "S-Flag"]
    return pd.read_csv(filepath, names=columns, index_col=False)


def add_earlier_date_values(data, date_col, id_cols, value_col, days):
    """Adds columns based on past day values to a dataframe."""
    
    df=data.copy()
    for i in np.arange(1,days+1):
        suffix = "_" + str(i * 24) + "H"
        temp_column = date_col + str(i)
        df[temp_column] = df[date_col] - pd.Timedelta(i, unit="D")
        df = df.merge(df[[*id_cols, date_col, value_col]], how="left", left_on=[*id_cols, temp_column],
                      right_on=[*id_cols, date_col], suffixes=(None, suffix))
        df = df.drop(columns = [temp_column, date_col+suffix])
    return df


def reduce_from_ghcn_daily_by_year(df, id_col, ele_col, flag_col, stations, elements):
    condition = (df[id_col].isin(stations)) & (df[ele_col].isin(elements)) & (df[flag_col].isna())
    return df[condition]


def expand_ghcn_daily_by_year(data, ele_col, val_col, element_dict, groupby_cols=None):
    df = data.copy()
    
    for element in element_dict.keys():
        df[element] = np.where(df[ele_col] == element, df[val_col], np.nan)
        
        # Adds unit conversions if they exists.
        if element_dict[element]:
            df[element] = df[element]/element_dict[element]
            
    df.drop(columns=[ele_col, val_col], inplace=True)
    if groupby_cols:
        df = df.groupby(groupby_cols).first().reset_index()
    return df