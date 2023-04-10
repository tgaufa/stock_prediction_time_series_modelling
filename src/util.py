import yaml
import joblib
from datetime import datetime
import pandas as pd

config_dir = "config/config.yaml"

def time_stamp() -> datetime:
    # Return current date and time
    return datetime.now()

def load_config() -> dict:
    # Try to load yaml file
    try:
        with open(config_dir, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError as fe:
        raise RuntimeError('Parameters file not found in path')
    
    # Return params in dict format
    return config

def pickle_load(file_path: str):
    # Load and return pickle file
    return joblib.load(file_path)

def pickle_dump(data, file_path: str) -> None:
    # Dump data into file
    joblib.dump(data, file_path)

params = load_config()
PRINT_DEBUG = params["print_debug"]

def print_debug(messages: str) -> None:
    # Check whether user wants to use print
    if PRINT_DEBUG == True:
        print(time_stamp(), messages)

# Check data statistics, since the column qty is a lot, then we summarize the describe feature in following function
def summary_dataset_describe(dataset):

    # Get the date index statistic info
    date_df = pd.Series(dataset.index).describe(include='datetime64', datetime_is_numeric=True)

    df = dataset.describe()
    # Get the minimum value for each row across all columns and convert it to a DataFrame
    row_min_df = df.min(axis=1).to_frame('Min')

    # Get the maximum value for each row across all columns and convert it to a DataFrame
    row_max_df = df.max(axis=1).to_frame('Max')

    # Concatenate the min and max DataFrames horizontally
    result_df = pd.concat([date_df, row_min_df, row_max_df], axis=1)


    return result_df