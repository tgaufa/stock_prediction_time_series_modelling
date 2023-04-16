from tqdm import tqdm
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
import joblib
import os
import yaml
import src.util as util
from sklearn.model_selection import TimeSeriesSplit

def read_raw_data(config: dict) -> pd.DataFrame:
    # Load and define stock ticker list at IDX
    stock_list = pd.read_excel(config['raw_dataset_dir'])

    # Add new column with a value suitable to ticker name at yfinance
    stock_list['ticker.jk'] = stock_list['Kode'] + config['ticker_ext']

    # Take only the needed column and change it from df to list
    ticker_list = stock_list['ticker.jk'].tolist()
    
    # Define the date range parameter
    start_date = config['start_date']
    end_date = date.today()
    interval = config['interval_date']

    # Download stock data from yfinance
    stock_data = {}
    for ticker in tqdm(ticker_list, desc='Downloading stock data'):
        stock_data[ticker] = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)

    # Convert the dictionary to a pandas DataFrame with a MultiIndex
    dataset = pd.concat(stock_data, axis=1)

    # re adjust the table only to show the required column (adj. closing price)
    dataset = dataset[dataset.columns[4::6]]
    dataset.columns = dataset.columns.droplevel(1)

    # return raw dataset
    return dataset

def check_data(input_data, params, print_errors=True):

    error_messages = []
    error_stock_tickers = []
    #input_data = input_data.fillna(0)
    try:
        # Check index data types
        assert input_data.index.dtype == params['datetime_index'], 'an error occurs in index format, should be datetime.'

        # Check index data type & range
        for column in input_data.columns:
            if input_data[column].dtype != 'float64':
                error_messages.append(f"Column ({column}) has a non-float data type")
                error_stock_tickers.append(column)

            if not (input_data[column] >= 0).sum() == len(input_data):
                error_messages.append(f'an error occurs in {column} column')
                if column not in error_stock_tickers:
                    error_stock_tickers.append(column)
        
        if error_messages:
            total_errors = len(error_messages)
            error_summary = f"\nTotal errors: {total_errors} errors out of {len(input_data.columns)}\n"
            raise AssertionError(error_summary + "\n".join(error_messages))
    
    except AssertionError as e:
        if print_errors:
            print(e)
    
    return error_stock_tickers

if __name__ == "__main__":
    # 1. Load configuration file
    config_data = util.load_config()

    # 2. Read all raw dataset
    raw_dataset = read_raw_data(config_data)

    # change the index format from object into datetime 
    raw_dataset.index = pd.to_datetime(raw_dataset.index)

    # sort the date index
    raw_dataset = raw_dataset.sort_index(ascending=True)

    # Delete unrequired rows & columns where all its value is NaN
    raw_dataset.dropna(axis=0, thresh=0.01*len(raw_dataset.columns), inplace=True)
    raw_dataset.dropna(axis=1, thresh=0.01*len(raw_dataset.index), inplace=True)
    raw_dataset.dropna(axis=1, how='any', inplace=True)

    # 3. Save Raw Dataset
    util.pickle_dump(raw_dataset, config_data['raw_dataset_path'])

    # Check the error stock in the dataset
    check_data(raw_dataset, config_data)

    # Found error in stock above (SCPI.JK), which after checking through news and yfinance data,
    # it is already delisted since 2013. So this stock should be removed since it is considered
    # as an anomaly.
    error_stock_tickers = check_data(raw_dataset, config_data, print_errors=False)
    raw_dataset.drop(error_stock_tickers, axis=1, inplace=True)

    # Recheck the data and found no issue
    check_data(raw_dataset, config_data)

    # Anomaly Handling; change into actual value based on other source
    raw_dataset['BMRI.JK'].loc['2023-03-30'] = float(5112)
    raw_dataset['MYOR.JK'].loc['2022-06-14'] = float(1602.730957)

    util.pickle_dump(raw_dataset, config_data["clean_dataset_path"])


    # Initialize TimeSeriesSplit object
    n_splits = 3
    tscv = TimeSeriesSplit(n_splits = n_splits)

    # Get the train & test_val indices at the last split
    for train_index, test_val_index in tscv.split(raw_dataset):
        pass

    # Calculate the size of the test and validation sets
    test_val_size = len(test_val_index)
    test_size = val_size = test_val_size // 2

    # Define test and validation indices
    val_index = test_val_index[:test_size]
    test_index = test_val_index[test_size:]

    # Extract the train, test, and validation sets
    train = raw_dataset.iloc[train_index]
    val = raw_dataset.iloc[val_index]
    test = raw_dataset.iloc[test_index]

    # Split feature and target columns for train, test, and validation sets
    feature_columns = raw_dataset.drop([config_data['target']], axis=1).columns
    target_column = config_data['target']
    X_train, y_train = train[feature_columns], train[target_column]
    X_test, y_test = test[feature_columns], test[target_column]
    X_val, y_val = val[feature_columns], val[target_column]

    util.pickle_dump(X_train, config_data["train_set_path"][0])
    util.pickle_dump(y_train, config_data["train_set_path"][1])

    util.pickle_dump(X_val, config_data["valid_set_path"][0])
    util.pickle_dump(y_val, config_data["valid_set_path"][1])

    util.pickle_dump(X_test, config_data["test_set_path"][0])
    util.pickle_dump(y_test, config_data["test_set_path"][1])
