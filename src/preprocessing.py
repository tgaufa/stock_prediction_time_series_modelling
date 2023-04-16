import pandas as pd
import numpy as np
import seaborn as sns
import src.util as util

def load_dataset(config_data: dict) -> pd.DataFrame:
    
    # Load every set of data
    clean_data = util.pickle_load(config_data['clean_dataset_path'])

    x_train = util.pickle_load(config_data["train_set_path"][0])
    y_train = util.pickle_load(config_data["train_set_path"][1])

    x_valid = util.pickle_load(config_data["valid_set_path"][0])
    y_valid = util.pickle_load(config_data["valid_set_path"][1])

    x_test = util.pickle_load(config_data["test_set_path"][0])
    y_test = util.pickle_load(config_data["test_set_path"][1])

    # Concatenate x and y each set
    train_set = pd.concat([x_train, y_train], axis = 1)
    valid_set = pd.concat([x_valid, y_valid], axis = 1)
    test_set = pd.concat([x_test, y_test], axis = 1)

    # Return 3 set of data
    return clean_data, train_set, valid_set, test_set

# as the way to normalize all of data value, its relevant if we change them into return percentage.
# the advantage are: 
# 1. the data value will vary from -0.5 to +0.5. While its possible, its less likely stock change will be up/down more than 50% within 2 days. 
# 2. the stock return is something we want to know anyway therefore its a representative approach in this case

def transform_to_stock_return(dataset, params):
    # define the return for all stock based on the next day of its price change percentage 
    dataset = (dataset.shift(periods=1)-dataset)*100/dataset
    
    #define the target return column name
    target_return_column_name = f"{params['target']} Return D+2"
    
    # add additional column of our targeted stock return
    dataset[target_return_column_name] = dataset[params['target']].shift(periods=-2)

    # handling missing value of shifted targeted column & its reference column
    dataset.dropna(subset=params['target'], inplace=True)
    dataset.dropna(subset=target_return_column_name, inplace=True)

    # handling missing value of the remaining columns
    #dataset.fillna(0, inplace=True)

    return dataset

def keep_correlated_features(train_set, val_set, test_set, params):
    #define the target return column name
    target_return_column_name = f"{params['target']} Return D+2"

    # define the correlated features
    corr_stock = train_set.corrwith(train_set[target_return_column_name], axis=0).nlargest(10).sort_values(ascending=True)

    # keep correlated features
    train_set = train_set[corr_stock.index]
    val_set = val_set[corr_stock.index]
    test_set = test_set[corr_stock.index]

    return corr_stock, train_set, val_set, test_set

if __name__ == "__main__":
    config_data = util.load_config()

    clean_data, train_set, valid_set, test_set = load_dataset(config_data)

    train_set_feng = transform_to_stock_return(dataset=train_set, params=config_data)

    val_set_feng = transform_to_stock_return(dataset=valid_set, params=config_data)

    test_set_feng = transform_to_stock_return(dataset=test_set, params=config_data)

    corr_stock, train_set_feng, val_set_feng, test_set_feng = keep_correlated_features(train_set= train_set_feng, val_set= val_set_feng, test_set= test_set_feng,params= config_data)

    X_train = train_set_feng.iloc[:,:-1]
    y_train = train_set_feng.iloc[:,-1]

    X_val = val_set_feng.iloc[:,:-1]
    y_val = val_set_feng.iloc[:,-1]

    X_test = test_set_feng.iloc[:,:-1]
    y_test = test_set_feng.iloc[:,-1]

    util.pickle_dump(X_train, config_data["train_feng_set_path"][0])
    util.pickle_dump(y_train, config_data["train_feng_set_path"][1])

    util.pickle_dump(X_val, config_data["valid_feng_set_path"][0])
    util.pickle_dump(y_val, config_data["valid_feng_set_path"][1])

    util.pickle_dump(X_test, config_data["test_feng_set_path"][0])
    util.pickle_dump(y_test, config_data["test_feng_set_path"][1])

