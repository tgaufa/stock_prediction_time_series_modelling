import preprocessing
import util as utils
import pandas as pd
import numpy as np

def test_transform_data():
    #arrange
    config = utils.load_config()

    data = {
    "AAPL": [150.23, 151.34, 149.87, 152.15],
    "GOOG": [2800.34, 2795.12, 2810.54, 2806.90],
    "TSLA": [700.45, 705.13, 710.20, 695.78],
    "BMRI.JK": [3300.21, 3312.55, 3295.00, 3320.10],
    }

    index = pd.date_range(start="2023-04-12", periods=4, freq="D")

    mock_data = pd.DataFrame(data, index=index)

    #act
    processed_data = preprocessing.transform_to_stock_return(dataset=mock_data, params=config)

    #assert
    assert processed_data.shape == (1,5)
