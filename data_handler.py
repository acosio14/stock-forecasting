import yfinance as yf
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from pathlib import Path

def extract_stock_close_price(
        ticker_name: str, period_length: str, filepath: str, columns: list[str],
) -> NDArray[np.float64]:
    """Extract the specific features from the desired stock through yahoo finance."""
    stock_data = yf.Ticker(ticker_name)
    close_price_df = stock_data.history(period=period_length)
    if not Path(filepath).exists():
        close_price_df.to_csv(filepath)
    return close_price_df[columns].to_numpy()

def data_transformation(array: NDArray[np.float64]):
    """Perform log transform and standardization of data."""
    log_array = np.log(array)
    return (log_array - log_array.mean()) / log_array.std()

def extract_test_set(array: NDArray[np.float64], ratio):
    array_length = len(array)
    test_set_size = int(ratio * array_length)
    step = int(array_length / test_set_size)

    return array[0::step]

def sliding_window(array: NDArray[np.float64], window_length: int, step: int) -> NDArray[np.float64]:
    """Sliding window of a variable size."""
    # Improvement: Add ability to change sliding window step
    # i.e. = [0,1,2], [1,2,3] or [0,1,2],[2,3,4] or [0,1,2],[3,4,5]
    # Make window size of 50 -> 40 train, 10 val, last will have extra
    array_length = array.shape[0]
    n = window_length - 1

    arr_list = []
    for i in range(window_length):
        idx1 = i
        idx2 = i - n
        if i == n:
            idx2 = array_length
        sequence = array[idx1:idx2:step]
        arr_list.append(sequence)

    return np.column_stack(arr_list)

def split_train_val_data(window: NDArray[np.float64], ratio: int):
    """ This is meant to split a np array of windows into train and val sets."""
    #[0...50]
    split_idx= int(ratio * window)
    train_set = window[:,:split_idx]
    val_set = window[:,split_idx:]

    return train_set, val_set

def split_feature_targets(data_set: NDArray[np.float64]):
    """Return X_set and y_set."""
    return data_set[:,:-1], data_set[:,-1]