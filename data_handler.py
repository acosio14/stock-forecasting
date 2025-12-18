from pathlib import Path

import numpy as np
import yfinance as yf
from numpy.typing import NDArray


def extract_stock_close_price(
    ticker_name: str,
    period_length: str,
    filepath: str,
    columns: list[str],
) -> NDArray[np.float64]:
    """Extract the specific features from the desired stock through yahoo finance."""
    stock_data = yf.Ticker(ticker_name)
    close_price_df = stock_data.history(period=period_length)
    if not Path(filepath).exists():
        close_price_df.to_csv(filepath)
    return close_price_df[columns].to_numpy()


def log_transformation(array: NDArray[np.float64]) -> NDArray[np.float64]:
    """Perform log transform and standardization of data."""
    return np.log(array)


def standardization(array: NDArray[np.float64]):
    """Standardize data."""
    array_mean = array.mean()
    array_std = array.std()
    z_score = (array - array_mean) / array_std

    return z_score, array_mean, array_std


def split_data(array: NDArray[np.float64], ratio: float):
    array_length = len(array)
    data_split_size = int(ratio * array_length)
    step = int(array_length / data_split_size)

    split_indicies = [*range(0, array_length, step)]
    train_indicies = [idx for idx in range(array_length) if idx not in split_indicies]

    return array[train_indicies], array[split_indicies]


def sliding_window(
    array: NDArray[np.float64], window_length: int, step: int,
) -> NDArray[np.float64]:
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


def split_feature_targets(data_set: NDArray[np.float64]):
    """Return X_set and y_set."""
    return data_set[:, :-1, np.newaxis], data_set[:, -1:]
