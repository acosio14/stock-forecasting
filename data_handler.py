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

def sliding_window(array: NDArray[np.float64], window_length: int) -> NDArray[np.float64]:
    """Sliding window of a variable size."""
    # Improvement: Add ability to change sliding window step
    # i.e. = [0,1,2], [1,2,3] or [0,1,2],[2,3,4] or [0,1,2],[3,4,5]
    array_length = array.shape[0]
    n = window_length - 1

    arr_list = []
    for i in range(window_length):
        idx1 = i
        idx2 = i - n
        if i == n:
            idx2 = array_length
        sequence = array[idx1:idx2]
        arr_list.append(sequence)

    return np.column_stack(arr_list)

