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

def normalize(array: NDArray[np.float64]):
    """Normalize the data."""
    return (array - np.min(array)) / (np.max(array) - np.min(array))

def sliding_window(array: NDArray[np.float64]):
    # Imporvement: Add variable window_length size
    """Sliding window of size 3."""
    return np.column_stack([array[0:-2]], array[1:-1], array[2:])

