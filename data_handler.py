import yfinance as yf
import pandas as pd
import numpy as np

nvidia_data = yf.Ticker("NVDA")
nvda_price_df = nvidia_data.history(period='max')
nvda_price_df.to_csv('data/nvidia_stock_price_max_hist.csv')


def normalize(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def sliding_window(array):
    # Imporvement: Add variable window_length size
    """Sliding window to of size 3."""
    return np.column_stack([array[0:-2]], array[1:-1], array[2:])

