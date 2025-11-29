import yfinance as yf
import pandas as pd

nvidia_data = yf.Ticker("NVDA")
nvda_price_df = nvidia_data.history(period='max')
nvda_price_df.to_csv('data/nvidia_stock_price_max_hist.csv')
