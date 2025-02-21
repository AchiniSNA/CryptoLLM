import yfinance as yf
import pandas as pd
import datetime as dt

def load_and_preprocess_data(crypto_symbol: str, start_date: dt.date, end_date: dt.date):
    """Loads data from yfinance, preprocesses it, and returns train/test splits.

    Args:
        crypto_symbol: Cryptocurrency symbol (e.g., 'BTC-USD').
        start_date: Start date for data retrieval.
        end_date: End date for data retrieval.

    Returns:
        Tuple: (Y_train_df, Y_test_df) - pandas DataFrames for training and testing.
    """
    data = yf.download(crypto_symbol, start=start_date, end=end_date, interval='1d')

    if data.empty:
        return None, None  # Indicate no data

    data = data.reset_index()
    data = data[['Date', 'Close']]
    data.columns = ['ds', 'y']
    data['ds'] = pd.to_datetime(data['ds'])
    data['unique_id'] = crypto_symbol

    train_size = int(len(data) * 0.9)
    Y_train_df = data[:train_size]
    Y_test_df = data[train_size:].reset_index(drop=True)

    return Y_train_df, Y_test_df
