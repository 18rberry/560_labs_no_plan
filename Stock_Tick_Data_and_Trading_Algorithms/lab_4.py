import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np


def fetch_stock_data(tickers, start_date="2024-01-01"):
    """Download stock data for given tickers and combine into a single DataFrame."""
    df = yf.download(tickers, start=start_date, end=datetime.today(), group_by='ticker')

    all_dfs = []
    for ticker in tickers:
        stock_df = df[ticker].copy()
        stock_df['Ticker'] = ticker
        stock_df = stock_df.reset_index()
        all_dfs.append(stock_df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    columns_order = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
    combined_df = combined_df[columns_order]

    return combined_df


def data_clean_and_validation(df):
    """Clean and validate the stock data."""
    # Flag/remove negative values
    columns_subset = ['Open', 'High', 'Low', 'Close']
    mask = (df[columns_subset] >= 0).all(axis=1)
    df = df[mask]

    # Remove duplicates (duplicate timestamps for the same stock)
    df_unique = df.drop_duplicates(subset=['Date', 'Ticker'], keep='first')

    # Sort by timestamp and ticker ascending
    df_unique = df_unique.sort_values(by=['Date', 'Ticker'])

    # Fill null values with previous day data
    df_unique = df_unique.ffill()

    # Reset index
    df_unique = df_unique.reset_index(drop=True)

    return df_unique


def feature_engineering(df):
    """Add technical indicators and features for downstream algorithms."""
    processed_dfs = []
    tickers = df['Ticker'].unique()

    for ticker in tickers:
        df_ticker = df[df['Ticker'] == ticker].copy()
        df_ticker = df_ticker.sort_values(by=['Date'])

        # Short term moving average (SMA)
        df_ticker['SMA_20'] = df_ticker['Close'].rolling(window=20).mean()
        # Long term moving average
        df_ticker['SMA_50'] = df_ticker['Close'].rolling(window=50).mean()

        # Exponential moving average features (EMA)
        # More responsive to recent price changes than SMA
        df_ticker['EMA_12'] = df_ticker['Close'].ewm(span=12, adjust=False).mean()
        df_ticker['EMA_26'] = df_ticker['Close'].ewm(span=26, adjust=False).mean()

        # Pre-calculate buy/sell signals
        df_ticker['MA_Signal'] = np.where(
            df_ticker['SMA_20'] > df_ticker['SMA_50'],
            1,   # Buy signal
            -1   # Sell signal
        )

        df_ticker['EMA_Signal'] = np.where(
            df_ticker['EMA_12'] > df_ticker['EMA_26'], 1, -1
        )

        # Features for LSTM
        df_ticker['daily_return'] = df_ticker['Close'].pct_change()

        # Previous close -- gives LSTM recent history
        df_ticker['Close_lag1'] = df_ticker['Close'].shift(1)

        # Volume ratio & volume context
        df_ticker['volume_ma_20'] = df_ticker['Volume'].rolling(20).mean()
        df_ticker['volume_ratio'] = df_ticker['Volume'] / df_ticker['volume_ma_20']

        # Price range: shows volatility within day
        df_ticker['price_range'] = df_ticker['High'] - df_ticker['Low']

        # Rolling volatility
        df_ticker['volatility_20'] = df_ticker['Close'].rolling(20).std()

        # Day of week, helps capture weekly patterns
        df_ticker['day_of_week'] = df_ticker['Date'].dt.dayofweek

        processed_dfs.append(df_ticker)

    # Combine all DataFrames into one
    df_combined = pd.concat(processed_dfs, ignore_index=True)
    df_combined = df_combined.sort_values(['Date', 'Ticker']).reset_index(drop=True)

    return df_combined


def main():
    # Track stock data for Apple, Microsoft, Google, Tesla, IBM, Oracle, Amazon
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'IBM', 'ORCL', 'AMZN']

    print("Fetching stock data...")
    df = fetch_stock_data(tickers)

    print("Cleaning and validating data...")
    df = data_clean_and_validation(df)

    print("Performing feature engineering...")
    df = feature_engineering(df)

    output_file = 'cleaned_yfinance_data.csv'
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
    print(f"Total rows: {len(df)}")


if __name__ == "__main__":
    main()
