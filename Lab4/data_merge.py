import pandas as pd

# Load data
df_prices = pd.read_csv('Lab4/cleaned_yfinance_data.csv')
df_signals = pd.read_csv('Lab4/portfolio_signals.csv')

# Format columns
df_prices = df_prices[['Date', 'Ticker', 'Close']]
df_signals = df_signals[['Date', 'Ticker', 'Signal']]
df_prices['Date'] = pd.to_datetime(df_prices['Date'])
df_signals['Date'] = pd.to_datetime(df_signals['Date'])

# Merge price and signal data
df = pd.merge(df_signals, df_prices, on=['Date', 'Ticker'], how='left')

df.to_csv('Lab4/price_signal_data.csv', index=False)


