import pandas as pd

# Load data
df_prices = pd.read_csv('Lab4/cleaned_yfinance_data.csv')
df_signals_baseline = pd.read_csv('Lab4/baseline_signals.csv')
df_signals_lstm = pd.read_csv('Lab4/portfolio_signals.csv')

# Format columns
df_prices = df_prices[['Date', 'Ticker', 'Close']]
df_signals_baseline = df_signals_baseline[['Date', 'Ticker', 'Signal', 'Confidence']]
df_signals_lstm = df_signals_lstm[['Date', 'Ticker', 'Signal', 'Confidence']]

df_prices['Date'] = pd.to_datetime(df_prices['Date'])
df_signals_baseline['Date'] = pd.to_datetime(df_signals_baseline['Date'])
df_signals_lstm['Date'] = pd.to_datetime(df_signals_lstm['Date'])

# Merge price and signal data
df_baseline = pd.merge(df_signals_baseline, df_prices, on=['Date', 'Ticker'], how='left')
df_lstm = pd.merge(df_signals_lstm, df_prices, on=['Date', 'Ticker'], how='left')

df_baseline.to_csv('Lab4/price_signal_baseline.csv', index=False)
df_lstm.to_csv('Lab4/price_signal_lstm.csv', index=False)


