"""
DSCI-560 Lab 4 - SMA Crossover Baseline Model for Stock Trading Signals
==================================================================================
Baseline model using Simple Moving Average (SMA) crossover strategy to generate
buy/sell/hold signals for the portfolio.

Strategy:
  - SMA_20 > SMA_50 (golden cross)  -> Buy signal (1)
  - SMA_20 < SMA_50 (death cross)   -> Sell signal (-1)

Output: baseline_signals.csv with columns [Date, Ticker, Signal, Confidence]
        for the mock trading environment to consume.

Authors: Lab 4 Team
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'IBM', 'ORCL', 'AMZN']


def load_data(filepath):
    """Load the cleaned CSV with pre-computed SMA features."""
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    print(f"Loaded {len(df)} rows, {df['Ticker'].nunique()} tickers")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

    return df


def generate_sma_signals(df_ticker):
    """
    Generate trading signals from SMA crossover.

    Signal logic:
      - SMA_20 > SMA_50 -> Buy (1)
      - SMA_20 < SMA_50 -> Sell (-1)

    Confidence is based on the magnitude of the SMA spread relative
    to the stock price -- a wider spread means a stronger signal.
    """
    signals = df_ticker['MA_Signal'].copy()  # already computed: 1 or -1

    # compute confidence as normalized SMA spread
    spread = (df_ticker['SMA_20'] - df_ticker['SMA_50']).abs()
    price = df_ticker['Close']
    # normalized spread as percentage of price, capped at 1.0
    confidence = (spread / price).clip(upper=1.0)
    # scale to [0.5, 1.0] range so even weak signals have reasonable confidence
    if confidence.max() > 0:
        confidence = 0.5 + 0.5 * (confidence / confidence.max())
    else:
        confidence = confidence + 0.5

    return signals, confidence


def plot_sma_signals(df_ticker, ticker):
    """Plot price with SMA lines and buy/sell markers for one stock."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(df_ticker['Date'], df_ticker['Close'], label='Close', color='black', alpha=0.7, linewidth=1)
    ax1.plot(df_ticker['Date'], df_ticker['SMA_20'], label='SMA 20', color='blue', alpha=0.8, linewidth=1)
    ax1.plot(df_ticker['Date'], df_ticker['SMA_50'], label='SMA 50', color='red', alpha=0.8, linewidth=1)

    # mark buy/sell signals
    buys = df_ticker[df_ticker['MA_Signal'] == 1]
    sells = df_ticker[df_ticker['MA_Signal'] == -1]
    ax1.scatter(buys['Date'], buys['Close'], marker='^', color='green', s=20, alpha=0.6, label='Buy Signal')
    ax1.scatter(sells['Date'], sells['Close'], marker='v', color='red', s=20, alpha=0.6, label='Sell Signal')

    ax1.set_title(f'{ticker} - SMA Crossover Signals')
    ax1.set_ylabel('Price ($)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # volume subplot
    ax2.bar(df_ticker['Date'], df_ticker['Volume'], color='steelblue', alpha=0.5, width=1)
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'plots/{ticker}_sma_baseline.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: plots/{ticker}_sma_baseline.png")


def main():
    print("=" * 70)
    print("DSCI-560 Lab 4 - SMA Crossover Baseline Model")
    print("=" * 70)

    df = load_data('cleaned_yfinance_data.csv')

    all_signals = []

    for ticker in TICKERS:
        print(f"\n{'='*60}")
        print(f"Processing {ticker}")

        # filter and sort data for this ticker
        df_ticker = df[df['Ticker'] == ticker].copy()
        df_ticker = df_ticker.sort_values('Date').reset_index(drop=True)

        # drop rows where SMAs aren't available yet (first ~50 rows)
        df_ticker = df_ticker.dropna(subset=['SMA_20', 'SMA_50']).reset_index(drop=True)
        print(f"  Usable rows: {len(df_ticker)}")

        # generate SMA signals
        df_ticker['sma_signal'], df_ticker['confidence'] = generate_sma_signals(df_ticker)

        # signal distribution
        buy_count = (df_ticker['sma_signal'] == 1).sum()
        sell_count = (df_ticker['sma_signal'] == -1).sum()
        print(f"  Signals: Buy={buy_count}, Sell={sell_count}")

        # collect signals for output CSV
        signal_df = df_ticker[['Date', 'sma_signal', 'confidence']].copy()
        signal_df = signal_df.rename(columns={'sma_signal': 'Signal', 'confidence': 'Confidence'})
        signal_df['Ticker'] = ticker
        signal_df['Confidence'] = signal_df['Confidence'].round(4)
        all_signals.append(signal_df)

        # plot
        plot_sma_signals(df_ticker, ticker)

    # combine all signals
    print("\n" + "=" * 70)
    print("Generating Baseline Signals CSV")
    print("=" * 70)

    baseline_signals = pd.concat(all_signals, ignore_index=True)
    baseline_signals = baseline_signals[['Date', 'Ticker', 'Signal', 'Confidence']]
    baseline_signals = baseline_signals.sort_values(['Date', 'Ticker']).reset_index(drop=True)

    baseline_signals.to_csv('baseline_signals.csv', index=False)
    print(f"Saved baseline_signals.csv with {len(baseline_signals)} rows")

    signal_map = {-1: 'Sell', 1: 'Buy'}
    print("\nOverall signal distribution:")
    for sig_val, sig_name in signal_map.items():
        count = (baseline_signals['Signal'] == sig_val).sum()
        pct = count / len(baseline_signals) * 100
        print(f"  {sig_name}: {count} ({pct:.1f}%)")

    print("\n" + "=" * 70)
    print("Done! Files generated:")
    print("  - baseline_signals.csv           (SMA trading signals for mock trading env)")
    print("  - plots/*_sma_baseline.png       (per-stock signal charts)")
    print("=" * 70)


if __name__ == "__main__":
    main()
