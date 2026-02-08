import pandas as pd
import numpy as np
import sys


def simulation(df, current_positions, STARTING_CASH):
    current_cash = STARTING_CASH
    history = []

    # Simulate daily trades
    for date, group in df.groupby('Date'):
        # Execute SELL orders
        for _, stock in group.iterrows():
            stock_ticker = stock['Ticker']
            close_price = stock['Close']
            signal = stock['Signal']
            
            # Sell stock
            if signal == -1 and current_positions[stock_ticker] > 0:
                current_cash += close_price * current_positions[stock_ticker]
                current_positions[stock_ticker] = 0

        # Identify BUY candidates
        buy_candidates = group[group['Signal'] == 1]
        total_confidence = buy_candidates['Confidence'].sum()

        # Execute BUY orders
        if not buy_candidates.empty and current_cash > 0:
            for _, stock in buy_candidates.iterrows():
                stock_ticker = stock['Ticker']
                close_price = stock['Close']
                confidence = stock['Confidence']

                # Allocate cash
                weight = confidence / total_confidence
                allocated_cash = weight * current_cash

                # Buy shares
                current_cash -= allocated_cash
                current_positions[stock_ticker] += allocated_cash / close_price

        # Calculate portfolio value
        portfolio_value = current_cash

        for _, row in group.iterrows():
            portfolio_value += row['Close'] * current_positions[row['Ticker']]

        history.append({'Date': date, 'Portfolio Value': portfolio_value})

    return history

            
if __name__ == '__main__':
    input_path = sys.argv[1]

    # Load price/signal data
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])

    # Intialize portfolio
    STARTING_CASH = 100000

    current_positions = {'AAPL': 0,
                         'MSFT': 0,
                         'GOOGL': 0,
                         'TSLA': 0,
                         'IBM': 0,
                         'ORCL': 0,
                         'AMZN': 0}
    
    # Run simulation
    history = simulation(df, current_positions, STARTING_CASH)

    # Format output
    portfolio_df = pd.DataFrame(history)
    print(portfolio_df, '\n')

    # Calculate performance metrics
    portfolio_value = portfolio_df['Portfolio Value'].iloc[-1]

    total_days = portfolio_df.shape[0]
    annualized_return = (portfolio_value / STARTING_CASH) ** (252 / total_days) - 1

    portfolio_df['Daily Return'] = portfolio_df['Portfolio Value'].pct_change()
    mean_return = portfolio_df['Daily Return'].mean()
    std_return = portfolio_df['Daily Return'].std()

    sharpe_ratio = (mean_return / std_return) * np.sqrt(252)

    # Display metrics
    print(f'Starting portfolio value: {STARTING_CASH}')
    print(f'Final portfolio value: {portfolio_value}')
    print(f'Annualized return: {annualized_return:%}')
    print(f'Sharpe ratio: {sharpe_ratio}')
    print(f'Number of days simulated: {total_days}')





