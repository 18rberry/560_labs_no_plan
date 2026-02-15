"""
DSCI-560 Lab 4 - LSTM Trading Signal Generator for Stock Portfolio
==================================================================================
This script trains a separate LSTM model for each stock in our portfolio and
generates buy/sell/hold signals. Each stock gets its own model because different
stocks have different price patterns (e.g., TSLA is way more volatile than IBM).

Output: portfolio_signals.csv with columns [Date, Ticker, Signal, Confidence]
        that our team's mock trading environment can read for portfolio decisions.

Authors: Lab 4 Team
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend so plots save without display issues
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

# Config - easy to tweak for experiments
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'IBM', 'ORCL', 'AMZN']
SEQUENCE_LENGTH = 20          # use 20 days of history to predict next day
THRESHOLD = 0.02              # 2% threshold for buy/sell classification
BATCH_SIZE = 32
HIDDEN_SIZE = 64              # LSTM hidden units
NUM_LAYERS = 2                # stacked LSTM layers
DROPOUT = 0.3
LEARNING_RATE = 0.001
MAX_EPOCHS = 150
PATIENCE = 15                 # early stopping patience
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
# test ratio is the remaining 0.2

# features we'll feed into the LSTM (subset of columns from the cleaned data)
FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
    'daily_return', 'Close_lag1', 'volume_ratio',
    'price_range', 'volatility_20', 'day_of_week'
]

# create output directories
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# use GPU if available (probably not on most laptops but worth checking)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Step 1: Data Loading and Preprocessing

def load_and_prepare_data(filepath):
    """
    Load the cleaned CSV and do some final prep before we can
    create sequences for the LSTM.
    """
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    print(f"Loaded {len(df)} rows, {df['Ticker'].nunique()} tickers")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

    return df


def create_target_labels(df_ticker):
    """
    Create the target variable: next-day price movement classification.

    Logic:
      - Calculate tomorrow's return = (Close_tomorrow - Close_today) / Close_today
      - If return > 2%  -> Buy signal (1)
      - If return < -2% -> Sell signal (-1)
      - Otherwise       -> Hold signal (0)

    We shift the close price by -1 to get "tomorrow's close" for each row.
    The last row won't have a target (NaN) since there's no "tomorrow" yet.
    """
    future_return = df_ticker['Close'].pct_change().shift(-1)

    labels = pd.Series(0, index=df_ticker.index, dtype=int)  # default to hold
    labels[future_return > THRESHOLD] = 1     # buy
    labels[future_return < -THRESHOLD] = -1   # sell

    return labels, future_return


def prepare_stock_data(df, ticker):
    """
    For a single stock:
    1. Filter rows for this ticker
    2. Drop rows with NaN in our features (from rolling windows at the start)
    3. Create target labels
    4. Scale features
    5. Split into train/val/test chronologically (no shuffling! this is time series)
    """
    df_ticker = df[df['Ticker'] == ticker].copy()
    df_ticker = df_ticker.sort_values('Date').reset_index(drop=True)

    # create labels before dropping anything
    df_ticker['target'], df_ticker['future_return'] = create_target_labels(df_ticker)

    # drop rows where features or target are NaN
    # (first ~50 rows will have NaN from SMA_50, last row has no target)
    cols_needed = FEATURE_COLS + ['target']
    df_ticker = df_ticker.dropna(subset=cols_needed).reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"Processing {ticker}: {len(df_ticker)} usable rows")

    # check class distribution -- important to know if classes are balanced
    label_counts = df_ticker['target'].value_counts().sort_index()
    print(f"  Signal distribution: Sell(-1)={label_counts.get(-1, 0)}, "
          f"Hold(0)={label_counts.get(0, 0)}, Buy(1)={label_counts.get(1, 0)}")

    # scale features using StandardScaler (fit only on training data to avoid leakage!)
    n = len(df_ticker)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    scaler = StandardScaler()
    # fit scaler on training data only
    scaler.fit(df_ticker.iloc[:train_end][FEATURE_COLS])

    # transform all data using the training scaler
    features_scaled = scaler.transform(df_ticker[FEATURE_COLS])

    # remap labels: -1 -> 0, 0 -> 1, 1 -> 2 (PyTorch CrossEntropyLoss needs 0-indexed classes)
    label_mapping = {-1: 0, 0: 1, 1: 2}
    labels = df_ticker['target'].map(label_mapping).values

    dates = df_ticker['Date'].values

    print(f"  Train: rows 0-{train_end-1} | Val: rows {train_end}-{val_end-1} | "
          f"Test: rows {val_end}-{n-1}")

    return features_scaled, labels, dates, scaler, train_end, val_end


# Step 2: PyTorch Dataset for Sequences

class StockSequenceDataset(Dataset):
    """
    Custom PyTorch Dataset that creates sliding window sequences.

    For each index i, we grab features[i : i+seq_len] as input
    and labels[i+seq_len] as the target.

    So if seq_len=20, we use 20 days of features to predict
    what happens on day 21.
    """
    def __init__(self, features, labels, seq_len):
        self.features = features
        self.labels = labels
        self.seq_len = seq_len

    def __len__(self):
        # we can create sequences starting from index 0 up to len-seq_len
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        # grab a window of features and the label at the end of that window
        x = self.features[idx : idx + self.seq_len]
        y = self.labels[idx + self.seq_len]
        return torch.FloatTensor(x), torch.LongTensor([y]).squeeze()


def create_data_loaders(features, labels, train_end, val_end, seq_len):
    """
    Split data chronologically and create DataLoaders.
    IMPORTANT: we do NOT shuffle because this is time series data!
    Shuffling would let the model "peek" into the future.
    """
    # figure out the valid index ranges for each split, accounting for sequence length
    # training sequences: indices [0, train_end - seq_len)
    # validation sequences: indices [train_end - seq_len, val_end - seq_len)
    # test sequences: indices [val_end - seq_len, end - seq_len)

    train_features = features[:train_end]
    train_labels = labels[:train_end]

    val_features = features[train_end - seq_len : val_end]
    val_labels = labels[train_end - seq_len : val_end]

    test_features = features[val_end - seq_len :]
    test_labels = labels[val_end - seq_len :]

    train_dataset = StockSequenceDataset(train_features, train_labels, seq_len)
    val_dataset = StockSequenceDataset(val_features, val_labels, seq_len)
    test_dataset = StockSequenceDataset(test_features, test_labels, seq_len)

    print(f"  Sequences -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | "
          f"Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


# Step 3: LSTM Model Definition

class TradingLSTM(nn.Module):
    """
    LSTM-based classifier for stock trading signals.

    Architecture:
      Input -> LSTM (2 layers, 64 hidden) -> Dropout -> FC -> 3 classes

    The LSTM processes the sequence of daily features and we take the
    output from the LAST time step to make our prediction.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(TradingLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer - batch_first=True means input shape is (batch, seq, features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0  # dropout between LSTM layers
        )

        # dropout before the fully connected layer
        self.dropout = nn.Dropout(dropout)

        # fully connected output layer: hidden_size -> 3 classes (sell/hold/buy)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_features)

        # initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # run through LSTM
        # out shape: (batch_size, sequence_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        # take only the last time step's output
        out = out[:, -1, :]  # shape: (batch_size, hidden_size)

        # dropout + fully connected layer
        out = self.dropout(out)
        out = self.fc(out)  # shape: (batch_size, num_classes)

        return out


# Step 4: Training Loop with Early Stopping

def train_model(model, train_loader, val_loader, ticker):
    """
    Train the LSTM model with:
    - CrossEntropyLoss (handles class imbalance somewhat with the weighting)
    - Adam optimizer
    - ReduceLROnPlateau scheduler (reduces LR when validation loss plateaus)
    - Early stopping (stop if val loss doesn't improve for PATIENCE epochs)
    """
    # compute class weights to handle imbalanced classes
    # (there are usually way more "hold" signals than buy/sell)
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    all_labels = np.array(all_labels)

    class_counts = np.bincount(all_labels, minlength=3)
    # avoid division by zero if a class has no samples
    class_counts = np.maximum(class_counts, 1)
    total = len(all_labels)
    class_weights = torch.FloatTensor([total / (3 * c) for c in class_counts]).to(device)
    print(f"  Class weights: {class_weights.cpu().numpy().round(3)}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # reduce learning rate when validation loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7
    )

    # tracking variables for early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print(f"  Training {ticker} model...")

    for epoch in range(MAX_EPOCHS):
        # ---- Training phase ----
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # gradient clipping to prevent exploding gradients (common with LSTMs)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        train_loss = epoch_loss / len(train_loader)
        train_acc = correct / total

        # ---- Validation phase ----
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        # update LR scheduler based on validation loss
        scheduler.step(val_loss)

        # record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1:3d}/{MAX_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | "
                  f"LR: {current_lr:.6f}")

        # ---- Early Stopping Check ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"    Early stopping at epoch {epoch+1} (best val loss: {best_val_loss:.4f})")
            break

    # restore best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    }

    return model, history



# Step 5: Evaluation


def evaluate_model(model, test_loader, ticker):
    """
    Run the model on the test set and compute classification metrics.
    Returns predictions, true labels, and confidence scores.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)

            # softmax to get probabilities (confidence scores)
            probs = torch.softmax(outputs, dim=1)

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # map back from 0,1,2 to -1,0,1 for the classification report
    reverse_map = {0: -1, 1: 0, 2: 1}
    preds_original = np.array([reverse_map[p] for p in all_preds])
    labels_original = np.array([reverse_map[l] for l in all_labels])

    # compute metrics
    acc = accuracy_score(all_labels, all_preds)
    # use zero_division=0 to avoid warnings when a class has no predictions
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f"\n  {ticker} Test Results:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1 Score:  {f1:.4f}")

    print(f"\n  Classification Report:")
    target_names = ['Sell (-1)', 'Hold (0)', 'Buy (1)']
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))

    # confidence = max probability across the 3 classes for each prediction
    confidences = np.max(all_probs, axis=1)

    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }

    return preds_original, labels_original, confidences, metrics


# Step 6: Generate Signals for the Full Dataset

def generate_signals(model, features_scaled, dates, seq_len, val_end):
    """
    Generate trading signals for the TEST portion of the data.
    We only generate signals for the test set (dates after val_end)
    since those are truly "unseen" by the model.

    Returns a DataFrame with Date, Signal, Confidence for this stock.
    """
    model.eval()
    signals = []

    # we generate signals starting from val_end (the test set)
    # for each position i in test set, we need features[i-seq_len : i] as input
    with torch.no_grad():
        for i in range(val_end, len(features_scaled)):
            if i < seq_len:
                continue  # not enough history yet

            # get the sequence
            seq = features_scaled[i - seq_len : i]
            x = torch.FloatTensor(seq).unsqueeze(0).to(device)  # add batch dimension

            output = model(x)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]

            pred_class = np.argmax(probs)
            confidence = float(np.max(probs))

            # map back: 0->-1(sell), 1->0(hold), 2->1(buy)
            reverse_map = {0: -1, 1: 0, 2: 1}
            signal = reverse_map[pred_class]

            signals.append({
                'Date': pd.Timestamp(dates[i]),
                'Signal': signal,
                'Confidence': round(confidence, 4)
            })

    return pd.DataFrame(signals)


# Step 7: Visualization Functions

def plot_training_history(history, ticker):
    """Plot training and validation loss/accuracy curves for one stock."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # loss
    ax1.plot(history['train_loss'], label='Train Loss', color='blue', alpha=0.8)
    ax1.plot(history['val_loss'], label='Val Loss', color='orange', alpha=0.8)
    ax1.set_title(f'{ticker} - Training & Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # accuracy
    ax2.plot(history['train_acc'], label='Train Acc', color='blue', alpha=0.8)
    ax2.plot(history['val_acc'], label='Val Acc', color='orange', alpha=0.8)
    ax2.set_title(f'{ticker} - Training & Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'plots/{ticker}_training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved training plot: plots/{ticker}_training_history.png")


def plot_performance_comparison(all_metrics):
    """
    Bar chart comparing accuracy, precision, recall, F1 across all stocks.
    This is the main comparison chart for the report.
    """
    tickers = list(all_metrics.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1']

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(tickers))
    width = 0.2

    for i, metric in enumerate(metrics_names):
        values = [all_metrics[t][metric] for t in tickers]
        bars = ax.bar(x + i * width, values, width, label=metric.capitalize(), alpha=0.85)
        # add value labels on top of each bar
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Stock Ticker')
    ax.set_ylabel('Score')
    ax.set_title('LSTM Model Performance Comparison Across Portfolio Stocks')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(tickers)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('plots/performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plots/performance_comparison.png")


def plot_signal_distribution(all_signals_df):
    """
    Show the distribution of buy/sell/hold signals for each stock.
    Helps us see if a model is biased toward one signal type.
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    signal_labels = {-1: 'Sell', 0: 'Hold', 1: 'Buy'}
    colors = {-1: '#e74c3c', 0: '#95a5a6', 1: '#2ecc71'}

    for i, ticker in enumerate(TICKERS):
        ax = axes[i]
        ticker_data = all_signals_df[all_signals_df['Ticker'] == ticker]

        counts = ticker_data['Signal'].value_counts().sort_index()
        signal_names = [signal_labels.get(s, str(s)) for s in counts.index]
        bar_colors = [colors.get(s, '#333') for s in counts.index]

        ax.bar(signal_names, counts.values, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.set_title(f'{ticker}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count')

        # add count labels
        for j, (name, count) in enumerate(zip(signal_names, counts.values)):
            ax.text(j, count + 0.5, str(count), ha='center', fontsize=9)

    # hide the 8th subplot (we only have 7 stocks)
    axes[7].set_visible(False)

    fig.suptitle('Trading Signal Distribution per Stock', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/signal_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plots/signal_distribution.png")


def plot_confusion_matrices(all_preds, all_labels):
    """
    Plot confusion matrices for each stock side by side.
    Useful for seeing where the model gets confused (e.g., sell vs hold).
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    class_names = ['Sell', 'Hold', 'Buy']

    for i, ticker in enumerate(TICKERS):
        if ticker not in all_preds:
            continue
        ax = axes[i]
        cm = confusion_matrix(all_labels[ticker], all_preds[ticker], labels=[-1, 0, 1])

        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(f'{ticker}', fontsize=11, fontweight='bold')
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(class_names, fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

        # add text annotations to each cell
        for row in range(3):
            for col in range(3):
                ax.text(col, row, str(cm[row, col]),
                        ha='center', va='center', fontsize=10,
                        color='white' if cm[row, col] > cm.max() / 2 else 'black')

    axes[7].set_visible(False)

    fig.suptitle('Confusion Matrices per Stock', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plots/confusion_matrices.png")


# Main Pipeline

def main():
    print("=" * 70)
    print("DSCI-560 Lab 4 - LSTM Portfolio Trading Signal Generator")
    print("=" * 70)

    # load data
    df = load_and_prepare_data('cleaned_yfinance_data.csv')

    # containers for results across all stocks
    all_metrics = {}
    all_histories = {}
    all_signals = []
    all_test_preds = {}
    all_test_labels = {}

    # ---- Train a separate model for each stock ----
    for ticker in TICKERS:
        # prepare data for this stock
        features_scaled, labels, dates, scaler, train_end, val_end = \
            prepare_stock_data(df, ticker)

        # create DataLoaders with sliding window sequences
        train_loader, val_loader, test_loader = \
            create_data_loaders(features_scaled, labels, train_end, val_end, SEQUENCE_LENGTH)

        # initialize model
        input_size = len(FEATURE_COLS)
        model = TradingLSTM(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_classes=3,
            dropout=DROPOUT
        ).to(device)

        print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

        # train with early stopping and LR scheduling
        model, history = train_model(model, train_loader, val_loader, ticker)
        all_histories[ticker] = history

        # evaluate on test set
        preds, labels_true, confidences, metrics = evaluate_model(model, test_loader, ticker)
        all_metrics[ticker] = metrics
        all_test_preds[ticker] = preds
        all_test_labels[ticker] = labels_true

        # generate signals for the test portion
        signals_df = generate_signals(model, features_scaled, dates, SEQUENCE_LENGTH, val_end)
        signals_df['Ticker'] = ticker
        all_signals.append(signals_df)

        # save model checkpoint
        checkpoint_path = f'models/{ticker}_lstm_model.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'ticker': ticker,
            'metrics': metrics,
            'feature_cols': FEATURE_COLS,
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            'sequence_length': SEQUENCE_LENGTH,
        }, checkpoint_path)
        print(f"  Model saved: {checkpoint_path}")

        # plot training history for this stock
        plot_training_history(history, ticker)

    # ---- Combine all signals into one portfolio CSV ----
    print("\n" + "=" * 70)
    print("Generating Portfolio Signals CSV")
    print("=" * 70)

    portfolio_signals = pd.concat(all_signals, ignore_index=True)
    portfolio_signals = portfolio_signals[['Date', 'Ticker', 'Signal', 'Confidence']]
    portfolio_signals = portfolio_signals.sort_values(['Date', 'Ticker']).reset_index(drop=True)

    # save to CSV
    portfolio_signals.to_csv('portfolio_signals.csv', index=False)
    print(f"Saved portfolio_signals.csv with {len(portfolio_signals)} rows")
    print(f"Date range: {portfolio_signals['Date'].min()} to {portfolio_signals['Date'].max()}")
    print(f"Tickers: {sorted(portfolio_signals['Ticker'].unique())}")

    # quick look at signal counts
    print("\nOverall signal distribution:")
    signal_map = {-1: 'Sell', 0: 'Hold', 1: 'Buy'}
    for sig_val, sig_name in signal_map.items():
        count = (portfolio_signals['Signal'] == sig_val).sum()
        pct = count / len(portfolio_signals) * 100
        print(f"  {sig_name}: {count} ({pct:.1f}%)")

    # ---- Generate comparison visualizations ----
    print("\n" + "=" * 70)
    print("Generating Visualizations")
    print("=" * 70)

    plot_performance_comparison(all_metrics)
    plot_signal_distribution(portfolio_signals)
    plot_confusion_matrices(all_test_preds, all_test_labels)

    # ---- Print summary table for the report ----
    print("\n" + "=" * 70)
    print("SUMMARY: Model Performance Across Portfolio")
    print("=" * 70)
    print(f"{'Ticker':<8} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1 Score':>10}")
    print("-" * 50)

    best_ticker = None
    best_f1 = 0

    for ticker in TICKERS:
        m = all_metrics[ticker]
        print(f"{ticker:<8} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>10.4f} {m['f1']:>10.4f}")
        if m['f1'] > best_f1:
            best_f1 = m['f1']
            best_ticker = ticker

    print("-" * 50)

    # compute averages
    avg_acc = np.mean([all_metrics[t]['accuracy'] for t in TICKERS])
    avg_prec = np.mean([all_metrics[t]['precision'] for t in TICKERS])
    avg_rec = np.mean([all_metrics[t]['recall'] for t in TICKERS])
    avg_f1 = np.mean([all_metrics[t]['f1'] for t in TICKERS])
    print(f"{'Average':<8} {avg_acc:>10.4f} {avg_prec:>10.4f} "
          f"{avg_rec:>10.4f} {avg_f1:>10.4f}")

    print(f"\nBest performing model: {best_ticker} (F1: {best_f1:.4f})")

    # save metrics to JSON for easy access later
    with open('models/portfolio_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print("Saved metrics to models/portfolio_metrics.json")

    print("\n" + "=" * 70)
    print("Done! Files generated:")
    print("  - portfolio_signals.csv     (trading signals for mock environment)")
    print("  - models/*_lstm_model.pth   (model checkpoints per stock)")
    print("  - models/portfolio_metrics.json")
    print("  - plots/*_training_history.png")
    print("  - plots/performance_comparison.png")
    print("  - plots/signal_distribution.png")
    print("  - plots/confusion_matrices.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
