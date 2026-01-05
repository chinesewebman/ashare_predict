"""
Visualization tools for landmark detection validation.

Uses mplfinance to plot landmarks on candlestick charts.
"""
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Try to import mplfinance, provide fallback if not available
try:
    import mplfinance as mpf
    MPLFINANCE_AVAILABLE = True
except ImportError:
    MPLFINANCE_AVAILABLE = False
    logger.warning("mplfinance not available, using basic matplotlib")


def plot_landmarks(symbol: str, df: pd.DataFrame, landmarks: List[Dict],
                   save_path: Optional[str] = None) -> None:
    """
    Plot landmarks on price chart.

    Args:
        symbol: Stock symbol
        df: DataFrame with OHLCV data
        landmarks: List of landmark dicts with 'date', 'price', 'type' keys
        save_path: Optional path to save the plot
    """
    if df.empty:
        logger.warning(f"Empty data for {symbol}")
        return

    if not landmarks:
        logger.warning(f"No landmarks to plot for {symbol}")
        return

    # Prepare data with English column names
    plot_df = df.copy()
    column_map = {
        '开盘': 'open',
        '收盘': 'close',
        '最高': 'high',
        '最低': 'low',
        '成交量': 'volume'
    }
    plot_df = plot_df.rename(columns=column_map)

    # Ensure we have the required columns
    required_cols = ['open', 'close', 'high', 'low']
    if not all(col in plot_df.columns for col in required_cols):
        logger.error(f"Missing required columns for plotting {symbol}")
        return

    # Extract landmark points for plotting
    low_dates = []
    low_prices = []
    high_dates = []
    high_prices = []

    for lm in landmarks:
        if lm.get('type') == 'low':
            low_dates.append(lm['date'])
            low_prices.append(lm['price'])
        elif lm.get('type') == 'high':
            high_dates.append(lm['date'])
            high_prices.append(lm['price'])

    if MPLFINANCE_AVAILABLE:
        _plot_with_mplfinance(symbol, plot_df, low_dates, low_prices,
                             high_dates, high_prices, save_path)
    else:
        _plot_with_matplotlib(symbol, plot_df, low_dates, low_prices,
                            high_dates, high_prices, save_path)


def _plot_with_mplfinance(symbol: str, df: pd.DataFrame,
                         low_dates: List, low_prices: List,
                         high_dates: List, high_prices: List,
                         save_path: Optional[str] = None) -> None:
    """Plot using mplfinance library."""
    # Create addplots for landmarks
    addplots = []

    if low_dates or low_prices:
        low_scatter = pd.Series(low_prices, index=low_dates)
        addplots.append(mpf.make_addplot(
            low_scatter,
            type='scatter',
            markersize=200,
            color='g',
            marker='^',
            panel=0
        ))

    if high_dates or high_prices:
        high_scatter = pd.Series(high_prices, index=high_dates)
        addplots.append(mpf.make_addplot(
            high_scatter,
            type='scatter',
            markersize=200,
            color='r',
            marker='v',
            panel=0
        ))

    # Plot
    try:
        mpf.plot(df,
                type='candle',
                style='charles',
                title=f'{symbol} - Landmark Detection',
                ylabel='Price',
                volume=True,
                addplot=addplots if addplots else None,
                savefig=save_path,
                figsize=(16, 10))
        logger.info(f"Plotted landmarks for {symbol} using mplfinance")
    except Exception as e:
        logger.error(f"mplfinance plotting failed: {e}")
        _plot_with_matplotlib(symbol, df, low_dates, low_prices,
                            high_dates, high_prices, save_path)


def _plot_with_matplotlib(symbol: str, df: pd.DataFrame,
                         low_dates: List, low_prices: List,
                         high_dates: List, high_prices: List,
                         save_path: Optional[str] = None) -> None:
    """Plot using basic matplotlib (fallback)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                   gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'{symbol} - Landmark Detection', fontsize=14)

    # Plot price
    ax1.plot(df.index, df['close'], label='Close Price', linewidth=1)

    # Add landmark markers
    if low_dates and low_prices:
        ax1.scatter(low_dates, low_prices, color='green', s=200,
                   marker='^', label='Low Landmark', zorder=5)

    if high_dates and high_prices:
        ax1.scatter(high_dates, high_prices, color='red', s=200,
                   marker='v', label='High Landmark', zorder=5)

    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot volume
    if 'volume' in df.columns:
        colors = ['green' if df['close'].iloc[i] >= df['open'].iloc[i]
                 else 'red' for i in range(len(df))]
        ax2.bar(df.index, df['volume'], color=colors, width=0.8)
        ax2.set_ylabel('Volume')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100)
        logger.info(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_prediction(symbol: str, df: pd.DataFrame,
                   landmarks: List[Dict],
                   predicted_price: float,
                   prediction_type: str = 'low',
                   save_path: Optional[str] = None) -> None:
    """
    Plot landmarks with predicted next landmark.

    Args:
        symbol: Stock symbol
        df: DataFrame with OHLCV data
        landmarks: List of confirmed landmarks
        predicted_price: Predicted next landmark price
        prediction_type: 'low' or 'high'
        save_path: Optional path to save the plot
    """
    if df.empty:
        logger.warning(f"Empty data for {symbol}")
        return

    # Prepare data
    plot_df = df.copy()
    column_map = {
        '开盘': 'open',
        '收盘': 'close',
        '最高': 'high',
        '最低': 'low',
        '成交量': 'volume'
    }
    plot_df = plot_df.rename(columns=column_map)

    # Extract existing landmarks
    existing_dates = []
    existing_prices = []

    for lm in landmarks:
        if lm.get('type') == prediction_type:
            existing_dates.append(lm['date'])
            existing_prices.append(lm['price'])

    # Use matplotlib for simpler plotting with prediction
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.suptitle(f'{symbol} - Pattern Prediction (Next {prediction_type.capitalize()})',
                fontsize=14)

    # Plot price
    ax.plot(plot_df.index, plot_df['close'], label='Close Price',
           linewidth=1, color='gray', alpha=0.7)

    # Plot existing landmarks
    if existing_dates and existing_prices:
        color = 'green' if prediction_type == 'low' else 'red'
        marker = '^' if prediction_type == 'low' else 'v'
        ax.scatter(existing_dates, existing_prices, color=color, s=200,
                  marker=marker, label=f'Historical {prediction_type}s', zorder=5)

        # Add price labels
        for date, price in zip(existing_dates, existing_prices):
            ax.annotate(f'{price:.0f}', (date, price),
                       xytext=(5, 10 if prediction_type == 'low' else -15),
                       textcoords='offset points', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))

    # Plot predicted landmark
    last_date = plot_df.index[-1]
    pred_color = 'green' if prediction_type == 'low' else 'red'
    ax.scatter(last_date, predicted_price, color=pred_color, s=300,
              marker='*', label=f'Predicted Next {prediction_type}', zorder=10)
    ax.annotate(f'Predicted: {predicted_price:.0f}', (last_date, predicted_price),
               xytext=(10, 15), textcoords='offset points', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
               fontweight='bold')

    # Add horizontal line at predicted price
    ax.axhline(y=predicted_price, color=pred_color, linestyle='--',
              alpha=0.5, linewidth=1)

    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100)
        logger.info(f"Saved prediction plot to {save_path}")
    else:
        plt.show()

    plt.close()


def test_visualizer():
    """Test the visualizer with sample data."""
    print("=== Testing Visualizer ===\n")

    # Create sample data
    import numpy as np
    from datetime import datetime, timedelta

    # Generate sample price data
    dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
    np.random.seed(42)

    close_prices = []
    price = 100.0
    for _ in range(200):
        change = np.random.normal(0, 0.02)
        price = price * (1 + change)
        close_prices.append(price)

    df = pd.DataFrame({
        'date': dates,
        'open': [p * (1 + np.random.normal(0, 0.005)) for p in close_prices],
        'close': close_prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in close_prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in close_prices],
        'volume': np.random.randint(1000000, 10000000, 200)
    })
    df.set_index('date', inplace=True)

    # Sample landmarks
    landmarks = [
        {'date': dates[20], 'price': close_prices[20], 'type': 'low'},
        {'date': dates[60], 'price': close_prices[60], 'type': 'high'},
        {'date': dates[100], 'price': close_prices[100], 'type': 'low'},
        {'date': dates[150], 'price': close_prices[150], 'type': 'high'},
    ]

    # Plot
    plot_landmarks('TEST', df, landmarks, save_path='test_landmarks.png')
    print("Plotted landmarks to test_landmarks.png")

    # Plot prediction
    predicted_price = close_prices[100] * 0.95  # Predict slightly lower
    plot_prediction('TEST', df, landmarks, predicted_price,
                   prediction_type='low', save_path='test_prediction.png')
    print("Plotted prediction to test_prediction.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_visualizer()
