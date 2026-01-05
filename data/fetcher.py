"""
A-share stock data fetcher using Tencent API with dual timeframe support.

Supports daily (1d) and weekly (1w) K-line data fetching.
"""
import requests
import json
import pandas as pd
import logging
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AshareFetcher:
    """A-share stock data fetcher using Tencent API."""

    # Base API URL
    API_URL = 'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get'

    # Exchange prefixes
    EXCHANGE_SH = ('600', '601', '603', '605', '688', '689')
    EXCHANGE_SZ = ('000', '002', '300', '301')

    def __init__(self, timeout: int = 30):
        """
        Initialize the fetcher.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout

    def _get_exchange_prefix(self, symbol: str) -> str:
        """
        Determine exchange prefix from stock symbol.

        Args:
            symbol: Stock symbol (e.g., '000001', '600036')

        Returns:
            'sh' or 'sz'
        """
        if symbol.startswith(self.EXCHANGE_SH):
            return 'sh'
        elif symbol.startswith(self.EXCHANGE_SZ):
            return 'sz'
        else:
            # Default to sz for unknown symbols
            logger.warning(f"Unknown symbol prefix for {symbol}, defaulting to 'sz'")
            return 'sz'

    def _fetch_from_api(self, symbol: str, frequency: str = 'day',
                        count: int = 1000) -> pd.DataFrame:
        """
        Fetch data from Tencent API.

        Args:
            symbol: Stock symbol (e.g., '000001', '600036')
            frequency: 'day' for daily, 'week' for weekly
            count: Maximum number of records to fetch

        Returns:
            DataFrame with stock data
        """
        try:
            # Format symbol with exchange prefix
            exchange_prefix = self._get_exchange_prefix(symbol)
            formatted_symbol = f"{exchange_prefix}{symbol}"

            # Build API URL
            # Format: param={symbol},{frequency},,,{count},qfq
            # qfq = 前复权 (forward-adjusted price)
            url = f"{self.API_URL}?param={formatted_symbol},{frequency},,,{count},qfq"

            logger.debug(f"Fetching {frequency} data for {symbol}: {url}")

            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()

            data = json.loads(response.content)

            # Parse response
            if 'data' not in data or formatted_symbol not in data['data']:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()

            stock_data = data['data'][formatted_symbol]

            # Get the appropriate data field
            if frequency == 'week':
                data_key = 'qfqweek' if 'qfqweek' in stock_data else 'week'
            else:
                data_key = 'qfqday' if 'qfqday' in stock_data else 'day'

            if data_key not in stock_data:
                logger.warning(f"No {frequency} data found for {symbol}")
                return pd.DataFrame()

            raw_data = stock_data[data_key]

            # Filter records (keep first 6 elements: date, OHLCV)
            filtered_data = [r[:6] if len(r) >= 6 else r for r in raw_data]
            filtered_data = [r for r in filtered_data if len(r) >= 6]

            if not filtered_data:
                logger.warning(f"No valid data after filtering for {symbol}")
                return pd.DataFrame()

            # Create DataFrame
            df = pd.DataFrame(filtered_data,
                            columns=['date', 'open', 'close', 'high', 'low', 'volume'])

            # Convert data types
            df[['open', 'close', 'high', 'low', 'volume']] = df[
                ['open', 'close', 'high', 'low', 'volume']
            ].astype(float)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            # Rename to Chinese column names (compatible with existing code)
            df = df.rename(columns={
                'open': '开盘',
                'close': '收盘',
                'high': '最高',
                'low': '最低',
                'volume': '成交量'
            })
            df['symbol'] = symbol

            logger.info(f"Fetched {len(df)} {frequency} records for {symbol}")
            return df

        except requests.RequestException as e:
            logger.error(f"Request error for {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def get_daily_data(self, symbol: str, count: int = 1000) -> pd.DataFrame:
        """
        Fetch daily K-line data.

        Args:
            symbol: Stock symbol (e.g., '000001', '600036')
            count: Maximum number of records to fetch (default: 1000)

        Returns:
            DataFrame with daily OHLCV data
        """
        return self._fetch_from_api(symbol, frequency='day', count=count)

    def get_weekly_data(self, symbol: str, count: int = 500) -> pd.DataFrame:
        """
        Fetch weekly K-line data.

        Args:
            symbol: Stock symbol (e.g., '000001', '600036')
            count: Maximum number of records to fetch (default: 500 weeks ~10 years)

        Returns:
            DataFrame with weekly OHLCV data
        """
        return self._fetch_from_api(symbol, frequency='week', count=count)

    def get_data(self, symbol: str, frequency: str = 'day',
                 count: int = 1000) -> pd.DataFrame:
        """
        Fetch K-line data with specified frequency.

        Args:
            symbol: Stock symbol (e.g., '000001', '600036')
            frequency: 'day' or 'week'
            count: Maximum number of records to fetch

        Returns:
            DataFrame with OHLCV data
        """
        if frequency == 'week':
            return self.get_weekly_data(symbol, count=count)
        else:
            return self.get_daily_data(symbol, count=count)

    def get_current_price(self, symbol: str) -> float:
        """
        Get the most recent closing price.

        Args:
            symbol: Stock symbol

        Returns:
            Current closing price or 0 if unavailable
        """
        df = self.get_daily_data(symbol, count=1)
        if not df.empty:
            return float(df.iloc[-1]['收盘'])
        return 0.0


def test_fetcher():
    """Test the AshareFetcher implementation."""
    print("=== Testing AshareFetcher ===\n")

    fetcher = AshareFetcher()

    test_symbols = ['000001', '600036', '000858']

    for symbol in test_symbols:
        print(f"\n--- Testing {symbol} ---")

        # Test daily data
        daily_df = fetcher.get_daily_data(symbol, count=5)
        if not daily_df.empty:
            print(f"✅ Daily data: {len(daily_df)} records")
            print(f"Latest close: {daily_df.iloc[-1]['收盘']:.2f}")
        else:
            print(f"❌ No daily data")

        # Test weekly data
        weekly_df = fetcher.get_weekly_data(symbol, count=5)
        if not weekly_df.empty:
            print(f"✅ Weekly data: {len(weekly_df)} records")
            print(f"Latest weekly close: {weekly_df.iloc[-1]['收盘']:.2f}")
        else:
            print(f"❌ No weekly data")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_fetcher()
