"""
SQLite cache manager for A-share stock data.

Provides caching functionality to reduce external API calls.
Reuses database schema from stock-operator-automation.
"""
import sqlite3
import pandas as pd
import logging
from typing import Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class CacheManager:
    """SQLite cache manager for stock data."""

    # Default database path
    DEFAULT_DB_PATH = "stock_cache.db"

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the cache manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or self.DEFAULT_DB_PATH
        self._init_database()

    def _init_database(self):
        """Initialize database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            # Daily data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stock_daily_data (
                    symbol TEXT,
                    date DATE,
                    open REAL,
                    close REAL,
                    high REAL,
                    low REAL,
                    volume REAL,
                    PRIMARY KEY (symbol, date)
                )
            """)

            # Weekly data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stock_weekly_data (
                    symbol TEXT,
                    date DATE,
                    open REAL,
                    close REAL,
                    high REAL,
                    low REAL,
                    volume REAL,
                    PRIMARY KEY (symbol, date)
                )
            """)

            # Cache metadata table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_symbol "
                        "ON stock_daily_data(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_date "
                        "ON stock_daily_data(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_weekly_symbol "
                        "ON stock_weekly_data(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_weekly_date "
                        "ON stock_weekly_data(date)")

            logger.info(f"Database initialized: {self.db_path}")

    def _get_table_name(self, frequency: str) -> str:
        """Get table name for given frequency."""
        return 'stock_weekly_data' if frequency == 'week' else 'stock_daily_data'

    def save_data(self, symbol: str, data: pd.DataFrame,
                  frequency: str = 'day'):
        """
        Save stock data to cache.

        Args:
            symbol: Stock symbol
            data: DataFrame with OHLCV data
            frequency: 'day' or 'week'
        """
        try:
            if data.empty:
                return

            table_name = self._get_table_name(frequency)

            with sqlite3.connect(self.db_path) as conn:
                # Prepare records
                records = []
                for date, row in data.iterrows():
                    date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                    records.append((
                        symbol,
                        date_str,
                        row.get('开盘', row.get('open', 0)),
                        row.get('收盘', row.get('close', 0)),
                        row.get('最高', row.get('high', 0)),
                        row.get('最低', row.get('low', 0)),
                        row.get('成交量', row.get('volume', 0))
                    ))

                # Insert or replace
                conn.executemany(f"""
                    INSERT OR REPLACE INTO {table_name}
                    (symbol, date, open, close, high, low, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, records)

                # Update metadata
                conn.execute("""
                    INSERT OR REPLACE INTO cache_metadata (key, value)
                    VALUES (?, ?)
                """, (f"{frequency}_data_last_updated_{symbol}",
                     datetime.now().isoformat()))

                logger.info(f"Saved {len(records)} {frequency} records for {symbol}")

        except Exception as e:
            logger.error(f"Error saving data for {symbol}: {e}")

    def get_data(self, symbol: str, frequency: str = 'day',
                start_date: Optional[str] = None,
                end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get stock data from cache.

        Args:
            symbol: Stock symbol
            frequency: 'day' or 'week'
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)

        Returns:
            DataFrame with cached data
        """
        try:
            table_name = self._get_table_name(frequency)

            with sqlite3.connect(self.db_path) as conn:
                query = f"SELECT * FROM {table_name} WHERE symbol = ?"
                params = [symbol]

                if start_date:
                    query += " AND date >= ?"
                    params.append(start_date)
                if end_date:
                    query += " AND date <= ?"
                    params.append(end_date)

                query += " ORDER BY date"

                df = pd.read_sql(query, conn, params=params)

                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    # Rename to Chinese format
                    df = df.rename(columns={
                        'open': '开盘',
                        'close': '收盘',
                        'high': '最高',
                        'low': '最低',
                        'volume': '成交量'
                    })
                    df['symbol'] = symbol

                return df

        except Exception as e:
            logger.error(f"Error retrieving data for {symbol}: {e}")
            return pd.DataFrame()

    def has_data(self, symbol: str, frequency: str = 'day') -> bool:
        """
        Check if cached data exists for symbol.

        Args:
            symbol: Stock symbol
            frequency: 'day' or 'week'

        Returns:
            True if data exists
        """
        try:
            table_name = self._get_table_name(frequency)
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute(
                    f"SELECT COUNT(*) FROM {table_name} WHERE symbol = ?",
                    (symbol,)
                ).fetchone()
                return result[0] > 0
        except Exception as e:
            logger.error(f"Error checking data for {symbol}: {e}")
            return False

    def get_date_range(self, symbol: str, frequency: str = 'day'):
        """
        Get the date range of cached data.

        Args:
            symbol: Stock symbol
            frequency: 'day' or 'week'

        Returns:
            Dict with 'min_date' and 'max_date'
        """
        try:
            table_name = self._get_table_name(frequency)
            with sqlite3.connect(self.db_path) as conn:
                min_date = conn.execute(
                    f"SELECT MIN(date) FROM {table_name} WHERE symbol = ?",
                    (symbol,)
                ).fetchone()[0]

                max_date = conn.execute(
                    f"SELECT MAX(date) FROM {table_name} WHERE symbol = ?",
                    (symbol,)
                ).fetchone()[0]

                return {
                    'min_date': min_date,
                    'max_date': max_date,
                    'count': conn.execute(
                        f"SELECT COUNT(*) FROM {table_name} WHERE symbol = ?",
                        (symbol,)
                    ).fetchone()[0]
                }
        except Exception as e:
            logger.error(f"Error getting date range for {symbol}: {e}")
            return {'min_date': None, 'max_date': None, 'count': 0}

    def clear_data(self, symbol: Optional[str] = None):
        """
        Clear cached data.

        Args:
            symbol: Stock symbol (clears all if None)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                if symbol:
                    conn.execute("DELETE FROM stock_daily_data WHERE symbol = ?",
                               (symbol,))
                    conn.execute("DELETE FROM stock_weekly_data WHERE symbol = ?",
                               (symbol,))
                    logger.info(f"Cleared data for {symbol}")
                else:
                    conn.execute("DELETE FROM stock_daily_data")
                    conn.execute("DELETE FROM stock_weekly_data")
                    logger.info("Cleared all data")
        except Exception as e:
            logger.error(f"Error clearing data: {e}")

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                daily_count = conn.execute(
                    "SELECT COUNT(*) FROM stock_daily_data"
                ).fetchone()[0]

                weekly_count = conn.execute(
                    "SELECT COUNT(*) FROM stock_weekly_data"
                ).fetchone()[0]

                unique_daily = conn.execute(
                    "SELECT COUNT(DISTINCT symbol) FROM stock_daily_data"
                ).fetchone()[0]

                unique_weekly = conn.execute(
                    "SELECT COUNT(DISTINCT symbol) FROM stock_weekly_data"
                ).fetchone()[0]

                return {
                    'daily_records': daily_count,
                    'weekly_records': weekly_count,
                    'unique_symbols_daily': unique_daily,
                    'unique_symbols_weekly': unique_weekly,
                    'db_path': self.db_path
                }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}

    def optimize(self):
        """Optimize database with VACUUM."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("VACUUM")
                logger.info("Database optimized")
        except Exception as e:
            logger.error(f"Error optimizing database: {e}")


def test_cache_manager():
    """Test the CacheManager implementation."""
    print("=== Testing CacheManager ===\n")

    # Use test database
    cache = CacheManager("test_cache.db")

    # Show stats
    stats = cache.get_cache_stats()
    print(f"Cache stats: {stats}")

    # Check date range for a symbol
    if cache.has_data('000001', 'day'):
        date_range = cache.get_date_range('000001', 'day')
        print(f"Date range for 000001: {date_range}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_cache_manager()
