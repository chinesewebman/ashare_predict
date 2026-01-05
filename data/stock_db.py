#!/usr/bin/env python3
"""
Stock Database with Pinyin Search Support.

Maps stock codes to names and provides pinyin abbreviation search.
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Try to import pypinyin for Chinese to pinyin conversion
try:
    from pypinyin import lazy_pinyin, Style
    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False
    logger.warning("pypinyin not installed, pinyin search will be limited")

# Try to import akshare for fetching full stock list
try:
    import akshare as ak
    HAS_AKSHARE = True
except ImportError:
    HAS_AKSHARE = False
    logger.warning("akshare not installed, full stock list fetching unavailable")


class StockDatabase:
    """Stock database with name and pinyin search support."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize stock database.

        Args:
            db_path: Path to JSON file with stock data
        """
        self.db_path = db_path or Path(__file__).parent.parent / "data" / "stocks.json"
        self.stocks: Dict[str, Dict] = {}
        self.pinyin_index: Dict[str, List[str]] = {}
        self.load()

    def load(self):
        """Load stock database from file."""
        if Path(self.db_path).exists():
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.stocks = data.get('stocks', {})
                logger.info(f"Loaded {len(self.stocks)} stocks from {self.db_path}")
            except Exception as e:
                logger.error(f"Error loading stock database: {e}")
                self.stocks = {}
        else:
            # Create default database with some common stocks
            self.stocks = self._get_default_stocks()
            self.save()
            logger.info(f"Created default stock database with {len(self.stocks)} stocks")

        # Build pinyin index
        self._build_pinyin_index()

    def save(self):
        """Save stock database to file."""
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump({'stocks': self.stocks}, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self.stocks)} stocks to {self.db_path}")
        except Exception as e:
            logger.error(f"Error saving stock database: {e}")

    def _get_default_stocks(self) -> Dict[str, Dict]:
        """Get default stock database."""
        return {
            "601933": {"name": "京东", "pinyin": "jd", "full_pinyin": "jingdong"},
            "000001": {"name": "平安银行", "pinyin": "payh", "full_pinyin": "pinganyinhang"},
            "600036": {"name": "招商银行", "pinyin": "zsyh", "full_pinyin": "zhaoshangyinhang"},
            "000858": {"name": "五粮液", "pinyin": "wly", "full_pinyin": "wuliangye"},
            "600519": {"name": "贵州茅台", "pinyin": "gzmt", "full_pinyin": "guizhoumaotai"},
            "601077": {"name": "渝农商行", "pinyin": "ynsh", "full_pinyin": "yunongshanghang"},
            "000002": {"name": "万科A", "pinyin": "wka", "full_pinyin": "wankea"},
            "600276": {"name": "恒瑞医药", "pinyin": "hryy", "full_pinyin": "hengruiyiyao"},
            "002415": {"name": "海康威视", "pinyin": "hkws", "full_pinyin": "haikangweishi"},
            "600009": {"name": "上海机场", "pinyin": "shjc", "full_pinyin": "shanghajichang"},
        }

    def _build_pinyin_index(self):
        """Build pinyin abbreviation index for fast search."""
        self.pinyin_index = {}

        for code, info in self.stocks.items():
            name = info.get('name', '')

            # Get pre-defined pinyin or generate it
            if 'pinyin' in info:
                abbr = info['pinyin']
            elif HAS_PYPINYIN:
                # Generate from Chinese name
                abbr = self._generate_pinyin_abbr(name)
            else:
                continue

            # Add to index
            if abbr not in self.pinyin_index:
                self.pinyin_index[abbr] = []
            if code not in self.pinyin_index[abbr]:
                self.pinyin_index[abbr].append(code)

            # Also index by full pinyin
            if 'full_pinyin' in info:
                full = info['full_pinyin']
                if full not in self.pinyin_index:
                    self.pinyin_index[full] = []
                if code not in self.pinyin_index[full]:
                    self.pinyin_index[full].append(code)

    def _generate_pinyin_abbr(self, chinese_name: str) -> str:
        """Generate pinyin abbreviation from Chinese name."""
        if not HAS_PYPINYIN:
            return ''

        try:
            # Get first letter of each character's pinyin
            py = lazy_pinyin(chinese_name, style=Style.FIRST_LETTER)
            return ''.join(py).lower()
        except Exception as e:
            logger.warning(f"Error generating pinyin for {chinese_name}: {e}")
            return ''

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search stocks by code, name, or pinyin abbreviation.

        Args:
            query: Search query (code, name, or pinyin)
            limit: Maximum results to return

        Returns:
            List of matching stocks with code, name, and match_type
        """
        query = query.strip().lower()
        results = []

        # Exact code match
        if query.isdigit() and len(query) == 6:
            code = query.zfill(6)
            if code in self.stocks:
                info = self.stocks[code]
                results.append({
                    'code': code,
                    'name': info['name'],
                    'match_type': 'code',
                    'pinyin': info.get('pinyin', ''),
                    'exchange': 'SH' if code.startswith('6') else 'SZ'
                })
                return results

        # Pinyin abbreviation match
        if query in self.pinyin_index:
            for code in self.pinyin_index[query][:limit]:
                info = self.stocks[code]
                results.append({
                    'code': code,
                    'name': info['name'],
                    'match_type': 'pinyin',
                    'pinyin': info.get('pinyin', ''),
                    'exchange': 'SH' if code.startswith('6') else 'SZ'
                })
            return results

        # Partial pinyin match
        for abbr, codes in self.pinyin_index.items():
            if query in abbr:
                for code in codes[:limit]:
                    info = self.stocks[code]
                    # Avoid duplicates
                    if not any(r['code'] == code for r in results):
                        results.append({
                            'code': code,
                            'name': info['name'],
                            'match_type': 'pinyin_partial',
                            'pinyin': info.get('pinyin', ''),
                            'exchange': 'SH' if code.startswith('6') else 'SZ'
                        })
                        if len(results) >= limit:
                            break
            if len(results) >= limit:
                break

        # Name partial match (for Chinese input)
        for code, info in self.stocks.items():
            if query in info['name']:
                results.append({
                    'code': code,
                    'name': info['name'],
                    'match_type': 'name',
                    'pinyin': info.get('pinyin', ''),
                    'exchange': 'SH' if code.startswith('6') else 'SZ'
                })
                if len(results) >= limit:
                    break

        return results

    def get_name(self, code: str) -> Optional[str]:
        """Get stock name by code."""
        code = code.zfill(6)
        if code in self.stocks:
            return self.stocks[code].get('name', code)
        return None

    def add_stock(self, code: str, name: str, pinyin: Optional[str] = None):
        """
        Add a stock to the database.

        Args:
            code: 6-digit stock code
            name: Stock name (Chinese)
            pinyin: Optional pinyin abbreviation (auto-generated if not provided)
        """
        code = code.zfill(6)
        info = {'name': name}

        if pinyin:
            info['pinyin'] = pinyin
        elif HAS_PYPINYIN:
            info['pinyin'] = self._generate_pinyin_abbr(name)
            info['full_pinyin'] = ''.join(lazy_pinyin(name))

        self.stocks[code] = info
        self._build_pinyin_index()
        self.save()

    def fetch_all_ashare_stocks(self) -> int:
        """
        Fetch all A-share stocks from AkShare and update database.

        Returns:
            Number of stocks fetched
        """
        if not HAS_AKSHARE:
            logger.error("AkShare not available, cannot fetch stock list")
            return 0

        try:
            logger.info("Fetching A-share stock list from AkShare...")
            # Fetch all A-share stocks
            df = ak.stock_zh_a_spot_em()

            if df.empty:
                logger.error("No data returned from AkShare")
                return 0

            # Check required columns
            if '代码' not in df or '名称' not in df:
                logger.error(f"Unexpected columns: {list(df.columns)}")
                return 0

            # Filter for A-shares only (6-digit codes starting with specific prefixes)
            # Shanghai: 600, 601, 603, 605, 688, 689
            # Shenzhen: 000, 002, 300, 301
            valid_prefixes = ('600', '601', '603', '605', '688', '689',
                            '000', '001', '002', '003', '300', '301')

            filtered = df[df['代码'].str.match(r'^\d{6}$')]
            filtered = filtered[filtered['代码'].str.startswith(valid_prefixes)]

            logger.info(f"Found {len(filtered)} A-share stocks (filtered from {len(df)})")

            # Update database
            count = 0
            for _, row in filtered.iterrows():
                code = str(row['代码']).zfill(6)
                name = str(row['名称'])

                # Skip if already exists (preserve existing data)
                if code in self.stocks:
                    continue

                info = {'name': name}

                # Generate pinyin if available
                if HAS_PYPINYIN:
                    info['pinyin'] = self._generate_pinyin_abbr(name)
                    info['full_pinyin'] = ''.join(lazy_pinyin(name))

                self.stocks[code] = info
                count += 1

            # Rebuild index and save
            self._build_pinyin_index()
            self.save()

            logger.info(f"Added {count} new stocks to database (total: {len(self.stocks)})")
            return count

        except Exception as e:
            logger.error(f"Error fetching stocks from AkShare: {e}")
            return 0

    def update_from_akshare(self, force: bool = False) -> bool:
        """
        Update stock database from AkShare.

        Args:
            force: If True, fetch even if database already has many stocks

        Returns:
            True if update was successful
        """
        # Skip if we already have many stocks (unless forced)
        if not force and len(self.stocks) > 1000:
            logger.info(f"Database already has {len(self.stocks)} stocks, skipping update")
            return True

        count = self.fetch_all_ashare_stocks()
        return count > 0


# Global instance
_stock_db: Optional[StockDatabase] = None


def get_stock_db() -> StockDatabase:
    """Get global stock database instance."""
    global _stock_db
    if _stock_db is None:
        _stock_db = StockDatabase()
    return _stock_db
