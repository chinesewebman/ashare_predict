#!/usr/bin/env python3
"""
Fetch and update the full A-share stock list from AkShare.

This script updates the local stock database with all A-share stocks
from Shanghai and Shenzhen exchanges, including pinyin support for search.
"""
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.stock_db import get_stock_db

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main function to update stock database."""
    print("="*60)
    print("🔄 A-Share Stock Database Updater")
    print("="*60)

    # Get stock database
    db = get_stock_db()

    print(f"\nCurrent database: {len(db.stocks)} stocks")

    # Fetch all stocks from AkShare
    print("\n📥 Fetching stock list from AkShare...")
    count = db.fetch_all_ashare_stocks()

    if count > 0:
        print(f"✅ Successfully added {count} new stocks")
        print(f"📊 Total stocks in database: {len(db.stocks)}")

        # Show some examples
        print("\n🔍 Sample stocks:")
        sample_codes = list(db.stocks.keys())[:5]
        for code in sample_codes:
            info = db.stocks[code]
            pinyin = info.get('pinyin', 'N/A')
            print(f"  {code}: {info['name']:10} (pinyin: {pinyin})")

        # Test search
        print("\n🔎 Testing pinyin search:")
        test_searches = ['jd', 'zsyh', 'payh', 'mt']
        for query in test_searches:
            results = db.search(query, limit=3)
            if results:
                names = [r['name'] for r in results[:3]]
                print(f"  '{query}' -> {', '.join(names)}")
            else:
                print(f"  '{query}' -> No results")

    else:
        print("❌ Failed to fetch stocks")
        return 1

    print("\n" + "="*60)
    print("✨ Database update complete!")
    print("="*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
