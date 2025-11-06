#!/usr/bin/env python3
"""
Test Exa API Integration
Quick script to verify Exa API is working correctly
"""

import sys
from pathlib import Path
# Add project's `src` directory to sys.path so imports like `from data_handler import DataHandler`
# work when running this script from the repo root. The previous code used parent.parent
# which points one level too high and caused ModuleNotFoundError.
sys.path.append(str(Path(__file__).parent / 'src'))

from data_handler import DataHandler
import logging
from dotenv import load_dotenv
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


def test_exa_basic():
    """Test basic Exa functionality."""
    print("="*60)
    print("TESTING EXA API INTEGRATION")
    print("="*60)
    
    # Check API key
    exa_key = os.getenv('EXA_API_KEY')
    if not exa_key:
        print("\n⚠️  WARNING: EXA_API_KEY not found in .env file")
        print("The system will fall back to mock news data.")
        print("\nTo use real news:")
        print("1. Get API key from https://exa.ai")
        print("2. Add to .env file: EXA_API_KEY=your_key_here")
        print("="*60)
        return False
    else:
        print(f"✓ EXA_API_KEY found: {exa_key[:8]}...")
    
    # Initialize handler
    print("\n📊 Initializing DataHandler...")
    handler = DataHandler(
        tickers=["AAPL"],
        start_date="2023-01-01",
        end_date="2024-01-01"
    )
    
    if not handler.exa:
        print("❌ Failed to initialize Exa API")
        return False
    
    print("✓ Exa API client initialized")
    
    # Fetch news
    print("\n📰 Fetching news for AAPL (last 7 days)...")
    news = handler.fetch_news_exa("AAPL", days_back=7, max_results=5)
    
    if not news:
        print("❌ No news articles returned")
        return False
    
    # Display results
    print(f"\n✓ Successfully fetched {len(news)} articles!")
    print("\n" + "="*60)
    print("NEWS ARTICLES:")
    print("="*60)
    
    for i, article in enumerate(news, 1):
        print(f"\n{i}. {article['title']}")
        print(f"   📅 Date: {article['published_date']}")
        print(f"   🌐 Source: {article['source']}")
        print(f"   🔗 URL: {article['url'][:60]}...")
        print(f"   📝 Summary: {article['summary'][:150]}...")
    
    print("\n" + "="*60)
    print("✅ EXA API TEST PASSED!")
    print("="*60)
    
    return True


def test_multiple_tickers():
    """Test fetching news for multiple tickers."""
    print("\n" + "="*60)
    print("TESTING MULTIPLE TICKERS")
    print("="*60)
    
    handler = DataHandler(
        tickers=["AAPL", "GOOGL", "MSFT"],
        start_date="2023-01-01",
        end_date="2024-01-01"
    )
    
    if not handler.exa:
        print("⚠️  Exa API not available, skipping multi-ticker test")
        return False
    
    tickers = ["AAPL", "GOOGL", "MSFT"]
    
    for ticker in tickers:
        print(f"\n📰 Fetching news for {ticker}...")
        news = handler.fetch_news_exa(ticker, days_back=3, max_results=3)
        print(f"   ✓ Found {len(news)} articles")
        
        if news:
            print(f"   Latest: {news[0]['title'][:60]}...")
    
    print("\n✅ Multi-ticker test completed!")
    return True


def test_date_filtering():
    """Test date filtering functionality."""
    print("\n" + "="*60)
    print("TESTING DATE FILTERING")
    print("="*60)
    
    handler = DataHandler(
        tickers=["AAPL"],
        start_date="2023-01-01",
        end_date="2024-01-01"
    )
    
    if not handler.exa:
        print("⚠️  Exa API not available, skipping date filter test")
        return False
    
    # Test different time windows
    windows = [1, 7, 30]
    
    for days in windows:
        print(f"\n📅 Fetching news from last {days} day(s)...")
        news = handler.fetch_news_exa("AAPL", days_back=days, max_results=5)
        print(f"   ✓ Found {len(news)} articles")
    
    print("\n✅ Date filtering test completed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "🔍"*30)
    print("EXA API INTEGRATION TEST SUITE")
    print("🔍"*30 + "\n")
    
    results = {
        'Basic Test': test_exa_basic(),
        'Multiple Tickers': test_multiple_tickers(),
        'Date Filtering': test_date_filtering()
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nYou're ready to use Exa API in your trading system!")
        print("\nNext steps:")
        print("1. Run: python scripts/train.py --mode test-llm")
        print("2. Try: python scripts/train.py --mode llm --timesteps 10000")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("\nMake sure:")
        print("1. EXA_API_KEY is in your .env file")
        print("2. Run: pip install exa-py")
        print("3. Check your API key is valid at https://exa.ai")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()