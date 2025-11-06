"""
Simplified Data Handler
Handles all data operations: fetching, processing, and feature engineering
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataHandler:
    """All-in-one data handler for trading system."""
    
    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}
        
    def fetch_and_process(self) -> pd.DataFrame:
        """Fetch data and add all features."""
        logger.info(f"Fetching data for {len(self.tickers)} tickers...")
        
        all_data = []
        for ticker in self.tickers:
            df = self._fetch_ticker(ticker)
            if df is not None and len(df) > 0:
                df = self._add_technical_indicators(df)
                df['ticker'] = ticker
                all_data.append(df)
                logger.info(f"✓ {ticker}: {len(df)} days")
        
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.dropna()
        
        logger.info(f"Total data: {len(combined)} rows")
        return combined
    
    def _fetch_ticker(self, ticker: str) -> pd.DataFrame:
        """Fetch data for single ticker."""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=self.start_date, end=self.end_date)
            df.columns = [col.lower() for col in df.columns]
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df = df.reset_index()
            return df
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return None
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators."""
        # Returns
        df['returns'] = df['close'].pct_change()
        
        # Moving Averages
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        bb_ma = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = bb_ma + (2 * bb_std)
        df['bb_lower'] = bb_ma - (2 * bb_std)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Momentum
        df['momentum'] = df['close'] - df['close'].shift(10)
        df['roc'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
        
        return df
    
    def train_test_split(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data chronologically."""
        split_idx = int(len(df) * (1 - test_size))
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()
        return train, test
    
    def get_market_summary(self, df: pd.DataFrame, ticker: str, date: str) -> Dict:
        """Get market summary for a specific date and ticker."""
        try:
            row = df[(df['ticker'] == ticker) & (df['date'] == date)].iloc[0]
            return {
                'ticker': ticker,
                'date': date,
                'close': float(row['close']),
                'volume': int(row['volume']),
                'sma_20': float(row['sma_20']),
                'rsi': float(row['rsi']),
                'macd': float(row['macd']),
                'bb_position': float(row['bb_position']),
                'volume_ratio': float(row['volume_ratio']),
                'momentum': float(row['momentum'])
            }
        except Exception as e:
            logger.error(f"Error getting market summary: {e}")
            return {}
    
    def generate_mock_news(self, ticker: str) -> List[Dict]:
        """Generate mock news for testing."""
        sentiments = [
            {"title": f"{ticker} reports strong quarterly earnings", "sentiment": "positive"},
            {"title": f"{ticker} announces new product launch", "sentiment": "positive"},
            {"title": f"Analysts upgrade {ticker} stock", "sentiment": "positive"},
            {"title": f"Market concerns impact {ticker}", "sentiment": "negative"},
            {"title": f"{ticker} faces regulatory scrutiny", "sentiment": "negative"},
        ]
        return np.random.choice(sentiments, size=2, replace=False).tolist()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    handler = DataHandler(
        tickers=["AAPL"],
        start_date="2023-01-01",
        end_date="2024-01-01"
    )
    
    df = handler.fetch_and_process()
    print("\nData shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())