"""
Alpha Vantage API Client for Crypto Trading System

Features:
- Real-time crypto data (SOL, BTC, ETH, HYPE)
- Technical indicators (RSI, MACD, Bollinger Bands, ATR)
- News sentiment API with AI-powered scoring
- Historical data for backtesting (minimum 2 years)
- Rate limiting: 75 requests/minute (premium tier)
- Intelligent caching and error handling
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path
import time
from enum import Enum
import hashlib

from .secure_manager import get_secure_api_manager

class TimeSeriesFunction(Enum):
    """Alpha Vantage time series function types."""
    DAILY = "DIGITAL_CURRENCY_DAILY"
    WEEKLY = "DIGITAL_CURRENCY_WEEKLY"
    MONTHLY = "DIGITAL_CURRENCY_MONTHLY"
    INTRADAY = "DIGITAL_CURRENCY_INTRADAY"

class TechnicalIndicator(Enum):
    """Supported technical indicators."""
    RSI = "RSI"
    MACD = "MACD"
    BBANDS = "BBANDS"
    ATR = "ATR"
    SMA = "SMA"
    EMA = "EMA"
    STOCH = "STOCH"
    ADX = "ADX"
    CCI = "CCI"
    AROON = "AROON"
    OBV = "OBV"

@dataclass
class CryptoPrice:
    """Crypto price data structure."""
    symbol: str
    price: float
    high: float
    low: float
    volume: float
    market_cap: float
    timestamp: datetime
    change_24h: float
    change_percentage_24h: float

@dataclass
class TechnicalIndicatorData:
    """Technical indicator data structure."""
    symbol: str
    indicator: str
    values: Dict[str, float]
    timestamp: datetime
    parameters: Dict[str, Any]

@dataclass
class NewsItem:
    """News item from Alpha Vantage."""
    title: str
    url: str
    time_published: datetime
    source: str
    summary: str
    sentiment_score: float
    sentiment_label: str
    ticker_sentiment: List[Dict[str, Any]]

class AlphaVantageClient:
    """
    Professional Alpha Vantage API client for crypto trading.
    
    Features:
    - Comprehensive crypto data collection
    - Technical indicator calculation
    - News sentiment analysis
    - Rate limiting and caching
    - Error handling and retry logic
    """
    
    def __init__(self, cache_dir: str = "cache/alpha_vantage"):
        """Initialize the Alpha Vantage client."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Get secure API manager
        self.api_manager = get_secure_api_manager()
        
        # Base URLs
        self.base_url = "https://www.alphavantage.co/query"
        
        # Rate limiting (75 requests/minute for premium)
        self.rate_limit = 75
        self.rate_window = 60  # seconds
        self.request_times = []
        
        # Cache settings
        self.cache_duration = {
            'intraday': 60,      # 1 minute for intraday data
            'daily': 300,        # 5 minutes for daily data
            'technical': 300,    # 5 minutes for technical indicators
            'news': 900,         # 15 minutes for news
            'overview': 3600     # 1 hour for company overview
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Primary crypto symbols
        self.primary_symbols = ['BTC', 'ETH', 'SOL']
        self.target_symbols = ['SOL', 'HYPE']
        
        self.logger.info("📊 Alpha Vantage client initialized")
    
    async def get_crypto_price(self, symbol: str, market: str = 'USD') -> Optional[CryptoPrice]:
        """
        Get current crypto price data.
        
        Args:
            symbol: Crypto symbol (BTC, ETH, SOL, etc.)
            market: Market currency (USD, EUR, etc.)
            
        Returns:
            CryptoPrice object or None if failed
        """
        try:
            # Check cache first
            cache_key = f"price_{symbol}_{market}"
            cached_data = self._get_cached_data(cache_key, 'intraday')
            if cached_data:
                return self._parse_price_data(cached_data, symbol)
            
            # Make API request
            params = {
                'function': 'DIGITAL_CURRENCY_DAILY',
                'symbol': symbol,
                'market': market,
                'apikey': self._get_api_key()
            }
            
            data = await self._make_request(params)
            if not data:
                return None
            
            # Cache the result
            self._cache_data(cache_key, data, 'intraday')
            
            return self._parse_price_data(data, symbol)
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching price for {symbol}: {e}")
            return None
    
    async def get_historical_data(self, symbol: str, market: str = 'USD', 
                                 function: TimeSeriesFunction = TimeSeriesFunction.DAILY,
                                 years_back: int = 2) -> Optional[pd.DataFrame]:
        """
        Get historical crypto data for backtesting.
        
        Args:
            symbol: Crypto symbol
            market: Market currency
            function: Time series function
            years_back: Years of historical data
            
        Returns:
            DataFrame with historical data or None if failed
        """
        try:
            cache_key = f"historical_{symbol}_{market}_{function.value}_{years_back}"
            cached_data = self._get_cached_data(cache_key, 'daily')
            if cached_data:
                return self._parse_historical_data(cached_data)
            
            params = {
                'function': function.value,
                'symbol': symbol,
                'market': market,
                'apikey': self._get_api_key()
            }
            
            data = await self._make_request(params)
            if not data:
                return None
            
            self._cache_data(cache_key, data, 'daily')
            
            return self._parse_historical_data(data)
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching historical data for {symbol}: {e}")
            return None
    
    async def get_technical_indicator(self, symbol: str, indicator: TechnicalIndicator,
                                    interval: str = 'daily', time_period: int = 14,
                                    **kwargs) -> Optional[TechnicalIndicatorData]:
        """
        Get technical indicator data.
        
        Args:
            symbol: Crypto symbol
            indicator: Technical indicator type
            interval: Time interval
            time_period: Period for calculation
            **kwargs: Additional parameters
            
        Returns:
            TechnicalIndicatorData object or None if failed
        """
        try:
            cache_key = f"tech_{symbol}_{indicator.value}_{interval}_{time_period}"
            cached_data = self._get_cached_data(cache_key, 'technical')
            if cached_data:
                return self._parse_technical_data(cached_data, symbol, indicator.value)
            
            params = {
                'function': indicator.value,
                'symbol': symbol,
                'interval': interval,
                'time_period': time_period,
                'apikey': self._get_api_key()
            }
            
            # Add additional parameters
            params.update(kwargs)
            
            data = await self._make_request(params)
            if not data:
                return None
            
            self._cache_data(cache_key, data, 'technical')
            
            return self._parse_technical_data(data, symbol, indicator.value)
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching {indicator.value} for {symbol}: {e}")
            return None
    
    async def get_multiple_indicators(self, symbol: str, indicators: List[TechnicalIndicator],
                                    interval: str = 'daily') -> Dict[str, TechnicalIndicatorData]:
        """
        Get multiple technical indicators for a symbol.
        
        Args:
            symbol: Crypto symbol
            indicators: List of indicators to fetch
            interval: Time interval
            
        Returns:
            Dictionary of indicator data
        """
        results = {}
        
        # Create tasks for concurrent execution
        tasks = []
        for indicator in indicators:
            task = self.get_technical_indicator(symbol, indicator, interval)
            tasks.append((indicator.value, task))
        
        # Execute all tasks concurrently
        for indicator_name, task in tasks:
            try:
                result = await task
                if result:
                    results[indicator_name] = result
            except Exception as e:
                self.logger.error(f"❌ Error fetching {indicator_name} for {symbol}: {e}")
        
        return results
    
    async def get_news_sentiment(self, topics: List[str] = None, 
                               time_from: datetime = None,
                               time_to: datetime = None) -> List[NewsItem]:
        """
        Get news sentiment data.
        
        Args:
            topics: List of topics to search for
            time_from: Start time for news search
            time_to: End time for news search
            
        Returns:
            List of NewsItem objects
        """
        try:
            if topics is None:
                topics = ['cryptocurrency', 'bitcoin', 'ethereum', 'solana', 'crypto']
            
            cache_key = f"news_{'_'.join(topics)}"
            cached_data = self._get_cached_data(cache_key, 'news')
            if cached_data:
                return self._parse_news_data(cached_data)
            
            params = {
                'function': 'NEWS_SENTIMENT',
                'topics': ','.join(topics),
                'apikey': self._get_api_key()
            }
            
            if time_from:
                params['time_from'] = time_from.strftime('%Y%m%dT%H%M')
            if time_to:
                params['time_to'] = time_to.strftime('%Y%m%dT%H%M')
            
            data = await self._make_request(params)
            if not data:
                return []
            
            self._cache_data(cache_key, data, 'news')
            
            return self._parse_news_data(data)
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching news sentiment: {e}")
            return []
    
    async def get_market_overview(self, symbols: List[str] = None) -> Dict[str, Any]:
        """
        Get market overview for multiple symbols.
        
        Args:
            symbols: List of symbols to get overview for
            
        Returns:
            Dictionary with market overview data
        """
        if symbols is None:
            symbols = self.primary_symbols
        
        overview_data = {}
        
        # Get data for each symbol concurrently
        tasks = []
        for symbol in symbols:
            task = self.get_crypto_price(symbol)
            tasks.append((symbol, task))
        
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        for i, (symbol, result) in enumerate(zip(symbols, results)):
            if isinstance(result, Exception):
                self.logger.error(f"❌ Error fetching overview for {symbol}: {result}")
                continue
            
            if result:
                overview_data[symbol] = result
        
        return overview_data
    
    async def get_sol_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive SOL analysis (primary target).
        
        Returns:
            Dictionary with SOL analysis data
        """
        try:
            sol_data = {}
            
            # Get current price
            price_data = await self.get_crypto_price('SOL')
            if price_data:
                sol_data['price'] = price_data.__dict__
            
            # Get technical indicators
            indicators = [
                TechnicalIndicator.RSI,
                TechnicalIndicator.MACD,
                TechnicalIndicator.BBANDS,
                TechnicalIndicator.ATR
            ]
            
            technical_data = await self.get_multiple_indicators('SOL', indicators)
            sol_data['technical'] = {k: v.__dict__ for k, v in technical_data.items()}
            
            # Get SOL-specific news
            news_data = await self.get_news_sentiment(['solana', 'sol'])
            sol_data['news'] = [item.__dict__ for item in news_data[:10]]  # Latest 10 items
            
            # Calculate overall sentiment
            if news_data:
                sentiment_scores = [item.sentiment_score for item in news_data]
                sol_data['overall_sentiment'] = {
                    'average_score': np.mean(sentiment_scores),
                    'positive_ratio': len([s for s in sentiment_scores if s > 0]) / len(sentiment_scores),
                    'article_count': len(news_data)
                }
            
            sol_data['timestamp'] = datetime.now().isoformat()
            
            return sol_data
            
        except Exception as e:
            self.logger.error(f"❌ Error in SOL analysis: {e}")
            return {}
    
    def _get_api_key(self) -> str:
        """Get Alpha Vantage API key from secure manager."""
        creds = self.api_manager.get_api_credentials('alpha_vantage')
        if not creds:
            raise ValueError("Alpha Vantage API credentials not found")
        return creds['api_key']
    
    async def _make_request(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make API request with rate limiting and error handling."""
        try:
            # Check rate limit
            await self._check_rate_limit()
            
            # Make request
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Check for API errors
                        if 'Error Message' in data:
                            self.logger.error(f"❌ API Error: {data['Error Message']}")
                            return None
                        
                        if 'Note' in data:
                            self.logger.warning(f"⚠️ API Note: {data['Note']}")
                            return None
                        
                        return data
                    else:
                        self.logger.error(f"❌ HTTP Error {response.status}: {await response.text()}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"❌ Request error: {e}")
            return None
    
    async def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        now = time.time()
        
        # Remove old requests
        self.request_times = [t for t in self.request_times if now - t < self.rate_window]
        
        # Check if we're at the limit
        if len(self.request_times) >= self.rate_limit:
            wait_time = self.rate_window - (now - self.request_times[0])
            if wait_time > 0:
                self.logger.warning(f"⏳ Rate limit hit, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        
        # Record this request
        self.request_times.append(now)
    
    def _get_cached_data(self, cache_key: str, data_type: str) -> Optional[Dict[str, Any]]:
        """Get cached data if still valid."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache is still valid
            cache_time = datetime.fromisoformat(cache_data['timestamp'])
            cache_duration = self.cache_duration.get(data_type, 300)
            
            if datetime.now() - cache_time > timedelta(seconds=cache_duration):
                return None
            
            return cache_data['data']
            
        except Exception as e:
            self.logger.error(f"❌ Error loading cache {cache_key}: {e}")
            return None
    
    def _cache_data(self, cache_key: str, data: Dict[str, Any], data_type: str):
        """Cache data with timestamp."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'data': data,
                'type': data_type
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"❌ Error caching data {cache_key}: {e}")
    
    def _parse_price_data(self, data: Dict[str, Any], symbol: str) -> Optional[CryptoPrice]:
        """Parse price data from API response."""
        try:
            if 'Time Series (Digital Currency Daily)' in data:
                time_series = data['Time Series (Digital Currency Daily)']
                latest_date = max(time_series.keys())
                latest_data = time_series[latest_date]
                
                return CryptoPrice(
                    symbol=symbol,
                    price=float(latest_data['4a. close (USD)']),
                    high=float(latest_data['2a. high (USD)']),
                    low=float(latest_data['3a. low (USD)']),
                    volume=float(latest_data['5. volume']),
                    market_cap=float(latest_data['6. market cap (USD)']),
                    timestamp=datetime.strptime(latest_date, '%Y-%m-%d'),
                    change_24h=0.0,  # Would need to calculate
                    change_percentage_24h=0.0  # Would need to calculate
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error parsing price data: {e}")
            return None
    
    def _parse_historical_data(self, data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Parse historical data into DataFrame."""
        try:
            if 'Time Series (Digital Currency Daily)' in data:
                time_series = data['Time Series (Digital Currency Daily)']
                
                df_data = []
                for date, values in time_series.items():
                    row = {
                        'date': pd.to_datetime(date),
                        'open': float(values['1a. open (USD)']),
                        'high': float(values['2a. high (USD)']),
                        'low': float(values['3a. low (USD)']),
                        'close': float(values['4a. close (USD)']),
                        'volume': float(values['5. volume']),
                        'market_cap': float(values['6. market cap (USD)'])
                    }
                    df_data.append(row)
                
                df = pd.DataFrame(df_data)
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
                
                return df
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error parsing historical data: {e}")
            return None
    
    def _parse_technical_data(self, data: Dict[str, Any], symbol: str, 
                            indicator: str) -> Optional[TechnicalIndicatorData]:
        """Parse technical indicator data."""
        try:
            # Find the technical analysis key
            tech_key = None
            for key in data.keys():
                if 'Technical Analysis' in key:
                    tech_key = key
                    break
            
            if not tech_key:
                return None
            
            tech_data = data[tech_key]
            latest_date = max(tech_data.keys())
            latest_values = tech_data[latest_date]
            
            return TechnicalIndicatorData(
                symbol=symbol,
                indicator=indicator,
                values=latest_values,
                timestamp=datetime.strptime(latest_date, '%Y-%m-%d'),
                parameters=data.get('Meta Data', {})
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error parsing technical data: {e}")
            return None
    
    def _parse_news_data(self, data: Dict[str, Any]) -> List[NewsItem]:
        """Parse news sentiment data."""
        try:
            if 'feed' not in data:
                return []
            
            news_items = []
            for item in data['feed']:
                news_item = NewsItem(
                    title=item.get('title', ''),
                    url=item.get('url', ''),
                    time_published=datetime.strptime(item.get('time_published', ''), '%Y%m%dT%H%M%S'),
                    source=item.get('source', ''),
                    summary=item.get('summary', ''),
                    sentiment_score=float(item.get('overall_sentiment_score', 0)),
                    sentiment_label=item.get('overall_sentiment_label', 'Neutral'),
                    ticker_sentiment=item.get('ticker_sentiment', [])
                )
                news_items.append(news_item)
            
            return news_items
            
        except Exception as e:
            self.logger.error(f"❌ Error parsing news data: {e}")
            return []
    
    def cleanup_cache(self, max_age_days: int = 7):
        """Clean up old cache files."""
        try:
            cutoff_time = datetime.now() - timedelta(days=max_age_days)
            
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if file_time < cutoff_time:
                        cache_file.unlink()
                        self.logger.info(f"🧹 Cleaned up cache file: {cache_file.name}")
                except Exception as e:
                    self.logger.error(f"❌ Error cleaning cache file {cache_file}: {e}")
                    
        except Exception as e:
            self.logger.error(f"❌ Error during cache cleanup: {e}")

# Convenience function
def get_alpha_vantage_client() -> AlphaVantageClient:
    """Get Alpha Vantage client instance."""
    return AlphaVantageClient()