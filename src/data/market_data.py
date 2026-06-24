"""
Market Data Collection Module

Handles collection of market data from multiple exchanges and sources
for the institutional AI crypto trading system.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass

from ..api.exchanges.binance import BinanceClient
from ..api.exchanges.okx import OKXClient
from ..api.exchanges.hyperliquid import HyperliquidClient
from ..api.alpha_vantage import AlphaVantageClient
from ..api.coingecko import CoinGeckoClient


@dataclass
class MarketDataPoint:
    """Market data point structure."""
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    high: float
    low: float
    open: float
    close: float
    source: str
    exchange: str


class MarketDataCollector:
    """Institutional-grade market data collector."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize market data collector."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Initialize API clients
        self.binance_client = BinanceClient()
        self.okx_client = OKXClient()
        self.hyperliquid_client = HyperliquidClient()
        self.alpha_vantage_client = AlphaVantageClient()
        self.coingecko_client = CoinGeckoClient()
        
        # Data storage
        self.market_data: Dict[str, List[MarketDataPoint]] = {}
        self.last_update: Dict[str, datetime] = {}
        
        # Collection settings
        self.update_frequency = self.config.get('update_frequency', 60)  # seconds
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 5)  # seconds
        
        self.logger.info("Market data collector initialized")
    
    async def collect_market_data(self, symbols: List[str], timeframe: str = '1m') -> Dict[str, List[MarketDataPoint]]:
        """
        Collect market data for multiple symbols from multiple sources.
        
        Args:
            symbols: List of trading symbols
            timeframe: Data timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            
        Returns:
            Dictionary of market data by symbol
        """
        try:
            self.logger.info(f"Collecting market data for {len(symbols)} symbols")
            
            # Collect data from multiple sources
            tasks = []
            for symbol in symbols:
                tasks.extend([
                    self._collect_from_binance(symbol, timeframe),
                    self._collect_from_okx(symbol, timeframe),
                    self._collect_from_hyperliquid(symbol, timeframe),
                    self._collect_from_alpha_vantage(symbol, timeframe)
                ])
            
            # Execute all collection tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Data collection error: {result}")
                elif result:
                    symbol, data = result
                    if symbol not in self.market_data:
                        self.market_data[symbol] = []
                    self.market_data[symbol].extend(data)
            
            # Update last update times
            for symbol in symbols:
                self.last_update[symbol] = datetime.now()
            
            self.logger.info(f"Collected data for {len(self.market_data)} symbols")
            return self.market_data
            
        except Exception as e:
            self.logger.error(f"Error collecting market data: {e}")
            return {}
    
    async def _collect_from_binance(self, symbol: str, timeframe: str) -> Optional[Tuple[str, List[MarketDataPoint]]]:
        """Collect data from Binance."""
        try:
            data = await self.binance_client.get_ohlcv(symbol, timeframe, limit=100)
            if data:
                market_data = self._convert_to_market_data_points(data, symbol, 'binance')
                return symbol, market_data
        except Exception as e:
            self.logger.warning(f"Binance data collection failed for {symbol}: {e}")
        return None
    
    async def _collect_from_okx(self, symbol: str, timeframe: str) -> Optional[Tuple[str, List[MarketDataPoint]]]:
        """Collect data from OKX."""
        try:
            data = await self.okx_client.get_ohlcv(symbol, timeframe, limit=100)
            if data:
                market_data = self._convert_to_market_data_points(data, symbol, 'okx')
                return symbol, market_data
        except Exception as e:
            self.logger.warning(f"OKX data collection failed for {symbol}: {e}")
        return None
    
    async def _collect_from_hyperliquid(self, symbol: str, timeframe: str) -> Optional[Tuple[str, List[MarketDataPoint]]]:
        """Collect data from Hyperliquid."""
        try:
            data = await self.hyperliquid_client.get_ohlcv(symbol, timeframe, limit=100)
            if data:
                market_data = self._convert_to_market_data_points(data, symbol, 'hyperliquid')
                return symbol, market_data
        except Exception as e:
            self.logger.warning(f"Hyperliquid data collection failed for {symbol}: {e}")
        return None
    
    async def _collect_from_alpha_vantage(self, symbol: str, timeframe: str) -> Optional[Tuple[str, List[MarketDataPoint]]]:
        """Collect data from Alpha Vantage."""
        try:
            data = await self.alpha_vantage_client.get_crypto_ohlcv(symbol, timeframe)
            if data:
                market_data = self._convert_to_market_data_points(data, symbol, 'alpha_vantage')
                return symbol, market_data
        except Exception as e:
            self.logger.warning(f"Alpha Vantage data collection failed for {symbol}: {e}")
        return None
    
    def _convert_to_market_data_points(self, data: List[Dict], symbol: str, source: str) -> List[MarketDataPoint]:
        """Convert raw data to MarketDataPoint objects."""
        market_data = []
        
        for item in data:
            try:
                point = MarketDataPoint(
                    timestamp=item.get('timestamp', datetime.now()),
                    symbol=symbol,
                    price=float(item.get('close', 0)),
                    volume=float(item.get('volume', 0)),
                    high=float(item.get('high', 0)),
                    low=float(item.get('low', 0)),
                    open=float(item.get('open', 0)),
                    close=float(item.get('close', 0)),
                    source=source,
                    exchange=source
                )
                market_data.append(point)
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Error converting data point: {e}")
                continue
        
        return market_data
    
    def get_latest_data(self, symbol: str) -> Optional[MarketDataPoint]:
        """Get latest market data for a symbol."""
        if symbol in self.market_data and self.market_data[symbol]:
            return max(self.market_data[symbol], key=lambda x: x.timestamp)
        return None
    
    def get_historical_data(self, symbol: str, start_time: datetime, end_time: datetime) -> List[MarketDataPoint]:
        """Get historical market data for a symbol."""
        if symbol not in self.market_data:
            return []
        
        return [
            point for point in self.market_data[symbol]
            if start_time <= point.timestamp <= end_time
        ]
    
    def get_dataframe(self, symbol: str) -> pd.DataFrame:
        """Convert market data to pandas DataFrame."""
        if symbol not in self.market_data:
            return pd.DataFrame()
        
        data = []
        for point in self.market_data[symbol]:
            data.append({
                'timestamp': point.timestamp,
                'symbol': point.symbol,
                'price': point.price,
                'volume': point.volume,
                'high': point.high,
                'low': point.low,
                'open': point.open,
                'close': point.close,
                'source': point.source,
                'exchange': point.exchange
            })
        
        return pd.DataFrame(data)
    
    def calculate_statistics(self, symbol: str) -> Dict[str, Any]:
        """Calculate market statistics for a symbol."""
        df = self.get_dataframe(symbol)
        if df.empty:
            return {}
        
        stats = {
            'current_price': df['close'].iloc[-1] if len(df) > 0 else 0,
            'price_change_24h': 0,
            'price_change_pct_24h': 0,
            'volume_24h': df['volume'].sum() if len(df) > 0 else 0,
            'high_24h': df['high'].max() if len(df) > 0 else 0,
            'low_24h': df['low'].min() if len(df) > 0 else 0,
            'volatility': df['close'].pct_change().std() if len(df) > 1 else 0,
            'avg_volume': df['volume'].mean() if len(df) > 0 else 0
        }
        
        # Calculate 24h price change if we have enough data
        if len(df) >= 1440:  # 24 hours of minute data
            stats['price_change_24h'] = df['close'].iloc[-1] - df['close'].iloc[-1440]
            stats['price_change_pct_24h'] = (stats['price_change_24h'] / df['close'].iloc[-1440]) * 100
        
        return stats
    
    async def start_continuous_collection(self, symbols: List[str], callback=None):
        """Start continuous market data collection."""
        self.logger.info(f"Starting continuous data collection for {len(symbols)} symbols")
        
        while True:
            try:
                data = await self.collect_market_data(symbols)
                
                if callback:
                    await callback(data)
                
                await asyncio.sleep(self.update_frequency)
                
            except Exception as e:
                self.logger.error(f"Error in continuous collection: {e}")
                await asyncio.sleep(self.retry_delay)
    
    def cleanup(self):
        """Clean up resources."""
        self.logger.info("Cleaning up market data collector")
        # Close any open connections or resources


# Factory function for creating market data collector
def create_market_data_collector(config: Optional[Dict[str, Any]] = None) -> MarketDataCollector:
    """Create a market data collector with the specified configuration."""
    return MarketDataCollector(config) 