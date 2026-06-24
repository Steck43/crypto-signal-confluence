"""
Custom Technical Indicators Module

Advanced technical indicators for the institutional AI crypto trading system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging


class CustomIndicators:
    """Custom technical indicators."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize custom indicators."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        self.logger.info("Custom indicators initialized")
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all custom indicators."""
        if df.empty:
            return df
        
        result_df = df.copy()
        
        # Volume indicators
        result_df = self._add_volume_indicators(result_df)
        
        # Price indicators
        result_df = self._add_price_indicators(result_df)
        
        # Momentum indicators
        result_df = self._add_momentum_indicators(result_df)
        
        # Volatility indicators
        result_df = self._add_volatility_indicators(result_df)
        
        return result_df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        # Volume moving average
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ma_50'] = df['volume'].rolling(window=50).mean()
        
        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        
        # Volume price trend
        df['vpt'] = (df['volume'] * ((df['close'] - df['close'].shift(1)) / df['close'].shift(1))).cumsum()
        
        return df
    
    def _add_price_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based indicators."""
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # Price channels
        df['upper_channel'] = df['high'].rolling(window=20).max()
        df['lower_channel'] = df['low'].rolling(window=20).min()
        df['middle_channel'] = (df['upper_channel'] + df['lower_channel']) / 2
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        # RSI
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        df['rsi_21'] = self._calculate_rsi(df['close'], 21)
        
        # MACD
        macd_data = self._calculate_macd(df['close'])
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_histogram'] = macd_data['histogram']
        
        # Stochastic
        stoch_data = self._calculate_stochastic(df, 14)
        df['stoch_k'] = stoch_data['k']
        df['stoch_d'] = stoch_data['d']
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        # Bollinger Bands
        bb_data = self._calculate_bollinger_bands(df['close'], 20, 2)
        df['bb_upper'] = bb_data['upper']
        df['bb_middle'] = bb_data['middle']
        df['bb_lower'] = bb_data['lower']
        df['bb_width'] = bb_data['width']
        
        # Average True Range
        df['atr_14'] = self._calculate_atr(df, 14)
        
        # Historical Volatility
        df['volatility_20'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
        """Calculate Stochastic oscillator."""
        lowest_low = df['low'].rolling(window=period).min()
        highest_high = df['high'].rolling(window=period).max()
        
        k = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=3).mean()
        
        return {
            'k': k,
            'd': d
        }
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        width = (upper - lower) / middle
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'width': width
        }
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        
        return atr


# Factory function
def create_custom_indicators(config: Optional[Dict[str, Any]] = None) -> CustomIndicators:
    """Create custom indicators."""
    return CustomIndicators(config) 