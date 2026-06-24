"""
Multi-Timeframe Analysis Module

Advanced multi-timeframe analysis for the institutional AI crypto trading system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging


class MultiTimeframeAnalyzer:
    """Multi-timeframe analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize multi-timeframe analyzer."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Timeframe configuration
        self.timeframes = self.config.get('timeframes', ['1m', '5m', '15m', '1h', '4h', '1d'])
        
        self.logger.info("Multi-timeframe analyzer initialized")
    
    def analyze_multiple_timeframes(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze data across multiple timeframes.
        
        Args:
            data_dict: Dictionary of DataFrames for each timeframe
            
        Returns:
            Analysis results for each timeframe
        """
        results = {}
        
        for timeframe, df in data_dict.items():
            if df.empty:
                continue
            
            analysis = self._analyze_single_timeframe(df, timeframe)
            results[timeframe] = analysis
        
        # Combine results
        combined_analysis = self._combine_timeframe_analysis(results)
        
        return combined_analysis
    
    def _analyze_single_timeframe(self, df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """Analyze a single timeframe."""
        analysis = {
            'timeframe': timeframe,
            'trend': self._analyze_trend(df),
            'momentum': self._analyze_momentum(df),
            'volatility': self._analyze_volatility(df),
            'support_resistance': self._find_support_resistance(df),
            'volume_analysis': self._analyze_volume(df)
        }
        
        return analysis
    
    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend direction and strength."""
        if df.empty:
            return {'direction': 'neutral', 'strength': 0.0}
        
        # Calculate moving averages
        ma_short = df['close'].rolling(window=10).mean()
        ma_long = df['close'].rolling(window=50).mean()
        
        current_short = ma_short.iloc[-1]
        current_long = ma_long.iloc[-1]
        
        if current_short > current_long:
            direction = 'uptrend'
            strength = min(1.0, (current_short - current_long) / current_long)
        elif current_short < current_long:
            direction = 'downtrend'
            strength = min(1.0, (current_long - current_short) / current_long)
        else:
            direction = 'neutral'
            strength = 0.0
        
        return {
            'direction': direction,
            'strength': strength,
            'ma_short': current_short,
            'ma_long': current_long
        }
    
    def _analyze_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze momentum indicators."""
        if df.empty:
            return {}
        
        # RSI
        rsi = self._calculate_rsi(df['close'], 14)
        current_rsi = rsi.iloc[-1] if not rsi.empty else 50
        
        # MACD
        macd_data = self._calculate_macd(df['close'])
        current_macd = macd_data['macd'].iloc[-1] if not macd_data['macd'].empty else 0
        current_signal = macd_data['signal'].iloc[-1] if not macd_data['signal'].empty else 0
        
        return {
            'rsi': current_rsi,
            'rsi_signal': 'oversold' if current_rsi < 30 else 'overbought' if current_rsi > 70 else 'neutral',
            'macd': current_macd,
            'macd_signal': current_signal,
            'macd_histogram': current_macd - current_signal
        }
    
    def _analyze_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volatility."""
        if df.empty:
            return {}
        
        # Calculate volatility
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=20).std() * np.sqrt(252)
        current_volatility = volatility.iloc[-1] if not volatility.empty else 0
        
        # Bollinger Bands
        bb_data = self._calculate_bollinger_bands(df['close'])
        current_price = df['close'].iloc[-1]
        bb_position = (current_price - bb_data['lower'].iloc[-1]) / (bb_data['upper'].iloc[-1] - bb_data['lower'].iloc[-1])
        
        return {
            'volatility': current_volatility,
            'bb_position': bb_position,
            'bb_signal': 'oversold' if bb_position < 0.2 else 'overbought' if bb_position > 0.8 else 'neutral'
        }
    
    def _find_support_resistance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Find support and resistance levels."""
        if df.empty:
            return {}
        
        # Simple support/resistance detection
        recent_highs = df['high'].tail(20)
        recent_lows = df['low'].tail(20)
        
        resistance = recent_highs.max()
        support = recent_lows.min()
        current_price = df['close'].iloc[-1]
        
        return {
            'support': support,
            'resistance': resistance,
            'distance_to_support': (current_price - support) / current_price,
            'distance_to_resistance': (resistance - current_price) / current_price
        }
    
    def _analyze_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns."""
        if df.empty:
            return {}
        
        # Volume analysis
        avg_volume = df['volume'].rolling(window=20).mean()
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume.iloc[-1] if not avg_volume.empty else 1.0
        
        return {
            'volume_ratio': volume_ratio,
            'volume_signal': 'high' if volume_ratio > 1.5 else 'low' if volume_ratio < 0.5 else 'normal'
        }
    
    def _combine_timeframe_analysis(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Combine analysis from multiple timeframes."""
        combined = {
            'timeframe_consensus': {},
            'overall_trend': 'neutral',
            'trend_strength': 0.0,
            'momentum_consensus': 'neutral',
            'volatility_consensus': 'normal'
        }
        
        # Analyze trend consensus
        trends = [result['trend']['direction'] for result in results.values() if 'trend' in result]
        if trends:
            combined['overall_trend'] = max(set(trends), key=trends.count)
        
        # Calculate average trend strength
        strengths = [result['trend']['strength'] for result in results.values() if 'trend' in result]
        if strengths:
            combined['trend_strength'] = np.mean(strengths)
        
        # Momentum consensus
        rsi_signals = [result['momentum']['rsi_signal'] for result in results.values() if 'momentum' in result]
        if rsi_signals:
            combined['momentum_consensus'] = max(set(rsi_signals), key=rsi_signals.count)
        
        return combined
    
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
        
        return {
            'macd': macd,
            'signal': signal_line
        }
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }


# Factory function
def create_multitimeframe_analyzer(config: Optional[Dict[str, Any]] = None) -> MultiTimeframeAnalyzer:
    """Create a multi-timeframe analyzer."""
    return MultiTimeframeAnalyzer(config) 