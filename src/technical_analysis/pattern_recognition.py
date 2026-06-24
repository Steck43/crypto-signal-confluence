"""
Pattern Recognition Module

Advanced pattern recognition algorithms for technical analysis
in the institutional AI crypto trading system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Pattern:
    """Pattern recognition result."""
    name: str
    confidence: float
    start_index: int
    end_index: int
    pattern_type: str  # 'bullish', 'bearish', 'neutral'
    description: str


class PatternRecognizer:
    """Advanced pattern recognition for technical analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize pattern recognizer."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Pattern detection parameters
        self.min_pattern_length = self.config.get('min_pattern_length', 5)
        self.max_pattern_length = self.config.get('max_pattern_length', 50)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        self.logger.info("Pattern recognizer initialized")
    
    def detect_patterns(self, df: pd.DataFrame) -> List[Pattern]:
        """
        Detect patterns in price data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of detected patterns
        """
        if df.empty or len(df) < self.min_pattern_length:
            return []
        
        patterns = []
        
        # Detect various pattern types
        patterns.extend(self._detect_candlestick_patterns(df))
        patterns.extend(self._detect_chart_patterns(df))
        patterns.extend(self._detect_support_resistance(df))
        patterns.extend(self._detect_trend_patterns(df))
        
        # Sort by confidence
        patterns.sort(key=lambda x: x.confidence, reverse=True)
        
        return patterns
    
    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> List[Pattern]:
        """Detect candlestick patterns."""
        patterns = []
        
        if len(df) < 3:
            return patterns
        
        # Doji pattern
        doji_patterns = self._find_doji_patterns(df)
        patterns.extend(doji_patterns)
        
        # Hammer pattern
        hammer_patterns = self._find_hammer_patterns(df)
        patterns.extend(hammer_patterns)
        
        # Engulfing patterns
        engulfing_patterns = self._find_engulfing_patterns(df)
        patterns.extend(engulfing_patterns)
        
        return patterns
    
    def _find_doji_patterns(self, df: pd.DataFrame) -> List[Pattern]:
        """Find doji candlestick patterns."""
        patterns = []
        
        for i in range(1, len(df) - 1):
            current = df.iloc[i]
            prev = df.iloc[i - 1]
            next_candle = df.iloc[i + 1]
            
            # Doji criteria: open and close are very close
            body_size = abs(current['close'] - current['open'])
            total_range = current['high'] - current['low']
            
            if total_range > 0 and body_size / total_range < 0.1:
                # Determine doji type
                if current['high'] - max(current['open'], current['close']) > 2 * body_size:
                    pattern_type = 'bearish'  # Gravestone doji
                elif min(current['open'], current['close']) - current['low'] > 2 * body_size:
                    pattern_type = 'bullish'  # Dragonfly doji
                else:
                    pattern_type = 'neutral'  # Standard doji
                
                pattern = Pattern(
                    name='Doji',
                    confidence=0.8,
                    start_index=i,
                    end_index=i,
                    pattern_type=pattern_type,
                    description=f'{pattern_type.capitalize()} doji pattern detected'
                )
                patterns.append(pattern)
        
        return patterns
    
    def _find_hammer_patterns(self, df: pd.DataFrame) -> List[Pattern]:
        """Find hammer candlestick patterns."""
        patterns = []
        
        for i in range(1, len(df) - 1):
            current = df.iloc[i]
            
            body_size = abs(current['close'] - current['open'])
            lower_shadow = min(current['open'], current['close']) - current['low']
            upper_shadow = current['high'] - max(current['open'], current['close'])
            
            # Hammer criteria
            if (lower_shadow > 2 * body_size and 
                upper_shadow < body_size and
                body_size > 0):
                
                pattern_type = 'bullish' if current['close'] > current['open'] else 'neutral'
                
                pattern = Pattern(
                    name='Hammer',
                    confidence=0.75,
                    start_index=i,
                    end_index=i,
                    pattern_type=pattern_type,
                    description=f'{pattern_type.capitalize()} hammer pattern detected'
                )
                patterns.append(pattern)
        
        return patterns
    
    def _find_engulfing_patterns(self, df: pd.DataFrame) -> List[Pattern]:
        """Find engulfing candlestick patterns."""
        patterns = []
        
        for i in range(1, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i - 1]
            
            current_body = abs(current['close'] - current['open'])
            prev_body = abs(prev['close'] - prev['open'])
            
            # Bullish engulfing
            if (current['close'] > current['open'] and  # Current is bullish
                prev['close'] < prev['open'] and  # Previous is bearish
                current['open'] < prev['close'] and  # Current opens below previous close
                current['close'] > prev['open'] and  # Current closes above previous open
                current_body > prev_body):  # Current body engulfs previous
                
                pattern = Pattern(
                    name='Bullish Engulfing',
                    confidence=0.85,
                    start_index=i-1,
                    end_index=i,
                    pattern_type='bullish',
                    description='Bullish engulfing pattern detected'
                )
                patterns.append(pattern)
            
            # Bearish engulfing
            elif (current['close'] < current['open'] and  # Current is bearish
                  prev['close'] > prev['open'] and  # Previous is bullish
                  current['open'] > prev['close'] and  # Current opens above previous close
                  current['close'] < prev['open'] and  # Current closes below previous open
                  current_body > prev_body):  # Current body engulfs previous
                
                pattern = Pattern(
                    name='Bearish Engulfing',
                    confidence=0.85,
                    start_index=i-1,
                    end_index=i,
                    pattern_type='bearish',
                    description='Bearish engulfing pattern detected'
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_chart_patterns(self, df: pd.DataFrame) -> List[Pattern]:
        """Detect chart patterns like triangles, flags, etc."""
        patterns = []
        
        # Head and shoulders pattern
        hns_patterns = self._find_head_and_shoulders(df)
        patterns.extend(hns_patterns)
        
        # Triangle patterns
        triangle_patterns = self._find_triangle_patterns(df)
        patterns.extend(triangle_patterns)
        
        return patterns
    
    def _find_head_and_shoulders(self, df: pd.DataFrame) -> List[Pattern]:
        """Find head and shoulders patterns."""
        patterns = []
        
        if len(df) < 20:
            return patterns
        
        # Look for potential head and shoulders pattern
        highs = df['high'].rolling(window=5, center=True).max()
        
        for i in range(10, len(df) - 10):
            # Check for three peaks with middle peak higher
            left_peak = highs.iloc[i-10:i-5].max()
            middle_peak = highs.iloc[i-5:i+5].max()
            right_peak = highs.iloc[i+5:i+10].max()
            
            if (middle_peak > left_peak and 
                middle_peak > right_peak and
                abs(left_peak - right_peak) / middle_peak < 0.1):  # Shoulders roughly equal
                
                pattern = Pattern(
                    name='Head and Shoulders',
                    confidence=0.8,
                    start_index=i-10,
                    end_index=i+10,
                    pattern_type='bearish',
                    description='Head and shoulders pattern detected'
                )
                patterns.append(pattern)
        
        return patterns
    
    def _find_triangle_patterns(self, df: pd.DataFrame) -> List[Pattern]:
        """Find triangle patterns."""
        patterns = []
        
        if len(df) < 15:
            return patterns
        
        # Look for converging trend lines
        highs = df['high'].rolling(window=3).max()
        lows = df['low'].rolling(window=3).min()
        
        # Simple triangle detection
        for i in range(10, len(df) - 5):
            high_slope = np.polyfit(range(5), highs.iloc[i-5:i], 1)[0]
            low_slope = np.polyfit(range(5), lows.iloc[i-5:i], 1)[0]
            
            # Symmetrical triangle
            if abs(high_slope + low_slope) < 0.01 and abs(high_slope) > 0.001:
                pattern = Pattern(
                    name='Symmetrical Triangle',
                    confidence=0.7,
                    start_index=i-5,
                    end_index=i,
                    pattern_type='neutral',
                    description='Symmetrical triangle pattern detected'
                )
                patterns.append(pattern)
            
            # Ascending triangle
            elif high_slope < -0.001 and abs(low_slope) < 0.001:
                pattern = Pattern(
                    name='Ascending Triangle',
                    confidence=0.75,
                    start_index=i-5,
                    end_index=i,
                    pattern_type='bullish',
                    description='Ascending triangle pattern detected'
                )
                patterns.append(pattern)
            
            # Descending triangle
            elif low_slope > 0.001 and abs(high_slope) < 0.001:
                pattern = Pattern(
                    name='Descending Triangle',
                    confidence=0.75,
                    start_index=i-5,
                    end_index=i,
                    pattern_type='bearish',
                    description='Descending triangle pattern detected'
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_support_resistance(self, df: pd.DataFrame) -> List[Pattern]:
        """Detect support and resistance levels."""
        patterns = []
        
        if len(df) < 10:
            return patterns
        
        # Find support levels
        support_levels = self._find_support_levels(df)
        for level in support_levels:
            pattern = Pattern(
                name='Support Level',
                confidence=0.8,
                start_index=level['index'],
                end_index=level['index'],
                pattern_type='bullish',
                description=f'Support level at {level["price"]:.2f}'
            )
            patterns.append(pattern)
        
        # Find resistance levels
        resistance_levels = self._find_resistance_levels(df)
        for level in resistance_levels:
            pattern = Pattern(
                name='Resistance Level',
                confidence=0.8,
                start_index=level['index'],
                end_index=level['index'],
                pattern_type='bearish',
                description=f'Resistance level at {level["price"]:.2f}'
            )
            patterns.append(pattern)
        
        return patterns
    
    def _find_support_levels(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find support levels."""
        levels = []
        
        for i in range(2, len(df) - 2):
            current_low = df.iloc[i]['low']
            prev_low = df.iloc[i-1]['low']
            next_low = df.iloc[i+1]['low']
            
            # Support: current low is lower than surrounding lows
            if current_low < prev_low and current_low < next_low:
                levels.append({
                    'index': i,
                    'price': current_low,
                    'strength': 1.0
                })
        
        return levels
    
    def _find_resistance_levels(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find resistance levels."""
        levels = []
        
        for i in range(2, len(df) - 2):
            current_high = df.iloc[i]['high']
            prev_high = df.iloc[i-1]['high']
            next_high = df.iloc[i+1]['high']
            
            # Resistance: current high is higher than surrounding highs
            if current_high > prev_high and current_high > next_high:
                levels.append({
                    'index': i,
                    'price': current_high,
                    'strength': 1.0
                })
        
        return levels
    
    def _detect_trend_patterns(self, df: pd.DataFrame) -> List[Pattern]:
        """Detect trend patterns."""
        patterns = []
        
        if len(df) < 20:
            return patterns
        
        # Detect trend direction
        trend = self._calculate_trend(df)
        
        if trend['direction'] == 'uptrend':
            pattern = Pattern(
                name='Uptrend',
                confidence=trend['strength'],
                start_index=0,
                end_index=len(df) - 1,
                pattern_type='bullish',
                description=f'Uptrend with {trend["strength"]:.2f} confidence'
            )
            patterns.append(pattern)
        elif trend['direction'] == 'downtrend':
            pattern = Pattern(
                name='Downtrend',
                confidence=trend['strength'],
                start_index=0,
                end_index=len(df) - 1,
                pattern_type='bearish',
                description=f'Downtrend with {trend["strength"]:.2f} confidence'
            )
            patterns.append(pattern)
        
        return patterns
    
    def _calculate_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trend direction and strength."""
        if len(df) < 10:
            return {'direction': 'neutral', 'strength': 0.0}
        
        # Calculate moving averages
        ma_short = df['close'].rolling(window=5).mean()
        ma_long = df['close'].rolling(window=20).mean()
        
        # Determine trend
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
            'strength': strength
        }


# Factory function
def create_pattern_recognizer(config: Optional[Dict[str, Any]] = None) -> PatternRecognizer:
    """Create a pattern recognizer."""
    return PatternRecognizer(config) 