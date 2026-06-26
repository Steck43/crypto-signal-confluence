"""
Macro Regime Detector for Economic Data Analysis

EXPLORATORY: macro regime research scaffold. Not on the validated ablation path. Not validated.

Uses FRED-shaped economic inputs to estimate risk-on/off regime.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class MacroRegimeDetector:
    def __init__(self):
        self.regime_thresholds = {
            'risk_on': 0.6,      # Above this = risk-on environment
            'risk_off': -0.6,    # Below this = risk-off environment
            'neutral': 0.2       # Between thresholds = neutral
        }
        
        # FRED API series IDs for key economic indicators
        self.fred_series = {
            'federal_funds_rate': 'FEDFUNDS',      # Federal Funds Rate
            'inflation_rate': 'CPIAUCSL',          # Consumer Price Index
            'unemployment_rate': 'UNRATE',         # Unemployment Rate
            'gdp_growth': 'GDP',                   # Gross Domestic Product
            'vix': 'VIXCLS'                        # VIX Volatility Index
        }
        
        # Economic indicator weights for regime calculation
        self.indicator_weights = {
            'federal_funds_rate': 0.25,    # Monetary policy impact
            'inflation_rate': 0.20,        # Inflation pressure
            'unemployment_rate': 0.20,     # Economic health
            'gdp_growth': 0.20,            # Economic growth
            'vix': 0.15                    # Market volatility
        }
        
    def analyze_macro_regime(self, economic_data: Dict) -> Dict:
        """
        Analyze economic data to determine market regime
        Returns: {'regime': 'risk_on'/'risk_off'/'neutral', 'score': float, 'confidence': float}
        """
        try:
            if not economic_data:
                return self._default_regime()
            
            # Calculate individual indicator scores
            indicator_scores = {}
            
            # Federal Funds Rate analysis (lower = more risk-on)
            if 'federal_funds_rate' in economic_data:
                ffr_score = self._analyze_federal_funds_rate(economic_data['federal_funds_rate'])
                indicator_scores['federal_funds_rate'] = ffr_score
            
            # Inflation Rate analysis (moderate inflation = risk-on, high = risk-off)
            if 'inflation_rate' in economic_data:
                inflation_score = self._analyze_inflation_rate(economic_data['inflation_rate'])
                indicator_scores['inflation_rate'] = inflation_score
            
            # Unemployment Rate analysis (lower = more risk-on)
            if 'unemployment_rate' in economic_data:
                unemployment_score = self._analyze_unemployment_rate(economic_data['unemployment_rate'])
                indicator_scores['unemployment_rate'] = unemployment_score
            
            # GDP Growth analysis (higher = more risk-on)
            if 'gdp_growth' in economic_data:
                gdp_score = self._analyze_gdp_growth(economic_data['gdp_growth'])
                indicator_scores['gdp_growth'] = gdp_score
            
            # VIX analysis (lower = more risk-on)
            if 'vix' in economic_data:
                vix_score = self._analyze_vix(economic_data['vix'])
                indicator_scores['vix'] = vix_score
            
            # Calculate weighted regime score
            regime_score = self._calculate_regime_score(indicator_scores)
            
            # Determine regime
            regime = self._determine_regime(regime_score)
            
            # Calculate confidence based on data availability
            confidence = self._calculate_confidence(indicator_scores)
            
            return {
                'regime': regime,
                'score': regime_score,
                'confidence': confidence,
                'indicator_scores': indicator_scores,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Macro regime analysis error: {e}")
            return self._default_regime()
    
    def _analyze_federal_funds_rate(self, ffr_data: Dict) -> float:
        """Analyze Federal Funds Rate (lower = more risk-on)"""
        try:
            current_rate = ffr_data.get('current', 5.0)  # Default to 5%
            historical_avg = ffr_data.get('historical_avg', 3.0)
            
            # Score based on deviation from historical average
            # Lower rates = more risk-on (positive score)
            deviation = historical_avg - current_rate
            score = np.clip(deviation / 2.0, -1.0, 1.0)  # Normalize to [-1, 1]
            
            return score
            
        except Exception as e:
            print(f"FFR analysis error: {e}")
            return 0.0
    
    def _analyze_inflation_rate(self, inflation_data: Dict) -> float:
        """Analyze inflation rate (moderate = risk-on, high = risk-off)"""
        try:
            current_inflation = inflation_data.get('current', 3.0)  # Default to 3%
            
            # Optimal inflation range for crypto: 2-4%
            if 2.0 <= current_inflation <= 4.0:
                score = 0.5  # Moderate inflation = good for crypto
            elif current_inflation < 2.0:
                score = 0.0  # Low inflation = neutral
            elif current_inflation > 6.0:
                score = -0.8  # High inflation = risk-off
            else:
                score = -0.3  # Elevated inflation = slightly risk-off
            
            return score
            
        except Exception as e:
            print(f"Inflation analysis error: {e}")
            return 0.0
    
    def _analyze_unemployment_rate(self, unemployment_data: Dict) -> float:
        """Analyze unemployment rate (lower = more risk-on)"""
        try:
            current_rate = unemployment_data.get('current', 4.0)  # Default to 4%
            
            # Score based on unemployment level
            if current_rate <= 3.5:
                score = 0.8  # Very low unemployment = risk-on
            elif current_rate <= 4.5:
                score = 0.4  # Low unemployment = somewhat risk-on
            elif current_rate <= 6.0:
                score = 0.0  # Moderate unemployment = neutral
            elif current_rate <= 8.0:
                score = -0.4  # High unemployment = risk-off
            else:
                score = -0.8  # Very high unemployment = very risk-off
            
            return score
            
        except Exception as e:
            print(f"Unemployment analysis error: {e}")
            return 0.0
    
    def _analyze_gdp_growth(self, gdp_data: Dict) -> float:
        """Analyze GDP growth (higher = more risk-on)"""
        try:
            current_growth = gdp_data.get('current', 2.0)  # Default to 2%
            
            # Score based on GDP growth rate
            if current_growth >= 3.0:
                score = 0.8  # Strong growth = risk-on
            elif current_growth >= 2.0:
                score = 0.4  # Moderate growth = somewhat risk-on
            elif current_growth >= 1.0:
                score = 0.0  # Slow growth = neutral
            elif current_growth >= 0.0:
                score = -0.4  # Stagnant = risk-off
            else:
                score = -0.8  # Recession = very risk-off
            
            return score
            
        except Exception as e:
            print(f"GDP analysis error: {e}")
            return 0.0
    
    def _analyze_vix(self, vix_data: Dict) -> float:
        """Analyze VIX volatility index (lower = more risk-on)"""
        try:
            current_vix = vix_data.get('current', 20.0)  # Default to 20
            
            # Score based on VIX level
            if current_vix <= 15:
                score = 0.6  # Low volatility = risk-on
            elif current_vix <= 25:
                score = 0.0  # Normal volatility = neutral
            elif current_vix <= 35:
                score = -0.4  # High volatility = risk-off
            else:
                score = -0.8  # Very high volatility = very risk-off
            
            return score
            
        except Exception as e:
            print(f"VIX analysis error: {e}")
            return 0.0
    
    def _calculate_regime_score(self, indicator_scores: Dict) -> float:
        """Calculate weighted regime score from indicator scores"""
        total_score = 0.0
        total_weight = 0.0
        
        for indicator, score in indicator_scores.items():
            weight = self.indicator_weights.get(indicator, 0.0)
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_regime(self, score: float) -> str:
        """Determine market regime based on score"""
        if score >= self.regime_thresholds['risk_on']:
            return 'risk_on'
        elif score <= self.regime_thresholds['risk_off']:
            return 'risk_off'
        else:
            return 'neutral'
    
    def _calculate_confidence(self, indicator_scores: Dict) -> float:
        """Calculate confidence based on data availability"""
        available_indicators = len(indicator_scores)
        total_indicators = len(self.indicator_weights)
        
        return min(1.0, available_indicators / total_indicators)
    
    def _default_regime(self) -> Dict:
        """Return default regime when no data available"""
        return {
            'regime': 'neutral',
            'score': 0.0,
            'confidence': 0.0,
            'indicator_scores': {},
            'timestamp': datetime.now()
        }
    
    def get_position_adjustment(self, regime_analysis: Dict) -> float:
        """
        Get position size adjustment factor based on regime
        Returns: 0.0-1.0 (0 = no positions, 1 = full positions)
        """
        regime = regime_analysis.get('regime', 'neutral')
        confidence = regime_analysis.get('confidence', 0.0)
        
        # Base adjustments by regime
        base_adjustments = {
            'risk_on': 1.0,      # Full positions
            'neutral': 0.7,      # Reduced positions
            'risk_off': 0.3      # Minimal positions
        }
        
        base_adjustment = base_adjustments.get(regime, 0.7)
        
        # Adjust by confidence (lower confidence = more conservative)
        confidence_adjustment = 0.5 + (confidence * 0.5)  # 0.5-1.0 range
        
        final_adjustment = base_adjustment * confidence_adjustment
        
        return np.clip(final_adjustment, 0.0, 1.0) 