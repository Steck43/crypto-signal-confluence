"""
Advanced Mathematical Utilities for Institutional Trading
Provides sophisticated mathematical functions for trading systems including:
- Risk metrics (VaR, CVaR, Sharpe, Sortino)
- Statistical analysis (correlation, cointegration, stationarity)
- Signal processing (filters, smoothing, noise reduction)
- Portfolio optimization (Kelly criterion, mean reversion)
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional
from scipy import stats
from scipy.optimize import minimize
import warnings

class RiskMetrics:
    """Calculate institutional-grade risk metrics"""
    
    @staticmethod
    def value_at_risk(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (0.05 = 95% VaR)
            
        Returns:
            VaR value
        """
        if len(returns) < 30:
            warnings.warn("Insufficient data for reliable VaR calculation")
            
        return np.percentile(returns.dropna(), confidence_level * 100)
    
    @staticmethod
    def conditional_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall)
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level
            
        Returns:
            CVaR value
        """
        var = RiskMetrics.value_at_risk(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe Ratio
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sharpe ratio
        """
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino Ratio (downside deviation)
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sortino ratio
        """
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            return np.inf
            
        return excess_returns.mean() / downside_std * np.sqrt(252)
    
    @staticmethod
    def maximum_drawdown(prices: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        Calculate Maximum Drawdown
        
        Args:
            prices: Series of prices
            
        Returns:
            Tuple of (max_drawdown, start_date, end_date)
        """
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        
        max_dd = drawdown.min()
        end_date = drawdown.idxmin()
        start_date = peak[:end_date].idxmax()
        
        return max_dd, start_date, end_date
    
    @staticmethod
    def calmar_ratio(returns: pd.Series) -> float:
        """
        Calculate Calmar Ratio (Annual Return / Max Drawdown)
        
        Args:
            returns: Series of returns
            
        Returns:
            Calmar ratio
        """
        annual_return = returns.mean() * 252
        prices = (1 + returns).cumprod()
        max_dd, _, _ = RiskMetrics.maximum_drawdown(prices)
        
        if max_dd == 0:
            return np.inf
            
        return annual_return / abs(max_dd)


class StatisticalAnalysis:
    """Advanced statistical analysis for trading"""
    
    @staticmethod
    def rolling_correlation(x: pd.Series, y: pd.Series, window: int = 30) -> pd.Series:
        """Calculate rolling correlation between two series"""
        return x.rolling(window).corr(y)
    
    @staticmethod
    def cointegration_test(x: pd.Series, y: pd.Series) -> Tuple[float, float, bool]:
        """
        Test for cointegration between two price series
        
        Returns:
            Tuple of (test_statistic, p_value, is_cointegrated)
        """
        try:
            from statsmodels.tsa.stattools import coint
            
            score, p_value, _ = coint(x.dropna(), y.dropna())
            is_cointegrated = p_value < 0.05
            
            return score, p_value, is_cointegrated
        except ImportError:
            warnings.warn("statsmodels not available, skipping cointegration test")
            return 0.0, 1.0, False
    
    @staticmethod
    def adf_test(series: pd.Series) -> Tuple[float, float, bool]:
        """
        Augmented Dickey-Fuller test for stationarity
        
        Returns:
            Tuple of (test_statistic, p_value, is_stationary)
        """
        try:
            from statsmodels.tsa.stattools import adfuller
            
            result = adfuller(series.dropna())
            is_stationary = result[1] < 0.05
            
            return result[0], result[1], is_stationary
        except ImportError:
            warnings.warn("statsmodels not available, skipping ADF test")
            return 0.0, 1.0, False
    
    @staticmethod
    def z_score(series: pd.Series, window: int = 30) -> pd.Series:
        """Calculate rolling z-score"""
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        return (series - rolling_mean) / rolling_std
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle_band = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        
        return upper_band, middle_band, lower_band


class SignalProcessing:
    """Signal processing utilities for trading data"""
    
    @staticmethod
    def exponential_smoothing(series: pd.Series, alpha: float = 0.3) -> pd.Series:
        """Apply exponential smoothing to reduce noise"""
        return series.ewm(alpha=alpha).mean()
    
    @staticmethod
    def kalman_filter(prices: pd.Series, process_variance: float = 1e-4, 
                     measurement_variance: float = 0.1) -> pd.Series:
        """
        Apply Kalman filter for noise reduction
        
        Args:
            prices: Price series
            process_variance: Process noise variance
            measurement_variance: Measurement noise variance
            
        Returns:
            Filtered price series
        """
        n = len(prices)
        filtered_prices = np.zeros(n)
        
        # Initialize
        x = prices.iloc[0]  # Initial state
        P = 1.0  # Initial uncertainty
        
        for i in range(n):
            # Prediction
            x_pred = x
            P_pred = P + process_variance
            
            # Update
            K = P_pred / (P_pred + measurement_variance)
            x = x_pred + K * (prices.iloc[i] - x_pred)
            P = (1 - K) * P_pred
            
            filtered_prices[i] = x
        
        return pd.Series(filtered_prices, index=prices.index)
    
    @staticmethod
    def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD indicator
        
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram


class PortfolioOptimization:
    """Portfolio optimization and position sizing"""
    
    @staticmethod
    def kelly_criterion(returns: pd.Series, bankroll: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion
        
        Args:
            returns: Series of trade returns
            bankroll: Current bankroll
            
        Returns:
            Optimal position size
        """
        if len(returns) < 10:
            warnings.warn("Insufficient data for Kelly criterion")
            return 0.01  # Conservative 1% position
        
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean()
        avg_loss = abs(returns[returns < 0].mean())
        
        if avg_loss == 0:
            return 0.25  # Max 25% if no losses recorded
        
        win_loss_ratio = avg_win / avg_loss
        kelly_percentage = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Apply fractional Kelly (25% of full Kelly for risk management)
        fractional_kelly = max(0, min(0.25, kelly_percentage * 0.25))
        
        return fractional_kelly
    
    @staticmethod
    def position_size_volatility_adjusted(price: float, volatility: float, 
                                        risk_per_trade: float, bankroll: float) -> int:
        """
        Calculate position size based on volatility targeting
        
        Args:
            price: Current price
            volatility: Historical volatility
            risk_per_trade: Risk per trade as percentage of bankroll
            bankroll: Current bankroll
            
        Returns:
            Position size in units
        """
        risk_amount = bankroll * risk_per_trade
        volatility_adjusted_risk = risk_amount / volatility
        position_size = int(volatility_adjusted_risk / price)
        
        return max(1, position_size)  # Minimum 1 unit
    
    @staticmethod
    def correlation_adjusted_position(base_position: float, correlation: float) -> float:
        """
        Adjust position size based on correlation with existing positions
        
        Args:
            base_position: Base position size
            correlation: Correlation with existing positions (-1 to 1)
            
        Returns:
            Adjusted position size
        """
        # Reduce position size for highly correlated assets
        correlation_factor = 1 - abs(correlation) * 0.5
        return base_position * correlation_factor


# Utility functions for easy access
def calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Quick Sharpe ratio calculation"""
    return RiskMetrics.sharpe_ratio(returns, risk_free_rate)

def calculate_max_drawdown(prices: pd.Series) -> float:
    """Quick max drawdown calculation"""
    max_dd, _, _ = RiskMetrics.maximum_drawdown(prices)
    return max_dd

def smooth_price_series(prices: pd.Series, method: str = 'kalman') -> pd.Series:
    """Apply smoothing to price series"""
    if method == 'kalman':
        return SignalProcessing.kalman_filter(prices)
    elif method == 'exponential':
        return SignalProcessing.exponential_smoothing(prices)
    else:
        return prices

def optimal_position_size(returns: pd.Series, bankroll: float, method: str = 'kelly') -> float:
    """Calculate optimal position size"""
    if method == 'kelly':
        return PortfolioOptimization.kelly_criterion(returns, bankroll)
    else:
        return 0.02  # Default 2% risk per trade 