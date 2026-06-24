import numpy as np
from datetime import datetime
from analysis.macro_regime_detector import MacroRegimeDetector

class RiskManager:
    def __init__(self, initial_capital=500):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_positions = 3
        self.max_position_size = 0.15  # 15% of capital
        self.min_position_size = 0.05  # 5% of capital
        self.max_drawdown = 0.15       # 15% max drawdown
        self.current_positions = []
        self.peak_capital = initial_capital
        self.macro_detector = MacroRegimeDetector()
        self.current_regime = {'regime': 'neutral', 'score': 0.0, 'confidence': 0.0}
        
    def calculate_position_size(self, signal_strength, win_rate=0.6, avg_return=0.08, economic_data=None):
        """
        Kelly Criterion position sizing with risk adjustments and macro regime analysis
        Kelly % = (bp - q) / b
        where: b = odds (avg_return), p = win_rate, q = 1-p
        """
        try:
            # Basic Kelly calculation
            if avg_return <= 0 or win_rate <= 0:
                return 0
            
            kelly_fraction = ((avg_return * win_rate) - (1 - win_rate)) / avg_return
            
            # Risk adjustments
            kelly_fraction *= 0.25  # Use quarter Kelly for safety
            kelly_fraction *= signal_strength  # Adjust by signal strength
            kelly_fraction *= self._volatility_adjustment()
            kelly_fraction *= self._consecutive_loss_adjustment()
            
            # Macro regime adjustment
            if economic_data:
                regime_analysis = self.macro_detector.analyze_macro_regime(economic_data)
                self.current_regime = regime_analysis
                regime_adjustment = self.macro_detector.get_position_adjustment(regime_analysis)
                kelly_fraction *= regime_adjustment
                print(f"Macro regime: {regime_analysis['regime']} (adjustment: {regime_adjustment:.2f})")
            
            # Apply position size limits
            kelly_fraction = max(kelly_fraction, 0)  # No negative positions
            kelly_fraction = min(kelly_fraction, self.max_position_size)
            kelly_fraction = max(kelly_fraction, self.min_position_size) if kelly_fraction > 0 else 0
            
            # Check maximum positions limit
            if len(self.current_positions) >= self.max_positions and kelly_fraction > 0:
                return 0
            
            # Check drawdown protection
            current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
            if current_drawdown > self.max_drawdown:
                return 0  # Stop trading during high drawdown
            
            position_size = kelly_fraction * self.current_capital
            
            return position_size
            
        except Exception as e:
            print(f"Position sizing error: {e}")
            return 0
    
    def check_risk_limits(self, proposed_trade):
        """Validate trade against all risk parameters"""
        checks = {
            'position_limit': len(self.current_positions) < self.max_positions,
            'position_size': proposed_trade['size'] <= self.max_position_size * self.current_capital,
            'sufficient_capital': proposed_trade['size'] <= self.current_capital * 0.95,
            'drawdown_ok': self._check_drawdown_limit(),
            'not_weekend': self._check_trading_hours(),
        }
        
        all_passed = all(checks.values())
        
        return {
            'approved': all_passed,
            'checks': checks,
            'risk_score': self._calculate_risk_score(proposed_trade)
        }
    
    def calculate_stop_loss(self, entry_price, position_side, volatility=0.02):
        """Calculate dynamic stop loss based on volatility"""
        if position_side == 'buy':
            stop_loss = entry_price * (1 - (2 * volatility))  # 2x volatility stop
        else:  # sell
            stop_loss = entry_price * (1 + (2 * volatility))
        
        return stop_loss
    
    def calculate_take_profit(self, entry_price, position_side, volatility=0.02):
        """Calculate take profit at 3:1 risk/reward ratio"""
        stop_distance = abs(entry_price - self.calculate_stop_loss(entry_price, position_side, volatility))
        
        if position_side == 'buy':
            take_profit = entry_price + (3 * stop_distance)  # 3:1 R/R
        else:  # sell
            take_profit = entry_price - (3 * stop_distance)
        
        return take_profit
    
    def update_capital(self, trade_result):
        """Update capital after trade completion"""
        self.current_capital += trade_result['pnl']
        self.peak_capital = max(self.peak_capital, self.current_capital)
        
        # Remove completed trade from positions
        self.current_positions = [p for p in self.current_positions if p['id'] != trade_result['trade_id']]
        
        return {
            'new_capital': self.current_capital,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'current_drawdown': (self.peak_capital - self.current_capital) / self.peak_capital
        }
    
    def _volatility_adjustment(self):
        """Reduce position size during high volatility"""
        # Placeholder - in real implementation, calculate recent volatility
        return 1.0  # No adjustment for now
    
    def _consecutive_loss_adjustment(self):
        """Reduce position size after consecutive losses"""
        # Placeholder - track recent trade outcomes
        return 1.0  # No adjustment for now
    
    def _check_drawdown_limit(self):
        """Check if current drawdown is within limits"""
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        return current_drawdown <= self.max_drawdown
    
    def _check_trading_hours(self):
        """Check if it's appropriate time to trade (crypto trades 24/7)"""
        return True  # Crypto markets are always open
    
    def _calculate_risk_score(self, trade):
        """Calculate overall risk score for the trade"""
        # Simple risk scoring based on position size and current exposure
        position_risk = trade['size'] / self.current_capital
        exposure_risk = len(self.current_positions) / self.max_positions
        
        risk_score = (position_risk + exposure_risk) / 2
        return min(risk_score, 1.0) 