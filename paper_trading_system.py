#!/usr/bin/env python3
"""
Institutional Paper Trading System

EXPLORATORY: paper-trading loop on synthetic data. Not validated. Not on the proven ablation path.
Demonstrates the manual-to-automated ceiling described in the README.

Continuous 24/7 paper trading with:
- Real-time signal generation
- Model training and adaptation
- Telegram notifications
- Performance tracking
- Risk management
"""

import asyncio
import time
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np
import pandas as pd

# Apply debug fixes
from debug_fixes import setup_unicode_logging, debug_trading_system
setup_unicode_logging()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from trading import SimplifiedInstitutionalSignalGenerator
from utils.telegram_notifier import TelegramNotifier
from exchanges.exchange_config import get_exchange_config

class InstitutionalPaperTradingSystem:
    """
    Institutional-grade paper trading system with continuous operation
    """
    
    def __init__(self, 
                 symbol: str = "SOL",
                 interval_minutes: int = 5):
        """
        Initialize paper trading system
        
        Args:
            symbol: Trading symbol
            interval_minutes: Signal generation interval
        """
        self.symbol = symbol
        self.interval_minutes = interval_minutes
        self.is_running = False
        
        # Initialize signal generator
        self.signal_generator = SimplifiedInstitutionalSignalGenerator()
        
        # Load Telegram credentials securely
        config = get_exchange_config()
        telegram_config = config.get_telegram_config()
        if telegram_config and telegram_config.bot_token and telegram_config.chat_id:
            self.telegram = TelegramNotifier(telegram_config.bot_token, telegram_config.chat_id)
            self.telegram_enabled = True
        else:
            self.telegram = None
            self.telegram_enabled = False
            print("[WARNING] Telegram credentials not found in secure config. Notifications are disabled.")
        
        # Trading state
        self.position = 0  # 0 = no position, 1 = long, -1 = short
        self.entry_price = 0.0
        self.entry_time = None
        self.paper_balance = 10000.0  # Starting balance
        self.trade_history = []
        
        # Performance tracking
        self.total_signals = 0
        self.correct_signals = 0
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        fh = logging.FileHandler(f'paper_trading_{symbol}_{datetime.now().strftime("%Y%m%d")}.log')
        fh.setLevel(logging.INFO)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        self.logger.info("Institutional Paper Trading System initialized")
    
    async def initialize_system(self):
        """Initialize the trading system with historical data"""
        try:
            # Generate historical data for initialization
            historical_data = self._generate_historical_data(500)
            
            # Initialize signal generator
            performance = await self.signal_generator.initialize_system(historical_data, self.symbol)
            
            self.logger.info(f"[SUCCESS] System initialized with {performance.get('feature_count', 0)} features")
            
            # Send initialization notification
            if self.telegram_enabled:
                self.telegram.send_system_status(self.signal_generator.get_system_status())
                self.telegram.send_performance_report(performance)
            
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] System initialization failed: {e}")
            return False
    
    async def start_trading(self):
        """Start continuous paper trading"""
        if not await self.initialize_system():
            self.logger.error("[ERROR] Failed to initialize system")
            return
        
        self.is_running = True
        self.logger.info(f"[STARTED] Starting paper trading for {self.symbol}")
        
        if self.telegram_enabled:
            self.telegram.send_message(f"INSTITUTIONAL PAPER TRADING STARTED\n\nSymbol: {self.symbol}\nInterval: {self.interval_minutes} minutes\nStarting Balance: ${self.paper_balance:,.2f}")
        
        try:
            while self.is_running:
                await self._trading_cycle()
                await asyncio.sleep(self.interval_minutes * 60)  # Wait for next cycle
                
        except KeyboardInterrupt:
            self.logger.info("[STOPPED] Paper trading stopped by user")
            await self._shutdown()
        except Exception as e:
            self.logger.error(f"[ERROR] Trading error: {e}")
            await self._shutdown()
    
    async def _trading_cycle(self):
        """Execute one trading cycle"""
        try:
            # Generate current market data
            current_data = self._generate_current_data()
            
            # Generate trading signal
            signal = await self.signal_generator.generate_trading_signals(current_data, self.symbol)
            
            # Process signal
            await self._process_signal(signal, current_data)
            
            # Update performance metrics
            self._update_performance_metrics()
            
            # Retrain models periodically (every 24 hours)
            if self.total_signals % 288 == 0:  # 288 = 24 hours / 5 minutes
                await self._retrain_models()
            
            self.total_signals += 1
            
        except Exception as e:
            self.logger.error(f"[ERROR] Trading cycle error: {e}")
    
    async def _process_signal(self, signal: Dict, market_data: pd.DataFrame):
        """Process trading signal and execute paper trade"""
        try:
            current_price = market_data['close'].iloc[-1]
            signal_strength = signal['strength']
            signal_confidence = signal['confidence']
            
            # Log signal
            self.logger.info(f"Signal: {signal['signal'].upper()} {self.symbol} | Strength: {signal_strength:.3f} | Confidence: {signal_confidence:.3f}")
            
            # Send Telegram notification for strong signals
            if signal_strength > 0.7 and signal_confidence > 0.6:
                if self.telegram_enabled:
                    self.telegram.send_signal_notification(signal['signal'], self.symbol, signal_strength, signal_confidence)
            
            # Execute paper trade based on signal
            if signal['signal'] == 'buy' and self.position <= 0:
                await self._execute_buy(current_price, signal)
            elif signal['signal'] == 'sell' and self.position >= 0:
                await self._execute_sell(current_price, signal)
            
        except Exception as e:
            self.logger.error(f"[ERROR] Signal processing error: {e}")
    
    async def _execute_buy(self, price: float, signal: Dict):
        """Execute buy order"""
        try:
            # Close short position if exists
            if self.position < 0:
                pnl = (self.entry_price - price) / self.entry_price
                self.paper_balance *= (1 + pnl)
                self._record_trade('close_short', self.entry_price, price, pnl, signal)
            
            # Open long position
            self.position = 1
            self.entry_price = price
            self.entry_time = datetime.now()
            
            self.logger.info(f"[BUY] BUY {self.symbol} @ ${price:.2f} | Balance: ${self.paper_balance:,.2f}")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Buy execution error: {e}")
    
    async def _execute_sell(self, price: float, signal: Dict):
        """Execute sell order"""
        try:
            # Close long position if exists
            if self.position > 0:
                pnl = (price - self.entry_price) / self.entry_price
                self.paper_balance *= (1 + pnl)
                self._record_trade('close_long', self.entry_price, price, pnl, signal)
            
            # Open short position
            self.position = -1
            self.entry_price = price
            self.entry_time = datetime.now()
            
            self.logger.info(f"[SELL] SELL {self.symbol} @ ${price:.2f} | Balance: ${self.paper_balance:,.2f}")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Sell execution error: {e}")
    
    def _record_trade(self, trade_type: str, entry_price: float, exit_price: float, pnl: float, signal: Dict):
        """Record trade for performance tracking"""
        trade = {
            'timestamp': datetime.now(),
            'type': trade_type,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'balance': self.paper_balance,
            'signal_strength': signal['strength'],
            'signal_confidence': signal['confidence']
        }
        
        self.trade_history.append(trade)
        self.performance_metrics['total_trades'] += 1
        
        if pnl > 0:
            self.performance_metrics['winning_trades'] += 1
        
        self.performance_metrics['total_pnl'] += pnl
        
        self.logger.info(f"[CHART] Trade: {trade_type} | PnL: {pnl:.2%} | Balance: ${self.paper_balance:,.2f}")
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        if len(self.trade_history) > 0:
            # Calculate max drawdown
            balances = [trade['balance'] for trade in self.trade_history]
            peak = max(balances)
            current_drawdown = (self.paper_balance - peak) / peak
            self.performance_metrics['max_drawdown'] = min(self.performance_metrics['max_drawdown'], current_drawdown)
            
            # Calculate win rate
            win_rate = self.performance_metrics['winning_trades'] / self.performance_metrics['total_trades']
            
            # Log performance every 50 trades
            if self.performance_metrics['total_trades'] % 50 == 0:
                self.logger.info(f"[PERFORMANCE] Performance Update | Trades: {self.performance_metrics['total_trades']} | Win Rate: {win_rate:.2%} | Balance: ${self.paper_balance:,.2f}")
                
                if self.telegram_enabled:
                    self.telegram.send_message(f"[PERFORMANCE] **PERFORMANCE UPDATE**\n\n[BALANCE] Balance: ${self.paper_balance:,.2f}\n[CHART] Total Trades: {self.performance_metrics['total_trades']}\n[TARGET] Win Rate: {win_rate:.2%}\n[DRAWDOWN] Max Drawdown: {self.performance_metrics['max_drawdown']:.2%}")
    
    async def _retrain_models(self):
        """Retrain models with new data"""
        try:
            self.logger.info("RETRAIN: Retraining models...")
            
            # Generate new historical data
            new_data = self._generate_historical_data(1000)
            
            # Retrain signal generator
            performance = await self.signal_generator.initialize_system(new_data, self.symbol)
            
            self.logger.info(f"[SUCCESS] Models retrained with {performance.get('feature_count', 0)} features")
            
            if self.telegram_enabled:
                self.telegram.send_performance_report(performance)
                
        except Exception as e:
            self.logger.error(f"[ERROR] Model retraining error: {e}")
    
    def _generate_historical_data(self, n_samples: int) -> pd.DataFrame:
        """Generate realistic historical data"""
        np.random.seed(int(time.time()) % 10000)  # Dynamic seed
        
        base_price = 100.0
        base_volume = 1000000
        
        # Generate price series
        price_changes = np.random.normal(0, 0.02, n_samples)
        prices = [base_price]
        
        for i in range(1, n_samples):
            trend = 0.0001 * i
            mean_reversion = -0.001 * (prices[-1] - base_price) / base_price
            new_price = prices[-1] * (1 + price_changes[i] + trend + mean_reversion)
            prices.append(max(new_price, 1.0))
        
        # Generate volume with anomalies
        volumes = []
        for i in range(n_samples):
            base_vol = base_volume * (1 + 0.5 * np.random.normal(0, 1))
            if np.random.random() < 0.05:
                base_vol *= np.random.uniform(2, 5)
            volumes.append(max(base_vol, 100000))
        
        # Create timestamps
        start_time = datetime.now() - timedelta(minutes=5 * n_samples)
        timestamps = [start_time + timedelta(minutes=5 * i) for i in range(n_samples)]
        
        # Create OHLCV data
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': volumes
        })
        
        # Ensure high/low are properly ordered
        data['high'] = data[['open', 'close', 'high']].max(axis=1)
        data['low'] = data[['open', 'close', 'low']].min(axis=1)
        
        return data
    
    def _generate_current_data(self) -> pd.DataFrame:
        """Generate current market data"""
        return self._generate_historical_data(50).tail(50)
    
    async def _shutdown(self):
        """Graceful shutdown"""
        self.is_running = False
        
        # Close any open positions
        if self.position != 0:
            current_data = self._generate_current_data()
            current_price = current_data['close'].iloc[-1]
            
            if self.position > 0:
                pnl = (current_price - self.entry_price) / self.entry_price
                self.paper_balance *= (1 + pnl)
            else:
                pnl = (self.entry_price - current_price) / self.entry_price
                self.paper_balance *= (1 + pnl)
            
            self.logger.info(f"[ENDED] Closing position | Final PnL: {pnl:.2%} | Final Balance: ${self.paper_balance:,.2f}")
        
        # Send final performance report
        if self.telegram_enabled:
            final_message = f"[ENDED] **TRADING SESSION ENDED**\n\n[BALANCE] Final Balance: ${self.paper_balance:,.2f}\n[CHART] Total Trades: {self.performance_metrics['total_trades']}\n[TARGET] Win Rate: {self.performance_metrics['winning_trades'] / max(self.performance_metrics['total_trades'], 1):.2%}\n[DRAWDOWN] Max Drawdown: {self.performance_metrics['max_drawdown']:.2%}"
            self.telegram.send_message(final_message)
        
        self.logger.info("[INSTITUTIONAL]  Institutional Paper Trading System shutdown complete")

# Main execution
async def main():
    """Main function to run the paper trading system"""
    
    # Apply debug wrapper
    if not debug_trading_system():
        print("[ERROR] Debug initialization failed")
        return
    
    # Configuration
    SYMBOL = "SOL"
    INTERVAL_MINUTES = 5
    
    # Telegram configuration (optional)
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')  # Set in environment variables
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')      # Set in environment variables
    
    print("[INSTITUTIONAL]  Institutional Paper Trading System")
    print("=" * 50)
    print(f"[CHART] Symbol: {SYMBOL}")
    print(f"[TIME] Interval: {INTERVAL_MINUTES} minutes")
    print(f"[TELEGRAM] Telegram: {'[SUCCESS] Enabled' if TELEGRAM_BOT_TOKEN else '[ERROR] Disabled'}")
    print("=" * 50)
    
    # Create and start trading system
    trading_system = InstitutionalPaperTradingSystem(
        symbol=SYMBOL,
        interval_minutes=INTERVAL_MINUTES
    )
    
    await trading_system.start_trading()

if __name__ == "__main__":
    asyncio.run(main()) 