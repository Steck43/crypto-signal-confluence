"""
Main trading bot that orchestrates the trading system
"""

import asyncio
import time
from datetime import datetime
from trading.signal_generator import SignalGenerator
from trading.risk_manager import RiskManager
from trading.order_executor import OrderExecutor

class TradingBot:
    def __init__(self, initial_capital=500):
        self.signal_generator = SignalGenerator()
        self.risk_manager = RiskManager(initial_capital)
        self.order_executor = OrderExecutor()
        self.running = False
        self.trading_pairs = ["SOL/USDT", "BTC/USDT", "ETH/USDT", "HYPE"]
        
    async def initialize(self):
        """Initialize all components"""
        print("Initializing institutional trading bot...")
        
        # Initialize the signal generator with historical data
        historical_data = await self._get_historical_data()
        await self.signal_generator.initialize_system(historical_data, "SOL")
        
        await self.order_executor.initialize()
        print("Institutional trading bot initialized successfully")
        
    async def _get_historical_data(self):
        """Get historical data for system initialization"""
        import pandas as pd
        import numpy as np
        
        # Generate 1000 historical data points for initialization
        base_price = 100.0
        prices = [base_price]
        for i in range(999):
            change = np.random.normal(0, 0.015)  # 1.5% daily volatility
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        volumes = [1000000 + np.random.normal(0, 300000) for _ in range(1000)]
        
        return pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.008))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.008))) for p in prices],
            'close': prices,
            'volume': volumes
        })
        
    async def start_trading(self, interval_seconds=60):
        """Start the trading loop"""
        self.running = True
        print(f"Starting trading bot with {interval_seconds}s intervals")
        
        while self.running:
            try:
                await self._trading_cycle()
                await asyncio.sleep(interval_seconds)
            except KeyboardInterrupt:
                print("Stopping trading bot...")
                self.running = False
                break
            except Exception as e:
                print(f"Trading cycle error: {e}")
                await asyncio.sleep(10)  # Wait before retrying
                
    async def stop_trading(self):
        """Stop the trading bot"""
        self.running = False
        print("Trading bot stopped")
        
    async def _trading_cycle(self):
        """Execute one complete trading cycle"""
        print(f"\n--- Trading Cycle {datetime.now()} ---")
        
        # Get economic data for macro regime analysis
        economic_data = await self._get_economic_data()
        
        for symbol in self.trading_pairs:
            try:
                # 1. Get market data (placeholder)
                market_data = await self._get_market_data(symbol)
                news_data = await self._get_news_data(symbol)
                
                # 2. Generate institutional trading signal
                signal = await self.signal_generator.generate_institutional_signals(market_data, news_data, symbol)
                print(f"{symbol}: {signal['signal']} (strength: {signal['strength']:.2f}, confidence: {signal['confidence']:.2f})")
                print(f"  Market Regime: {signal['metadata']['market_regime']}")
                print(f"  Risk Score: {signal['risk_metrics'].get('risk_score', 0):.3f}")
                
                # 3. Check if we should trade
                if signal['signal'] in ['buy', 'sell'] and signal['strength'] > 0.5:
                    # 4. Calculate position size with macro regime adjustment
                    position_size = self.risk_manager.calculate_position_size(
                        signal_strength=signal['strength'],
                        economic_data=economic_data
                    )
                    
                    if position_size > 0:
                        # 5. Check risk limits
                        proposed_trade = {
                            'symbol': symbol,
                            'size': position_size,
                            'signal': signal
                        }
                        
                        risk_check = self.risk_manager.check_risk_limits(proposed_trade)
                        
                        if risk_check['approved']:
                            # 6. Execute trade
                            print(f"Executing {signal['signal']} order for {symbol}: ${position_size:.2f}")
                            result = await self.order_executor.execute_trade(
                                signal, position_size, symbol
                            )
                            
                            if result:
                                print(f"Order executed: {result['order_id']}")
                                # Track position in risk manager
                                self.risk_manager.current_positions.append({
                                    'id': result['order_id'],
                                    'symbol': symbol,
                                    'size': position_size,
                                    'entry_time': datetime.now()
                                })
                            else:
                                print(f"Order execution failed for {symbol}")
                        else:
                            print(f"Risk check failed for {symbol}: {risk_check['checks']}")
                    else:
                        print(f"Position size too small for {symbol}")
                else:
                    print(f"No actionable signal for {symbol}")
                    
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                
    async def _get_market_data(self, symbol):
        """Get market data for symbol (placeholder)"""
        # In real implementation, this would fetch from exchanges
        # Return DataFrame format for institutional signal generator
        import pandas as pd
        import numpy as np
        
        # Generate realistic market data
        base_price = 100.0
        prices = [base_price]
        for i in range(49):  # 50 data points
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        volumes = [1000000 + np.random.normal(0, 200000) for _ in range(50)]
        
        return pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': volumes,
            'symbol': symbol
        })
        
    async def _get_news_data(self, symbol):
        """Get news data for symbol (placeholder)"""
        # In real implementation, this would fetch from RSS feeds
        return [
            {
                'title': f'Positive news about {symbol}',
                'content': f'{symbol} shows bullish momentum',
                'timestamp': datetime.now()
            }
        ]
        
    async def _get_economic_data(self):
        """Get economic data for macro regime analysis (placeholder)"""
        # In real implementation, this would fetch from FRED API
        return {
            'federal_funds_rate': {
                'current': 5.25,  # Current FFR
                'historical_avg': 3.0
            },
            'inflation_rate': {
                'current': 3.2  # Current inflation rate
            },
            'unemployment_rate': {
                'current': 3.9  # Current unemployment rate
            },
            'gdp_growth': {
                'current': 2.1  # Current GDP growth rate
            },
            'vix': {
                'current': 18.5  # Current VIX level
            }
        }
        
    def get_performance_summary(self):
        """Get current trading performance"""
        return {
            'current_capital': self.risk_manager.current_capital,
            'total_return': (self.risk_manager.current_capital - self.risk_manager.initial_capital) / self.risk_manager.initial_capital,
            'active_positions': len(self.risk_manager.current_positions),
            'peak_capital': self.risk_manager.peak_capital,
            'current_drawdown': (self.risk_manager.peak_capital - self.risk_manager.current_capital) / self.risk_manager.peak_capital
        }

# Example usage
async def main():
    bot = TradingBot(initial_capital=1000)
    await bot.initialize()
    
    # Run for a few cycles to demonstrate
    print("Running trading bot for 3 cycles...")
    for i in range(3):
        await bot._trading_cycle()
        await asyncio.sleep(5)  # 5 second intervals for demo
        
    # Show performance
    performance = bot.get_performance_summary()
    print(f"\nPerformance Summary:")
    print(f"Current Capital: ${performance['current_capital']:.2f}")
    print(f"Total Return: {performance['total_return']:.2%}")
    print(f"Active Positions: {performance['active_positions']}")
    print(f"Current Drawdown: {performance['current_drawdown']:.2%}")

if __name__ == "__main__":
    asyncio.run(main()) 