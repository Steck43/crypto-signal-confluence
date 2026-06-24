"""
Multi-Exchange Trading Manager

Orchestrates trading across multiple exchanges:
- Hyperliquid (decentralized)
- Binance (centralized)
- OKX (centralized)

Provides unified interface for:
- Account management
- Market data aggregation
- Order execution
- Arbitrage detection
- Portfolio consolidation
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json

# Import our types and connectors
from .exchange_types import (
    Position, OrderResult, MarketData, AccountInfo, ExchangeConfig,
    OrderSide, OrderType, OrderStatus, PositionSide,
    TradingSignal, ArbitrageOpportunity, ExchangeStatus,
    ExchangeError, AuthenticationError, RateLimitError, ConnectionError
)
from .exchange_connectors import (
    HyperliquidConnector, BinanceConnector, OKXConnector
)

logger = logging.getLogger(__name__)


class MultiExchangeManager:
    """Orchestrates trading across multiple exchanges"""
    
    def __init__(self, exchange_configs: Dict[str, ExchangeConfig]):
        """
        Initialize multi-exchange manager
        
        Args:
            exchange_configs: Configuration for each exchange
        """
        self.exchanges = {}
        self.positions = {}
        self.market_data = {}
        self.arbitrage_opportunities = []
        self.exchange_status = {}
        
        # Initialize exchange connectors
        self._initialize_exchanges(exchange_configs)
        
        logger.info(f"🚀 Multi-exchange manager initialized with {len(self.exchanges)} exchanges")
    
    def _initialize_exchanges(self, configs: Dict[str, ExchangeConfig]):
        """Initialize exchange connectors"""
        for exchange_name, config in configs.items():
            try:
                if exchange_name == "hyperliquid":
                    self.exchanges[exchange_name] = HyperliquidConnector(config)
                elif exchange_name == "binance":
                    self.exchanges[exchange_name] = BinanceConnector(config)
                elif exchange_name == "okx":
                    self.exchanges[exchange_name] = OKXConnector(config)
                else:
                    logger.warning(f"Unknown exchange: {exchange_name}")
                    continue
                
                # Initialize status
                self.exchange_status[exchange_name] = ExchangeStatus(
                    exchange=exchange_name,
                    connected=False,
                    last_heartbeat=None
                )
                
                logger.info(f"✅ {exchange_name} connector initialized")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize {exchange_name}: {e}")
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all exchanges"""
        results = {}
        
        for exchange_name, connector in self.exchanges.items():
            try:
                success = await connector.connect()
                results[exchange_name] = success
                
                if success:
                    self.exchange_status[exchange_name].connected = True
                    self.exchange_status[exchange_name].last_heartbeat = datetime.now()
                    logger.info(f"✅ {exchange_name} connected")
                else:
                    logger.error(f"❌ {exchange_name} connection failed")
                    
            except Exception as e:
                results[exchange_name] = False
                self.exchange_status[exchange_name].error_count += 1
                self.exchange_status[exchange_name].last_error = str(e)
                logger.error(f"❌ {exchange_name} connection error: {e}")
        
        return results
    
    async def disconnect_all(self):
        """Disconnect from all exchanges"""
        for exchange_name, connector in self.exchanges.items():
            try:
                await connector.disconnect()
                self.exchange_status[exchange_name].connected = False
                logger.info(f"🔌 {exchange_name} disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting {exchange_name}: {e}")
    
    async def update_all_account_info(self) -> Dict[str, AccountInfo]:
        """Update account information from all exchanges"""
        account_info = {}
        
        for exchange_name, connector in self.exchanges.items():
            try:
                if not connector.connected:
                    await connector.connect()
                
                info = await connector.get_account_info()
                account_info[exchange_name] = info
                
                # Update consolidated positions
                for symbol, position in info.positions.items():
                    key = f"{exchange_name}_{symbol}"
                    self.positions[key] = position
                
                # Update status
                self.exchange_status[exchange_name].last_heartbeat = datetime.now()
                self.exchange_status[exchange_name].error_count = 0
                self.exchange_status[exchange_name].last_error = None
                
            except Exception as e:
                logger.error(f"Error updating {exchange_name} account info: {e}")
                account_info[exchange_name] = None
                
                # Update error status
                self.exchange_status[exchange_name].error_count += 1
                self.exchange_status[exchange_name].last_error = str(e)
        
        return account_info
    
    async def update_market_data(self, symbols: List[str]) -> Dict[str, Dict[str, MarketData]]:
        """Update market data from all exchanges"""
        market_data = {}
        
        for symbol in symbols:
            market_data[symbol] = {}
            
            for exchange_name, connector in self.exchanges.items():
                try:
                    if not connector.connected:
                        await connector.connect()
                    
                    # Convert symbol format for each exchange
                    formatted_symbol = self._format_symbol_for_exchange(symbol, exchange_name)
                    data = await connector.get_market_data(formatted_symbol)
                    market_data[symbol][exchange_name] = data
                    
                    # Update status
                    self.exchange_status[exchange_name].last_heartbeat = datetime.now()
                    
                except Exception as e:
                    logger.error(f"Error getting {symbol} data from {exchange_name}: {e}")
                    market_data[symbol][exchange_name] = None
                    
                    # Update error status
                    self.exchange_status[exchange_name].error_count += 1
                    self.exchange_status[exchange_name].last_error = str(e)
        
        self.market_data = market_data
        return market_data
    
    def _format_symbol_for_exchange(self, symbol: str, exchange: str) -> str:
        """Format symbol for specific exchange"""
        # Basic symbol formatting - customize based on exchange requirements
        if exchange == "hyperliquid":
            # Hyperliquid uses simple coin names
            return symbol.split('/')[0] if '/' in symbol else symbol
        elif exchange == "binance":
            # Binance futures format
            if '/' not in symbol:
                return f"{symbol}/USDT"
            return symbol
        elif exchange == "okx":
            # OKX swap format
            if '/' not in symbol:
                return f"{symbol}/USDT:USDT"
            return symbol.replace('/', '/USDT:')
        
        return symbol
    
    async def execute_signal(self, signal: TradingSignal) -> List[OrderResult]:
        """
        Execute trading signal across exchanges
        
        Args:
            signal: Trading signal with routing preferences
        """
        results = []
        symbol = signal.symbol
        signal_type = signal.signal_type
        total_size = signal.total_size
        allocation = signal.exchange_allocation
        order_type = signal.order_type
        price = signal.price
        
        # Execute orders on each exchange
        for exchange_name, allocation_pct in allocation.items():
            if exchange_name not in self.exchanges:
                logger.warning(f"Exchange {exchange_name} not available")
                continue
                
            connector = self.exchanges[exchange_name]
            size = total_size * allocation_pct
            
            if size <= 0:
                continue
            
            # Format symbol for this exchange
            formatted_symbol = self._format_symbol_for_exchange(symbol, exchange_name)
            
            # Execute order
            try:
                if not connector.connected:
                    await connector.connect()
                
                result = await connector.place_order(
                    symbol=formatted_symbol,
                    side=signal_type,
                    size=size,
                    order_type=order_type,
                    price=price
                )
                results.append(result)
                
                logger.info(f"✅ {exchange_name} order: {signal_type.value} {size} {symbol} - {result.status.value}")
                
            except Exception as e:
                logger.error(f"❌ Failed to execute on {exchange_name}: {e}")
                
                # Create failed order result
                failed_result = OrderResult(
                    exchange=exchange_name,
                    symbol=formatted_symbol,
                    order_id='',
                    status=OrderStatus.REJECTED,
                    side=signal_type,
                    order_type=order_type,
                    filled_size=0.0,
                    remaining_size=size,
                    avg_fill_price=0.0,
                    fee=0.0,
                    timestamp=datetime.now()
                )
                results.append(failed_result)
        
        return results
    
    def detect_arbitrage_opportunities(self, min_profit_bps: int = 50) -> List[ArbitrageOpportunity]:
        """
        Detect cross-exchange arbitrage opportunities
        
        Args:
            min_profit_bps: Minimum profit in basis points (50 = 0.5%)
        """
        opportunities = []
        
        for symbol, exchange_data in self.market_data.items():
            if len(exchange_data) < 2:
                continue
            
            # Find best bid and ask across exchanges
            best_bid = {'price': 0, 'exchange': ''}
            best_ask = {'price': float('inf'), 'exchange': ''}
            
            for exchange_name, data in exchange_data.items():
                if not data or not data.price:
                    continue
                
                price = data.price
                bid = data.bid or price
                ask = data.ask or price
                
                if bid > best_bid['price']:
                    best_bid = {'price': bid, 'exchange': exchange_name}
                
                if ask < best_ask['price']:
                    best_ask = {'price': ask, 'exchange': exchange_name}
            
            # Calculate potential profit
            if (best_bid['exchange'] != best_ask['exchange'] and 
                best_bid['price'] > 0 and best_ask['price'] < float('inf')):
                
                profit_bps = ((best_bid['price'] - best_ask['price']) / best_ask['price']) * 10000
                
                if profit_bps >= min_profit_bps:
                    opportunities.append(ArbitrageOpportunity(
                        symbol=symbol,
                        buy_exchange=best_ask['exchange'],
                        sell_exchange=best_bid['exchange'],
                        buy_price=best_ask['price'],
                        sell_price=best_bid['price'],
                        profit_bps=profit_bps,
                        volume_available=0.0,  # Would need additional API call
                        timestamp=datetime.now()
                    ))
        
        self.arbitrage_opportunities = opportunities
        return opportunities
    
    def get_consolidated_portfolio(self) -> Dict[str, Any]:
        """Get consolidated view of positions across all exchanges"""
        consolidated = {}
        total_value = 0
        
        # Group positions by symbol
        symbol_positions = {}
        
        for position_key, position in self.positions.items():
            symbol = position.symbol.split('/')[0]  # Get base symbol
            
            if symbol not in symbol_positions:
                symbol_positions[symbol] = {
                    'total_size': 0,
                    'weighted_entry_price': 0,
                    'total_unrealized_pnl': 0,
                    'exchanges': {}
                }
            
            # Add position data
            symbol_data = symbol_positions[symbol]
            size_adjustment = position.size if position.side == PositionSide.LONG else -position.size
            symbol_data['total_size'] += size_adjustment
            symbol_data['total_unrealized_pnl'] += position.unrealized_pnl
            symbol_data['exchanges'][position.exchange] = position
            
            # Calculate weighted average entry price
            total_value += position.size * position.entry_price
        
        consolidated = {
            'positions_by_symbol': symbol_positions,
            'total_exchanges': len(self.exchanges),
            'total_positions': len(self.positions),
            'arbitrage_opportunities': len(self.arbitrage_opportunities),
            'exchange_status': self.exchange_status,
            'last_update': datetime.now()
        }
        
        return consolidated
    
    def get_exchange_status(self) -> Dict[str, ExchangeStatus]:
        """Get status of all exchanges"""
        return self.exchange_status
    
    async def health_check(self) -> Dict[str, bool]:
        """Perform health check on all exchanges"""
        health_status = {}
        
        for exchange_name, connector in self.exchanges.items():
            try:
                # Try to get account info as health check
                await connector.get_account_info()
                health_status[exchange_name] = True
                
                # Update status
                self.exchange_status[exchange_name].connected = True
                self.exchange_status[exchange_name].last_heartbeat = datetime.now()
                self.exchange_status[exchange_name].error_count = 0
                
            except Exception as e:
                health_status[exchange_name] = False
                self.exchange_status[exchange_name].connected = False
                self.exchange_status[exchange_name].error_count += 1
                self.exchange_status[exchange_name].last_error = str(e)
                logger.error(f"Health check failed for {exchange_name}: {e}")
        
        return health_status
    
    def get_available_exchanges(self) -> List[str]:
        """Get list of available exchanges"""
        return list(self.exchanges.keys())
    
    def get_connected_exchanges(self) -> List[str]:
        """Get list of connected exchanges"""
        return [name for name, status in self.exchange_status.items() 
                if status.connected]
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect_all()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect_all()


# Example usage and testing
async def test_multi_exchange_manager():
    """Test the multi-exchange manager"""
    
    # Example configuration (use your actual API keys)
    configs = {
        "hyperliquid": ExchangeConfig(
            name="hyperliquid",
            private_key="your_ethereum_private_key",
            wallet_address="your_wallet_address",
            testnet=True
        ),
        "binance": ExchangeConfig(
            name="binance",
            api_key="your_binance_api_key",
            api_secret="your_binance_secret",
            sandbox=True
        ),
        "okx": ExchangeConfig(
            name="okx",
            api_key="your_okx_api_key", 
            api_secret="your_okx_secret",
            passphrase="your_okx_passphrase",
            sandbox=True
        )
    }
    
    # Initialize manager
    manager = MultiExchangeManager(configs)
    
    try:
        # Connect to exchanges
        connection_results = await manager.connect_all()
        print("Connection Results:", connection_results)
        
        # Update account info
        accounts = await manager.update_all_account_info()
        print("Account Info:", accounts)
        
        # Update market data
        market_data = await manager.update_market_data(['SOL', 'BTC', 'ETH'])
        print("Market Data:", market_data)
        
        # Check for arbitrage
        arbitrage = manager.detect_arbitrage_opportunities(min_profit_bps=25)
        print("Arbitrage Opportunities:", arbitrage)
        
        # Get consolidated portfolio
        portfolio = manager.get_consolidated_portfolio()
        print("Portfolio:", portfolio)
        
        # Health check
        health = await manager.health_check()
        print("Health Check:", health)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
    
    finally:
        # Disconnect
        await manager.disconnect_all()


if __name__ == "__main__":
    # Run test
    asyncio.run(test_multi_exchange_manager()) 