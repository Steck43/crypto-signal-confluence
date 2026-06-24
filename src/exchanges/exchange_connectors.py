"""
Exchange Connectors

Individual exchange connectors for Hyperliquid, Binance, and OKX.
Each connector handles the specific API requirements and data formats.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

# Import our types
from .exchange_types import (
    Position, OrderResult, MarketData, AccountInfo, ExchangeConfig,
    OrderSide, OrderType, OrderStatus, PositionSide,
    ExchangeError, AuthenticationError, RateLimitError, ConnectionError
)

logger = logging.getLogger(__name__)


class BaseExchangeConnector:
    """Base class for all exchange connectors"""
    
    def __init__(self, config: ExchangeConfig):
        self.config = config
        self.connected = False
        self.last_heartbeat = None
        self.error_count = 0
        self.rate_limit_remaining = config.rate_limit
        self.rate_limit_reset = time.time() + 60
        
    async def connect(self) -> bool:
        """Connect to exchange"""
        raise NotImplementedError
        
    async def disconnect(self) -> None:
        """Disconnect from exchange"""
        raise NotImplementedError
        
    async def get_account_info(self) -> AccountInfo:
        """Get account information"""
        raise NotImplementedError
        
    async def get_market_data(self, symbol: str) -> MarketData:
        """Get market data for symbol"""
        raise NotImplementedError
        
    async def place_order(self, symbol: str, side: OrderSide, size: float,
                         order_type: OrderType = OrderType.MARKET,
                         price: Optional[float] = None) -> OrderResult:
        """Place order on exchange"""
        raise NotImplementedError
        
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel order"""
        raise NotImplementedError
        
    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        raise NotImplementedError
        
    def _rate_limit_check(self):
        """Check and enforce rate limits"""
        current_time = time.time()
        
        # Reset rate limit if time has passed
        if current_time > self.rate_limit_reset:
            self.rate_limit_remaining = self.config.rate_limit
            self.rate_limit_reset = current_time + 60
            
        # Check if we have remaining calls
        if self.rate_limit_remaining <= 0:
            wait_time = self.rate_limit_reset - current_time
            if wait_time > 0:
                logger.warning(f"Rate limit exceeded, waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)
                self.rate_limit_remaining = self.config.rate_limit
                
        self.rate_limit_remaining -= 1


class HyperliquidConnector(BaseExchangeConnector):
    """Hyperliquid exchange connector"""
    
    def __init__(self, config: ExchangeConfig):
        super().__init__(config)
        self.exchange = None
        self.info = None
        
        # Try to import Hyperliquid SDK
        try:
            import hyperliquid.utils.constants as constants
            from hyperliquid.exchange import Exchange as HyperliquidExchange
            from hyperliquid.info import Info as HyperliquidInfo
            
            self.constants = constants
            self.HyperliquidExchange = HyperliquidExchange
            self.HyperliquidInfo = HyperliquidInfo
            self.sdk_available = True
            
        except ImportError:
            logger.warning("Hyperliquid SDK not available. Install with: pip install hyperliquid-python-sdk")
            self.sdk_available = False
    
    async def connect(self) -> bool:
        """Connect to Hyperliquid"""
        if not self.sdk_available:
            raise ConnectionError("hyperliquid", "SDK not available")
            
        if not self.config.is_configured():
            raise AuthenticationError("hyperliquid", "Missing private key or wallet address")
            
        try:
            # Initialize Hyperliquid clients
            base_url = self.constants.TESTNET_API_URL if self.config.testnet else self.constants.MAINNET_API_URL
            self.exchange = self.HyperliquidExchange(self.config.private_key, base_url)
            self.info = self.HyperliquidInfo(base_url)
            
            # Test connection
            await self._test_connection()
            
            self.connected = True
            self.last_heartbeat = datetime.now()
            logger.info("✅ Hyperliquid connected")
            return True
            
        except Exception as e:
            logger.error(f"❌ Hyperliquid connection failed: {e}")
            self.connected = False
            raise ConnectionError("hyperliquid", str(e))
    
    async def disconnect(self) -> None:
        """Disconnect from Hyperliquid"""
        self.connected = False
        self.exchange = None
        self.info = None
        logger.info("🔌 Hyperliquid disconnected")
    
    async def _test_connection(self):
        """Test connection to Hyperliquid"""
        try:
            # Get market info to test connection
            meta = self.info.meta()
            if not meta:
                raise ConnectionError("hyperliquid", "Failed to get market info")
        except Exception as e:
            raise ConnectionError("hyperliquid", f"Connection test failed: {e}")
    
    async def get_account_info(self) -> AccountInfo:
        """Get Hyperliquid account information"""
        if not self.connected:
            await self.connect()
            
        try:
            self._rate_limit_check()
            
            # Get user state
            user_state = self.info.user_state(self.exchange.wallet.address)
            
            # Parse positions
            positions = {}
            if 'assetPositions' in user_state:
                for pos in user_state['assetPositions']:
                    symbol = pos['position']['coin']
                    size = float(pos['position']['szi'])
                    if size != 0:  # Only include non-zero positions
                        positions[symbol] = Position(
                            exchange='hyperliquid',
                            symbol=symbol,
                            size=abs(size),
                            side=PositionSide.LONG if size > 0 else PositionSide.SHORT,
                            entry_price=float(pos['position']['entryPx']),
                            current_price=float(pos['position']['positionValue']) / abs(size),
                            unrealized_pnl=float(pos['position']['unrealizedPnl']),
                            timestamp=datetime.now()
                        )
            
            margin_summary = user_state.get('marginSummary', {})
            
            return AccountInfo(
                exchange='hyperliquid',
                total_balance=float(margin_summary.get('accountValue', 0)),
                available_balance=float(margin_summary.get('freeCollateral', 0)),
                margin_balance=float(margin_summary.get('totalMarginUsed', 0)),
                unrealized_pnl=float(margin_summary.get('unrealizedPnl', 0)),
                realized_pnl=0.0,  # Would need additional API call
                margin_ratio=float(margin_summary.get('marginRatio', 0)),
                positions=positions,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting Hyperliquid account info: {e}")
            raise ExchangeError("hyperliquid", f"Failed to get account info: {e}")
    
    async def get_market_data(self, symbol: str) -> MarketData:
        """Get Hyperliquid market data"""
        if not self.connected:
            await self.connect()
            
        try:
            self._rate_limit_check()
            
            # Get current market info
            meta = self.info.meta()
            all_mids = self.info.all_mids()
            
            # Find symbol in meta
            symbol_info = None
            for universe in meta.get('universe', []):
                if universe.get('name') == symbol:
                    symbol_info = universe
                    break
            
            if not symbol_info:
                raise ExchangeError("hyperliquid", f"Symbol {symbol} not found")
            
            current_price = float(all_mids.get(symbol, 0))
            
            return MarketData(
                exchange='hyperliquid',
                symbol=symbol,
                price=current_price,
                bid=current_price * 0.999,  # Approximate
                ask=current_price * 1.001,  # Approximate
                volume_24h=0.0,  # Would need additional API call
                high_24h=current_price * 1.05,  # Approximate
                low_24h=current_price * 0.95,  # Approximate
                price_change_24h=0.0,  # Would need additional API call
                price_change_percent_24h=0.0,  # Would need additional API call
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting Hyperliquid market data: {e}")
            raise ExchangeError("hyperliquid", f"Failed to get market data: {e}")
    
    async def place_order(self, symbol: str, side: OrderSide, size: float,
                         order_type: OrderType = OrderType.MARKET,
                         price: Optional[float] = None) -> OrderResult:
        """Place order on Hyperliquid"""
        if not self.connected:
            await self.connect()
            
        try:
            self._rate_limit_check()
            
            # Convert side to Hyperliquid format
            is_buy = side == OrderSide.BUY
            
            # Prepare order
            if order_type == OrderType.MARKET:
                order_result = self.exchange.market_order(
                    coin=symbol,
                    is_buy=is_buy,
                    sz=size,
                    reduce_only=False
                )
            else:  # limit order
                if price is None:
                    raise ValueError("Price required for limit orders")
                
                order_result = self.exchange.order(
                    coin=symbol,
                    is_buy=is_buy,
                    sz=size,
                    limit_px=price,
                    order_type={'limit': {'tif': 'Gtc'}},
                    reduce_only=False
                )
            
            # Parse result
            if order_result and 'status' in order_result:
                status = order_result['status']
                if status == 'ok' and 'response' in order_result:
                    response = order_result['response']
                    if 'data' in response and 'statuses' in response['data']:
                        order_status = response['data']['statuses'][0]
                        
                        return OrderResult(
                            exchange='hyperliquid',
                            symbol=symbol,
                            order_id=str(order_status.get('oid', '')),
                            status=OrderStatus.FILLED if order_status.get('filled') else OrderStatus.PENDING,
                            side=side,
                            order_type=order_type,
                            filled_size=float(order_status.get('totalSz', 0)),
                            remaining_size=size - float(order_status.get('totalSz', 0)),
                            avg_fill_price=float(order_status.get('avgPx', price or 0)),
                            fee=0.0,  # Fee calculation would need additional API call
                            timestamp=datetime.now()
                        )
            
            # If we get here, order failed
            logger.error(f"Hyperliquid order failed: {order_result}")
            return OrderResult(
                exchange='hyperliquid',
                symbol=symbol,
                order_id='',
                status=OrderStatus.REJECTED,
                side=side,
                order_type=order_type,
                filled_size=0.0,
                remaining_size=size,
                avg_fill_price=0.0,
                fee=0.0,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error placing Hyperliquid order: {e}")
            return OrderResult(
                exchange='hyperliquid',
                symbol=symbol,
                order_id='',
                status=OrderStatus.REJECTED,
                side=side,
                order_type=order_type,
                filled_size=0.0,
                remaining_size=size,
                avg_fill_price=0.0,
                fee=0.0,
                timestamp=datetime.now()
            )
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel order on Hyperliquid"""
        # Implementation would depend on Hyperliquid API
        logger.warning("Cancel order not implemented for Hyperliquid")
        return False
    
    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        account_info = await self.get_account_info()
        return account_info.positions


class BinanceConnector(BaseExchangeConnector):
    """Binance exchange connector using CCXT"""
    
    def __init__(self, config: ExchangeConfig):
        super().__init__(config)
        self.exchange = None
        
        # Try to import CCXT
        try:
            import ccxt
            self.ccxt = ccxt
            self.sdk_available = True
        except ImportError:
            logger.warning("CCXT not available. Install with: pip install ccxt")
            self.sdk_available = False
    
    async def connect(self) -> bool:
        """Connect to Binance"""
        if not self.sdk_available:
            raise ConnectionError("binance", "CCXT not available")
            
        if not self.config.is_configured():
            raise AuthenticationError("binance", "Missing API key or secret")
            
        try:
            self.exchange = self.ccxt.binance({
                'apiKey': self.config.api_key,
                'secret': self.config.api_secret,
                'sandbox': self.config.sandbox,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',  # Use futures by default
                }
            })
            
            # Test connection
            await self._test_connection()
            
            self.connected = True
            self.last_heartbeat = datetime.now()
            logger.info("✅ Binance connected")
            return True
            
        except Exception as e:
            logger.error(f"❌ Binance connection failed: {e}")
            self.connected = False
            raise ConnectionError("binance", str(e))
    
    async def disconnect(self) -> None:
        """Disconnect from Binance"""
        self.connected = False
        self.exchange = None
        logger.info("🔌 Binance disconnected")
    
    async def _test_connection(self):
        """Test connection to Binance"""
        try:
            # Test with a simple API call
            await self.exchange.fetch_time()
        except Exception as e:
            raise ConnectionError("binance", f"Connection test failed: {e}")
    
    async def get_account_info(self) -> AccountInfo:
        """Get Binance account information"""
        if not self.connected:
            await self.connect()
            
        try:
            self._rate_limit_check()
            
            # Get futures account info
            balance = await self.exchange.fetch_balance()
            positions_data = await self.exchange.fetch_positions()
            
            # Parse positions
            positions = {}
            for pos in positions_data:
                if float(pos['contracts']) != 0:  # Only non-zero positions
                    symbol = pos['symbol']
                    positions[symbol] = Position(
                        exchange='binance',
                        symbol=symbol,
                        size=float(pos['contracts']),
                        side=PositionSide.LONG if pos['side'] == 'long' else PositionSide.SHORT,
                        entry_price=float(pos['entryPrice']),
                        current_price=float(pos['markPrice']),
                        unrealized_pnl=float(pos['unrealizedPnl']),
                        timestamp=datetime.now()
                    )
            
            return AccountInfo(
                exchange='binance',
                total_balance=float(balance['total']['USDT']),
                available_balance=float(balance['free']['USDT']),
                margin_balance=float(balance['used']['USDT']),
                unrealized_pnl=sum(pos.unrealized_pnl for pos in positions.values()),
                realized_pnl=0.0,  # Would need additional API call
                margin_ratio=0.0,  # Would need additional API call
                positions=positions,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting Binance account info: {e}")
            raise ExchangeError("binance", f"Failed to get account info: {e}")
    
    async def get_market_data(self, symbol: str) -> MarketData:
        """Get Binance market data"""
        if not self.connected:
            await self.connect()
            
        try:
            self._rate_limit_check()
            
            ticker = await self.exchange.fetch_ticker(symbol)
            
            return MarketData(
                exchange='binance',
                symbol=symbol,
                price=float(ticker['last']),
                bid=float(ticker['bid']),
                ask=float(ticker['ask']),
                volume_24h=float(ticker['baseVolume']),
                high_24h=float(ticker['high']),
                low_24h=float(ticker['low']),
                price_change_24h=float(ticker['change']),
                price_change_percent_24h=float(ticker['percentage']),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting Binance market data: {e}")
            raise ExchangeError("binance", f"Failed to get market data: {e}")
    
    async def place_order(self, symbol: str, side: OrderSide, size: float,
                         order_type: OrderType = OrderType.MARKET,
                         price: Optional[float] = None) -> OrderResult:
        """Place order on Binance"""
        if not self.connected:
            await self.connect()
            
        try:
            self._rate_limit_check()
            
            # Place order using CCXT
            if order_type == OrderType.MARKET:
                order = await self.exchange.create_market_order(symbol, side.value, size)
            else:
                if price is None:
                    raise ValueError("Price required for limit orders")
                order = await self.exchange.create_limit_order(symbol, side.value, size, price)
            
            return OrderResult(
                exchange='binance',
                symbol=symbol,
                order_id=order['id'],
                status=OrderStatus(order['status']),
                side=side,
                order_type=order_type,
                filled_size=float(order['filled']),
                remaining_size=float(order['remaining']),
                avg_fill_price=float(order['average']) if order['average'] else 0.0,
                fee=float(order['fee']['cost']) if order.get('fee') else 0.0,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error placing Binance order: {e}")
            return OrderResult(
                exchange='binance',
                symbol=symbol,
                order_id='',
                status=OrderStatus.REJECTED,
                side=side,
                order_type=order_type,
                filled_size=0.0,
                remaining_size=size,
                avg_fill_price=0.0,
                fee=0.0,
                timestamp=datetime.now()
            )
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel order on Binance"""
        if not self.connected:
            await self.connect()
            
        try:
            self._rate_limit_check()
            await self.exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.error(f"Error canceling Binance order: {e}")
            return False
    
    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        account_info = await self.get_account_info()
        return account_info.positions


class OKXConnector(BaseExchangeConnector):
    """OKX exchange connector using CCXT"""
    
    def __init__(self, config: ExchangeConfig):
        super().__init__(config)
        self.exchange = None
        
        # Try to import CCXT
        try:
            import ccxt
            self.ccxt = ccxt
            self.sdk_available = True
        except ImportError:
            logger.warning("CCXT not available. Install with: pip install ccxt")
            self.sdk_available = False
    
    async def connect(self) -> bool:
        """Connect to OKX"""
        if not self.sdk_available:
            raise ConnectionError("okx", "CCXT not available")
            
        if not self.config.is_configured():
            raise AuthenticationError("okx", "Missing API key, secret, or passphrase")
            
        try:
            self.exchange = self.ccxt.okx({
                'apiKey': self.config.api_key,
                'secret': self.config.api_secret,
                'password': self.config.passphrase,
                'sandbox': self.config.sandbox,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap',  # Use perpetual swaps
                }
            })
            
            # Test connection
            await self._test_connection()
            
            self.connected = True
            self.last_heartbeat = datetime.now()
            logger.info("✅ OKX connected")
            return True
            
        except Exception as e:
            logger.error(f"❌ OKX connection failed: {e}")
            self.connected = False
            raise ConnectionError("okx", str(e))
    
    async def disconnect(self) -> None:
        """Disconnect from OKX"""
        self.connected = False
        self.exchange = None
        logger.info("🔌 OKX disconnected")
    
    async def _test_connection(self):
        """Test connection to OKX"""
        try:
            # Test with a simple API call
            await self.exchange.fetch_time()
        except Exception as e:
            raise ConnectionError("okx", f"Connection test failed: {e}")
    
    async def get_account_info(self) -> AccountInfo:
        """Get OKX account information"""
        if not self.connected:
            await self.connect()
            
        try:
            self._rate_limit_check()
            
            balance = await self.exchange.fetch_balance()
            positions_data = await self.exchange.fetch_positions()
            
            # Parse positions
            positions = {}
            for pos in positions_data:
                if float(pos['contracts']) != 0:
                    symbol = pos['symbol']
                    positions[symbol] = Position(
                        exchange='okx',
                        symbol=symbol,
                        size=float(pos['contracts']),
                        side=PositionSide.LONG if pos['side'] == 'long' else PositionSide.SHORT,
                        entry_price=float(pos['entryPrice']),
                        current_price=float(pos['markPrice']),
                        unrealized_pnl=float(pos['unrealizedPnl']),
                        timestamp=datetime.now()
                    )
            
            return AccountInfo(
                exchange='okx',
                total_balance=float(balance['total']['USDT']),
                available_balance=float(balance['free']['USDT']),
                margin_balance=float(balance['used']['USDT']),
                unrealized_pnl=sum(pos.unrealized_pnl for pos in positions.values()),
                realized_pnl=0.0,  # Would need additional API call
                margin_ratio=0.0,  # Would need additional API call
                positions=positions,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting OKX account info: {e}")
            raise ExchangeError("okx", f"Failed to get account info: {e}")
    
    async def get_market_data(self, symbol: str) -> MarketData:
        """Get OKX market data"""
        if not self.connected:
            await self.connect()
            
        try:
            self._rate_limit_check()
            
            ticker = await self.exchange.fetch_ticker(symbol)
            
            return MarketData(
                exchange='okx',
                symbol=symbol,
                price=float(ticker['last']),
                bid=float(ticker['bid']),
                ask=float(ticker['ask']),
                volume_24h=float(ticker['baseVolume']),
                high_24h=float(ticker['high']),
                low_24h=float(ticker['low']),
                price_change_24h=float(ticker['change']),
                price_change_percent_24h=float(ticker['percentage']),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting OKX market data: {e}")
            raise ExchangeError("okx", f"Failed to get market data: {e}")
    
    async def place_order(self, symbol: str, side: OrderSide, size: float,
                         order_type: OrderType = OrderType.MARKET,
                         price: Optional[float] = None) -> OrderResult:
        """Place order on OKX"""
        if not self.connected:
            await self.connect()
            
        try:
            self._rate_limit_check()
            
            if order_type == OrderType.MARKET:
                order = await self.exchange.create_market_order(symbol, side.value, size)
            else:
                if price is None:
                    raise ValueError("Price required for limit orders")
                order = await self.exchange.create_limit_order(symbol, side.value, size, price)
            
            return OrderResult(
                exchange='okx',
                symbol=symbol,
                order_id=order['id'],
                status=OrderStatus(order['status']),
                side=side,
                order_type=order_type,
                filled_size=float(order['filled']),
                remaining_size=float(order['remaining']),
                avg_fill_price=float(order['average']) if order['average'] else 0.0,
                fee=float(order['fee']['cost']) if order.get('fee') else 0.0,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error placing OKX order: {e}")
            return OrderResult(
                exchange='okx',
                symbol=symbol,
                order_id='',
                status=OrderStatus.REJECTED,
                side=side,
                order_type=order_type,
                filled_size=0.0,
                remaining_size=size,
                avg_fill_price=0.0,
                fee=0.0,
                timestamp=datetime.now()
            )
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel order on OKX"""
        if not self.connected:
            await self.connect()
            
        try:
            self._rate_limit_check()
            await self.exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.error(f"Error canceling OKX order: {e}")
            return False
    
    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        account_info = await self.get_account_info()
        return account_info.positions 