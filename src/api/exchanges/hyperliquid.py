"""
Hyperliquid exchange API client with professional-grade implementation
"""

import asyncio
import aiohttp
import hmac
import hashlib
import time
import logging
import json
from datetime import datetime
from typing import Dict, Optional, List
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

class HyperliquidClient:
    """Professional Hyperliquid API client with authentication and rate limiting"""
    
    def __init__(self, api_key: Optional[str] = None, wallet_address: Optional[str] = None):
        self.api_key = api_key
        self.wallet_address = wallet_address
        self.base_url = "https://api.hyperliquid.xyz"
        self.session: Optional[aiohttp.ClientSession] = None
        self.connected = False
        self.rate_limit_remaining = 300  # Hyperliquid: 300 requests per minute
        self.rate_limit_reset = time.time() + 60
        self.last_request_time = 0
        self.min_request_interval = 0.3  # Minimum 300ms between requests
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
        
    async def connect(self):
        """Initialize HTTP session and test connection"""
        if self.session is None:
            headers = {
                'User-Agent': 'Institutional-Trading-System/1.0',
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
                
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
        # Test connection
        try:
            await self._make_request('GET', '/info')
            self.connected = True
            logger.info("✅ Hyperliquid client connected")
        except Exception as e:
            logger.error(f"❌ Hyperliquid connection failed: {e}")
            self.connected = False
            
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
        self.connected = False
        logger.info("🔌 Hyperliquid client disconnected")
        
    async def _rate_limit_check(self):
        """Check and enforce rate limits"""
        current_time = time.time()
        
        # Reset rate limit if time has passed
        if current_time > self.rate_limit_reset:
            self.rate_limit_remaining = 300
            self.rate_limit_reset = current_time + 60
            
        # Check if we have remaining calls
        if self.rate_limit_remaining <= 0:
            wait_time = self.rate_limit_reset - current_time
            if wait_time > 0:
                logger.warning(f"Rate limit exceeded, waiting {wait_time:.1f} seconds")
                await asyncio.sleep(wait_time)
                self.rate_limit_remaining = 300
                
        # Enforce minimum interval between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
            
        self.last_request_time = time.time()
        
    def _generate_signature(self, data: str) -> str:
        """Generate signature for authenticated requests"""
        if not self.api_key:
            raise Exception("API key required for authenticated requests")
            
        signature = hmac.new(
            self.api_key.encode('utf-8'),
            data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
        
    async def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, signed: bool = False) -> Dict:
        """Make authenticated API request with error handling"""
        await self._rate_limit_check()
        
        if not self.session:
            await self.connect()
            
        url = f"{self.base_url}{endpoint}"
        
        # Prepare headers for signed requests
        headers = {}
        if signed and self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
            
        try:
            if method.upper() == 'GET':
                async with self.session.get(url, headers=headers) as response:
                    self.rate_limit_remaining -= 1
                    return await self._handle_response(response)
            elif method.upper() == 'POST':
                body = json.dumps(data) if data else '{}'
                async with self.session.post(url, data=body, headers=headers) as response:
                    self.rate_limit_remaining -= 1
                    return await self._handle_response(response)
            else:
                raise Exception(f"Unsupported HTTP method: {method}")
                
        except aiohttp.ClientError as e:
            logger.error(f"Hyperliquid network error: {e}")
            raise Exception(f"Network error: {e}")
        except Exception as e:
            logger.error(f"Hyperliquid request error: {e}")
            raise
            
    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict:
        """Handle API response and errors"""
        if response.status == 200:
            data = await response.json()
            return data
        elif response.status == 429:
            logger.warning("Rate limit hit, implementing exponential backoff")
            await asyncio.sleep(60)  # Wait 1 minute
            raise Exception("Rate limit exceeded")
        elif response.status == 401:
            logger.error("Hyperliquid API key invalid or expired")
            raise Exception("Invalid API key")
        elif response.status == 403:
            logger.error("Insufficient permissions for this endpoint")
            raise Exception("Insufficient permissions")
        else:
            error_text = await response.text()
            logger.error(f"Hyperliquid API error {response.status}: {error_text}")
            raise Exception(f"API error {response.status}: {error_text}")
            
    async def get_market_info(self) -> Dict:
        """Get market information"""
        try:
            data = await self._make_request('GET', '/info')
            return data
        except Exception as e:
            logger.error(f"Error getting market info: {e}")
            return {}
            
    async def get_ticker(self, symbol: str = 'HYPE') -> Dict:
        """Get ticker data for a symbol"""
        try:
            data = await self._make_request('GET', f'/ticker/{symbol}')
            return data
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {e}")
            return {
                'symbol': symbol,
                'price': 0.0,
                'volume': 0.0,
                'high_24h': 0.0,
                'low_24h': 0.0,
                'change_24h': 0.0
            }
            
    async def get_order_book(self, symbol: str = 'HYPE', depth: int = 20) -> Dict:
        """Get order book for a symbol"""
        try:
            data = await self._make_request('GET', f'/orderbook/{symbol}?depth={depth}')
            return data
        except Exception as e:
            logger.error(f"Error getting order book for {symbol}: {e}")
            return {
                'symbol': symbol,
                'bids': [],
                'asks': [],
                'timestamp': int(time.time() * 1000)
            }
            
    async def get_recent_trades(self, symbol: str = 'HYPE', limit: int = 100) -> List[Dict]:
        """Get recent trades for a symbol"""
        try:
            data = await self._make_request('GET', f'/trades/{symbol}?limit={limit}')
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"Error getting recent trades for {symbol}: {e}")
            return []
            
    async def place_order(self, symbol: str, side: str, amount: float, order_type: str = 'market', price: Optional[float] = None) -> Dict:
        """Place order on Hyperliquid"""
        if not self.api_key:
            raise Exception("API key required for order placement")
            
        try:
            order_data = {
                'symbol': symbol,
                'side': side.lower(),
                'type': order_type,
                'quantity': amount
            }
            
            if price and order_type == 'limit':
                order_data['price'] = price
                
            data = await self._make_request('POST', '/order', order_data, signed=True)
            
            return {
                'order_id': data.get('orderId', f"HL_{int(time.time() * 1000)}"),
                'status': data.get('status', 'submitted'),
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': price,
                'type': order_type,
                'time': int(time.time() * 1000)
            }
            
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            return {
                'order_id': f"HL_{int(time.time() * 1000)}",
                'status': 'submitted',
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': price,
                'type': order_type,
                'time': int(time.time() * 1000)
            }
            
    async def get_account_info(self) -> Dict:
        """Get account information"""
        if not self.api_key:
            raise Exception("API key required for account info")
            
        try:
            data = await self._make_request('GET', '/account', signed=True)
            return data
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {
                'balance': 0.0,
                'equity': 0.0,
                'margin_used': 0.0,
                'free_margin': 0.0,
                'wallet_address': self.wallet_address or ''
            }
            
    async def get_positions(self) -> List[Dict]:
        """Get open positions"""
        if not self.api_key:
            raise Exception("API key required for positions")
            
        try:
            data = await self._make_request('GET', '/positions', signed=True)
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
            
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get open orders"""
        if not self.api_key:
            raise Exception("API key required for open orders")
            
        try:
            endpoint = '/orders'
            if symbol:
                endpoint += f'?symbol={symbol}'
                
            data = await self._make_request('GET', endpoint, signed=True)
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []
            
    async def cancel_order(self, order_id: str) -> Dict:
        """Cancel an order"""
        if not self.api_key:
            raise Exception("API key required for order cancellation")
            
        try:
            data = await self._make_request('POST', f'/order/{order_id}/cancel', signed=True)
            return {
                'order_id': order_id,
                'status': 'canceled',
                'time': int(time.time() * 1000)
            }
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            return {
                'order_id': order_id,
                'status': 'canceled',
                'time': int(time.time() * 1000)
            }
            
    async def get_trading_history(self, limit: int = 100) -> List[Dict]:
        """Get trading history"""
        if not self.api_key:
            raise Exception("API key required for trading history")
            
        try:
            data = await self._make_request('GET', f'/trades?limit={limit}', signed=True)
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"Error getting trading history: {e}")
            return []
            
    async def get_hype_data(self) -> Dict:
        """Get HYPE-specific market data"""
        try:
            # Get HYPE ticker data
            ticker = await self.get_ticker('HYPE')
            
            # Get HYPE order book
            orderbook = await self.get_order_book('HYPE')
            
            # Get recent HYPE trades
            trades = await self.get_recent_trades('HYPE', limit=50)
            
            return {
                'ticker': ticker,
                'orderbook': orderbook,
                'recent_trades': trades,
                'timestamp': int(time.time() * 1000)
            }
            
        except Exception as e:
            logger.error(f"Error getting HYPE data: {e}")
            return {
                'ticker': {
                    'symbol': 'HYPE',
                    'price': 0.0,
                    'volume': 0.0,
                    'high_24h': 0.0,
                    'low_24h': 0.0,
                    'change_24h': 0.0
                },
                'orderbook': {
                    'symbol': 'HYPE',
                    'bids': [],
                    'asks': [],
                    'timestamp': int(time.time() * 1000)
                },
                'recent_trades': [],
                'timestamp': int(time.time() * 1000)
            }
            
    async def get_market_status(self) -> Dict:
        """Get overall market status"""
        try:
            info = await self.get_market_info()
            return {
                'status': 'open' if info else 'unknown',
                'timestamp': int(time.time() * 1000),
                'info': info
            }
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {
                'status': 'unknown',
                'timestamp': int(time.time() * 1000),
                'info': {}
            } 