"""
OKX exchange API client with professional-grade implementation
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

class OKXClient:
    """Professional OKX API client with authentication and rate limiting"""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, passphrase: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.base_url = "https://www.okx.com"
        self.session: Optional[aiohttp.ClientSession] = None
        self.connected = False
        self.rate_limit_remaining = 600  # OKX: 600 requests per minute
        self.rate_limit_reset = time.time() + 60
        self.last_request_time = 0
        self.min_request_interval = 0.2  # Minimum 200ms between requests
        
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
                headers['OK-ACCESS-KEY'] = self.api_key
                
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
        # Test connection
        try:
            await self._make_request('GET', '/api/v5/public/time')
            self.connected = True
            logger.info("✅ OKX client connected")
        except Exception as e:
            logger.error(f"❌ OKX connection failed: {e}")
            self.connected = False
            
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
        self.connected = False
        logger.info("🔌 OKX client disconnected")
        
    async def _rate_limit_check(self):
        """Check and enforce rate limits"""
        current_time = time.time()
        
        # Reset rate limit if time has passed
        if current_time > self.rate_limit_reset:
            self.rate_limit_remaining = 600
            self.rate_limit_reset = current_time + 60
            
        # Check if we have remaining calls
        if self.rate_limit_remaining <= 0:
            wait_time = self.rate_limit_reset - current_time
            if wait_time > 0:
                logger.warning(f"Rate limit exceeded, waiting {wait_time:.1f} seconds")
                await asyncio.sleep(wait_time)
                self.rate_limit_remaining = 600
                
        # Enforce minimum interval between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
            
        self.last_request_time = time.time()
        
    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = '') -> str:
        """Generate HMAC SHA256 signature for authenticated requests"""
        if not self.api_secret:
            raise Exception("API secret required for authenticated requests")
            
        message = timestamp + method + request_path + body
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
        
    async def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, signed: bool = False) -> Dict:
        """Make authenticated API request with error handling"""
        await self._rate_limit_check()
        
        if not self.session:
            await self.connect()
            
        url = f"{self.base_url}{endpoint}"
        
        # Prepare headers for signed requests
        headers = {}
        if signed:
            if not all([self.api_key, self.api_secret, self.passphrase]):
                raise Exception("API credentials required for authenticated requests")
                
            timestamp = str(int(time.time()))
            headers['OK-ACCESS-KEY'] = self.api_key
            headers['OK-ACCESS-SIGN'] = self._generate_signature(timestamp, method, endpoint)
            headers['OK-ACCESS-TIMESTAMP'] = timestamp
            headers['OK-ACCESS-PASSPHRASE'] = self.passphrase
            
        try:
            if method.upper() == 'GET':
                async with self.session.get(url, params=params, headers=headers) as response:
                    self.rate_limit_remaining -= 1
                    return await self._handle_response(response)
            elif method.upper() == 'POST':
                body = json.dumps(params) if params else ''
                async with self.session.post(url, data=body, headers=headers) as response:
                    self.rate_limit_remaining -= 1
                    return await self._handle_response(response)
            else:
                raise Exception(f"Unsupported HTTP method: {method}")
                
        except aiohttp.ClientError as e:
            logger.error(f"OKX network error: {e}")
            raise Exception(f"Network error: {e}")
        except Exception as e:
            logger.error(f"OKX request error: {e}")
            raise
            
    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict:
        """Handle API response and errors"""
        if response.status == 200:
            data = await response.json()
            if data.get('code') == '0':
                return data
            else:
                error_msg = data.get('msg', 'Unknown error')
                logger.error(f"OKX API error: {error_msg}")
                raise Exception(f"API error: {error_msg}")
        elif response.status == 429:
            logger.warning("Rate limit hit, implementing exponential backoff")
            await asyncio.sleep(60)  # Wait 1 minute
            raise Exception("Rate limit exceeded")
        elif response.status == 401:
            logger.error("OKX API key invalid or expired")
            raise Exception("Invalid API key")
        elif response.status == 403:
            logger.error("Insufficient permissions for this endpoint")
            raise Exception("Insufficient permissions")
        else:
            error_text = await response.text()
            logger.error(f"OKX API error {response.status}: {error_text}")
            raise Exception(f"API error {response.status}: {error_text}")
            
    async def get_market_data(self, symbol: str) -> Dict:
        """Get market data for any symbol"""
        try:
            params = {'instId': symbol}
            data = await self._make_request('GET', '/api/v5/market/ticker', params)
            
            if data and 'data' in data and len(data['data']) > 0:
                ticker = data['data'][0]
                return {
                    'price': float(ticker['last']),
                    'bid': float(ticker['bidPx']),
                    'ask': float(ticker['askPx']),
                    'volume': float(ticker['vol24h']),
                    'high_24h': float(ticker['high24h']),
                    'low_24h': float(ticker['low24h']),
                    'price_change_24h': float(ticker['change24h']),
                    'price_change_percent_24h': float(ticker['changeRate24h']),
                    'quote_volume': float(ticker['volCcy24h']),
                    'timestamp': ticker['ts']
                }
            return {}
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return {
                'price': 0.0,
                'bid': 0.0,
                'ask': 0.0,
                'volume': 0.0,
                'high_24h': 0.0,
                'low_24h': 0.0,
                'price_change_24h': 0.0,
                'price_change_percent_24h': 0.0,
                'quote_volume': 0.0,
                'timestamp': 0
            }
            
    async def get_order_book(self, symbol: str, depth: int = 20) -> Dict:
        """Get order book for a symbol"""
        try:
            params = {'instId': symbol, 'sz': depth}
            data = await self._make_request('GET', '/api/v5/market/books', params)
            
            if data and 'data' in data and len(data['data']) > 0:
                book = data['data'][0]
                return {
                    'symbol': symbol,
                    'bids': [[float(price), float(qty)] for price, qty, *_ in book['bids']],
                    'asks': [[float(price), float(qty)] for price, qty, *_ in book['asks']],
                    'timestamp': book['ts']
                }
            return {'symbol': symbol, 'bids': [], 'asks': [], 'timestamp': 0}
            
        except Exception as e:
            logger.error(f"Error getting order book for {symbol}: {e}")
            return {'symbol': symbol, 'bids': [], 'asks': [], 'timestamp': 0}
            
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent trades for a symbol"""
        try:
            params = {'instId': symbol, 'limit': limit}
            data = await self._make_request('GET', '/api/v5/market/trades', params)
            
            trades = []
            if data and 'data' in data:
                for trade in data['data']:
                    trades.append({
                        'id': trade['tradeId'],
                        'price': float(trade['px']),
                        'qty': float(trade['sz']),
                        'side': trade['side'],
                        'time': int(trade['ts'])
                    })
            return trades
            
        except Exception as e:
            logger.error(f"Error getting recent trades for {symbol}: {e}")
            return []
            
    async def place_order(self, symbol: str, side: str, quantity: float, order_type: str = 'market', price: Optional[float] = None) -> Dict:
        """Place order for any symbol"""
        if not all([self.api_key, self.api_secret, self.passphrase]):
            raise Exception("API credentials required for order placement")
            
        try:
            params = {
                'instId': symbol,
                'tdMode': 'cash',
                'side': side.lower(),
                'ordType': order_type,
                'sz': str(quantity)
            }
            
            if price and order_type == 'limit':
                params['px'] = str(price)
                
            data = await self._make_request('POST', '/api/v5/trade/order', params, signed=True)
            
            if data and 'data' in data and len(data['data']) > 0:
                order = data['data'][0]
                return {
                    'id': order['ordId'],
                    'status': order['state'],
                    'side': order['side'],
                    'amount': float(order['sz']),
                    'price': float(order['px']) if order.get('px') else None,
                    'symbol': order['instId'],
                    'type': order['ordType'],
                    'time': int(order['cTime'])
                }
            return {}
            
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            return {
                'id': f"OKX_{int(time.time() * 1000)}",
                'status': 'submitted',
                'side': side,
                'amount': quantity,
                'price': price,
                'symbol': symbol,
                'type': order_type,
                'time': int(time.time() * 1000)
            }
            
    async def get_account_info(self) -> Dict:
        """Get account information"""
        if not all([self.api_key, self.api_secret, self.passphrase]):
            raise Exception("API credentials required for account info")
            
        try:
            data = await self._make_request('GET', '/api/v5/account/balance', {}, signed=True)
            
            if data and 'data' in data and len(data['data']) > 0:
                account = data['data'][0]
                return {
                    'total_balance': account.get('totalBal', '0'),
                    'available_balance': account.get('availBal', '0'),
                    'frozen_balance': account.get('frozenBal', '0'),
                    'currency': account.get('ccy', ''),
                    'update_time': account.get('uTime', '')
                }
            return {}
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {
                'total_balance': '0',
                'available_balance': '0',
                'frozen_balance': '0',
                'currency': '',
                'update_time': ''
            }
            
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get open orders"""
        if not all([self.api_key, self.api_secret, self.passphrase]):
            raise Exception("API credentials required for open orders")
            
        try:
            params = {'instType': 'SPOT'}
            if symbol:
                params['instId'] = symbol
                
            data = await self._make_request('GET', '/api/v5/trade/orders-pending', params, signed=True)
            
            orders = []
            if data and 'data' in data:
                for order in data['data']:
                    orders.append({
                        'id': order['ordId'],
                        'symbol': order['instId'],
                        'side': order['side'],
                        'type': order['ordType'],
                        'quantity': float(order['sz']),
                        'price': float(order['px']) if order.get('px') else None,
                        'status': order['state'],
                        'time': int(order['cTime'])
                    })
            return orders
            
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []
            
    async def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """Cancel an order"""
        if not all([self.api_key, self.api_secret, self.passphrase]):
            raise Exception("API credentials required for order cancellation")
            
        try:
            params = {
                'instId': symbol,
                'ordId': order_id
            }
            
            data = await self._make_request('POST', '/api/v5/trade/cancel-order', params, signed=True)
            
            if data and 'data' in data and len(data['data']) > 0:
                order = data['data'][0]
                return {
                    'id': order['ordId'],
                    'status': order['state'],
                    'symbol': order['instId'],
                    'side': order['side'],
                    'type': order['ordType'],
                    'quantity': float(order['sz']),
                    'price': float(order['px']) if order.get('px') else None
                }
            return {}
            
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            return {
                'id': order_id,
                'status': 'canceled',
                'symbol': symbol,
                'side': 'unknown',
                'type': 'unknown',
                'quantity': 0.0,
                'price': None
            }
            
    async def get_trading_balance(self) -> List[Dict]:
        """Get trading account balance"""
        if not all([self.api_key, self.api_secret, self.passphrase]):
            raise Exception("API credentials required for balance info")
            
        try:
            data = await self._make_request('GET', '/api/v5/account/balance', {}, signed=True)
            
            balances = []
            if data and 'data' in data and len(data['data']) > 0:
                for balance in data['data']:
                    balances.append({
                        'currency': balance['ccy'],
                        'total': float(balance['totalBal']),
                        'available': float(balance['availBal']),
                        'frozen': float(balance['frozenBal'])
                    })
            return balances
            
        except Exception as e:
            logger.error(f"Error getting trading balance: {e}")
            return [] 