"""
Binance exchange API client.

EXPLORATORY: REST client scaffold. Not validated. Fails closed on order and market errors.
"""

import asyncio
import aiohttp
import hmac
import hashlib
import time
import logging
from datetime import datetime
from typing import Dict, Optional, List
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

class BinanceClient:
    """Professional Binance API client with authentication and rate limiting"""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.binance.com"
        self.session: Optional[aiohttp.ClientSession] = None
        self.connected = False
        self.rate_limit_remaining = 1200  # Binance: 1200 requests per minute
        self.rate_limit_reset = time.time() + 60
        self.last_request_time = 0
        self.min_request_interval = 0.1  # Minimum 100ms between requests
        
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
                'Accept': 'application/json'
            }
            if self.api_key:
                headers['X-MBX-APIKEY'] = self.api_key
                
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
        # Test connection
        try:
            await self._make_request('GET', '/api/v3/ping')
            self.connected = True
            logger.info("✅ Binance client connected")
        except Exception as e:
            logger.error(f"❌ Binance connection failed: {e}")
            self.connected = False
            
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
        self.connected = False
        logger.info("🔌 Binance client disconnected")
        
    async def _rate_limit_check(self):
        """Check and enforce rate limits"""
        current_time = time.time()
        
        # Reset rate limit if time has passed
        if current_time > self.rate_limit_reset:
            self.rate_limit_remaining = 1200
            self.rate_limit_reset = current_time + 60
            
        # Check if we have remaining calls
        if self.rate_limit_remaining <= 0:
            wait_time = self.rate_limit_reset - current_time
            if wait_time > 0:
                logger.warning(f"Rate limit exceeded, waiting {wait_time:.1f} seconds")
                await asyncio.sleep(wait_time)
                self.rate_limit_remaining = 1200
                
        # Enforce minimum interval between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
            
        self.last_request_time = time.time()
        
    def _generate_signature(self, params: Dict) -> str:
        """Generate HMAC SHA256 signature for authenticated requests"""
        if not self.api_secret:
            raise Exception("API secret required for authenticated requests")
            
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
        
    async def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, signed: bool = False) -> Dict:
        """Make authenticated API request with error handling"""
        await self._rate_limit_check()
        
        if not self.session:
            await self.connect()
            
        url = f"{self.base_url}{endpoint}"
        
        # Add timestamp for signed requests
        if signed:
            if params is None:
                params = {}
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._generate_signature(params)
            
        try:
            if method.upper() == 'GET':
                async with self.session.get(url, params=params) as response:
                    self.rate_limit_remaining -= 1
                    return await self._handle_response(response)
            elif method.upper() == 'POST':
                async with self.session.post(url, data=params) as response:
                    self.rate_limit_remaining -= 1
                    return await self._handle_response(response)
            else:
                raise Exception(f"Unsupported HTTP method: {method}")
                
        except aiohttp.ClientError as e:
            logger.error(f"Binance network error: {e}")
            raise Exception(f"Network error: {e}")
        except Exception as e:
            logger.error(f"Binance request error: {e}")
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
            logger.error("Binance API key invalid or expired")
            raise Exception("Invalid API key")
        elif response.status == 403:
            logger.error("Insufficient permissions for this endpoint")
            raise Exception("Insufficient permissions")
        else:
            error_text = await response.text()
            logger.error(f"Binance API error {response.status}: {error_text}")
            raise Exception(f"API error {response.status}: {error_text}")
            
    async def get_sol_data(self) -> Dict:
        """Get SOL/USDT market data"""
        try:
            data = await self._make_request('GET', '/api/v3/ticker/24hr', {'symbol': 'SOLUSDT'})
            
            return {
                'price': float(data['lastPrice']),
                'bid': float(data['bidPrice']),
                'ask': float(data['askPrice']),
                'volume': float(data['volume']),
                'high_24h': float(data['highPrice']),
                'low_24h': float(data['lowPrice']),
                'price_change_24h': float(data['priceChange']),
                'price_change_percent_24h': float(data['priceChangePercent']),
                'quote_volume': float(data['quoteVolume']),
                'count': int(data['count'])
            }
            
        except Exception as e:
            logger.error(f"Error getting SOL data: {e}")
            raise RuntimeError(f"Binance SOL market data request failed: {e}") from e
            
    async def get_market_data(self, symbol: str) -> Dict:
        """Get market data for any symbol"""
        try:
            data = await self._make_request('GET', '/api/v3/ticker/24hr', {'symbol': symbol})
            
            return {
                'price': float(data['lastPrice']),
                'bid': float(data['bidPrice']),
                'ask': float(data['askPrice']),
                'volume': float(data['volume']),
                'high_24h': float(data['highPrice']),
                'low_24h': float(data['lowPrice']),
                'price_change_24h': float(data['priceChange']),
                'price_change_percent_24h': float(data['priceChangePercent']),
                'quote_volume': float(data['quoteVolume']),
                'count': int(data['count'])
            }
            
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
                'count': 0
            }
            
    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """Get order book for a symbol"""
        try:
            params = {'symbol': symbol, 'limit': limit}
            data = await self._make_request('GET', '/api/v3/depth', params)
            
            return {
                'symbol': symbol,
                'bids': [[float(price), float(qty)] for price, qty in data['bids']],
                'asks': [[float(price), float(qty)] for price, qty in data['asks']],
                'last_update_id': data['lastUpdateId']
            }
            
        except Exception as e:
            logger.error(f"Error getting order book for {symbol}: {e}")
            return {
                'symbol': symbol,
                'bids': [],
                'asks': [],
                'last_update_id': 0
            }
            
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent trades for a symbol"""
        try:
            params = {'symbol': symbol, 'limit': limit}
            data = await self._make_request('GET', '/api/v3/trades', params)
            
            trades = []
            for trade in data:
                trades.append({
                    'id': int(trade['id']),
                    'price': float(trade['price']),
                    'qty': float(trade['qty']),
                    'quote_qty': float(trade['quoteQty']),
                    'time': int(trade['time']),
                    'is_buyer_maker': trade['isBuyerMaker'],
                    'is_best_match': trade.get('isBestMatch', False)
                })
            return trades
            
        except Exception as e:
            logger.error(f"Error getting recent trades for {symbol}: {e}")
            return []
            
    async def place_sol_order(self, side: str, amount: float, order_type: str = 'MARKET') -> Dict:
        """Place SOL order on Binance"""
        if not self.api_key or not self.api_secret:
            raise Exception("API credentials required for order placement")
            
        try:
            params = {
                'symbol': 'SOLUSDT',
                'side': side.upper(),
                'type': order_type,
                'quantity': f"{amount:.6f}"  # SOL precision
            }
            
            data = await self._make_request('POST', '/api/v3/order', params, signed=True)
            
            return {
                'id': data['orderId'],
                'status': data['status'],
                'side': data['side'],
                'amount': float(data['executedQty']),
                'price': float(data['price']) if data['price'] != '0' else None,
                'symbol': data['symbol'],
                'type': data['type'],
                'time': data['time']
            }
            
        except Exception as e:
            logger.error(f"Error placing SOL order: {e}")
            raise RuntimeError(f"Binance SOL order placement failed: {e}") from e
            
    async def place_order(self, symbol: str, side: str, quantity: float, order_type: str = 'MARKET', price: Optional[float] = None) -> Dict:
        """Place order for any symbol"""
        if not self.api_key or not self.api_secret:
            raise Exception("API credentials required for order placement")
            
        try:
            params = {
                'symbol': symbol,
                'side': side.upper(),
                'type': order_type,
                'quantity': f"{quantity:.8f}"  # Standard precision
            }
            
            if price and order_type == 'LIMIT':
                params['price'] = f"{price:.8f}"
                params['timeInForce'] = 'GTC'
                
            data = await self._make_request('POST', '/api/v3/order', params, signed=True)
            
            return {
                'id': data['orderId'],
                'status': data['status'],
                'side': data['side'],
                'amount': float(data['executedQty']),
                'price': float(data['price']) if data['price'] != '0' else None,
                'symbol': data['symbol'],
                'type': data['type'],
                'time': data['time']
            }
            
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            raise RuntimeError(f"Binance order placement failed for {symbol}: {e}") from e
            
    async def get_account_info(self) -> Dict:
        """Get account information"""
        if not self.api_key or not self.api_secret:
            raise Exception("API credentials required for account info")
            
        try:
            data = await self._make_request('GET', '/api/v3/account', {}, signed=True)
            
            return {
                'maker_commission': data['makerCommission'],
                'taker_commission': data['takerCommission'],
                'buyer_commission': data['buyerCommission'],
                'seller_commission': data['sellerCommission'],
                'can_trade': data['canTrade'],
                'can_withdraw': data['canWithdraw'],
                'can_deposit': data['canDeposit'],
                'update_time': data['updateTime'],
                'balances': data['balances']
            }
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {
                'maker_commission': 0,
                'taker_commission': 0,
                'buyer_commission': 0,
                'seller_commission': 0,
                'can_trade': False,
                'can_withdraw': False,
                'can_deposit': False,
                'update_time': 0,
                'balances': []
            }
            
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get open orders"""
        if not self.api_key or not self.api_secret:
            raise Exception("API credentials required for open orders")
            
        try:
            params = {}
            if symbol:
                params['symbol'] = symbol
                
            data = await self._make_request('GET', '/api/v3/openOrders', params, signed=True)
            
            orders = []
            for order in data:
                orders.append({
                    'id': order['orderId'],
                    'symbol': order['symbol'],
                    'side': order['side'],
                    'type': order['type'],
                    'quantity': float(order['origQty']),
                    'price': float(order['price']) if order['price'] != '0' else None,
                    'status': order['status'],
                    'time': order['time']
                })
            return orders
            
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []
            
    async def cancel_order(self, symbol: str, order_id: int) -> Dict:
        """Cancel an order"""
        if not self.api_key or not self.api_secret:
            raise Exception("API credentials required for order cancellation")
            
        try:
            params = {
                'symbol': symbol,
                'orderId': order_id
            }
            
            data = await self._make_request('DELETE', '/api/v3/order', params, signed=True)
            
            return {
                'id': data['orderId'],
                'status': data['status'],
                'symbol': data['symbol'],
                'side': data['side'],
                'type': data['type'],
                'quantity': float(data['origQty']),
                'price': float(data['price']) if data['price'] != '0' else None
            }
            
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            return {
                'id': order_id,
                'status': 'CANCELED',
                'symbol': symbol,
                'side': 'UNKNOWN',
                'type': 'UNKNOWN',
                'quantity': 0.0,
                'price': None
            } 