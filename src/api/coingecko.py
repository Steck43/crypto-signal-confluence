"""
CoinGecko API Client for Cryptocurrency Data
Provides market data, price feeds, and meme coin information
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class CoinGeckoMarketData:
    """Structured market data from CoinGecko"""
    id: str
    symbol: str
    name: str
    current_price: float
    market_cap: float
    market_cap_rank: int
    total_volume: float
    high_24h: float
    low_24h: float
    price_change_24h: float
    price_change_percentage_24h: float
    market_cap_change_24h: float
    market_cap_change_percentage_24h: float
    circulating_supply: float
    total_supply: float
    max_supply: Optional[float]
    ath: float
    ath_change_percentage: float
    ath_date: str
    atl: float
    atl_change_percentage: float
    atl_date: str
    last_updated: str
    sparkline_in_7d: Optional[Dict] = None

class CoinGeckoClient:
    """Professional CoinGecko API client with rate limiting and error handling"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.coingecko.com/api/v3"):
        self.api_key = api_key
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit_remaining = 50  # CoinGecko free tier: 50 calls/minute
        self.rate_limit_reset = time.time() + 60
        self.last_request_time = 0
        self.min_request_interval = 1.2  # Minimum 1.2 seconds between requests
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
        
    async def connect(self):
        """Initialize HTTP session"""
        if self.session is None:
            headers = {
                'User-Agent': 'Institutional-Trading-System/1.0',
                'Accept': 'application/json'
            }
            if self.api_key:
                headers['X-CG-API-KEY'] = self.api_key
                
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            )
            logger.info("✅ CoinGecko client connected")
            
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("🔌 CoinGecko client disconnected")
            
    async def _rate_limit_check(self):
        """Check and enforce rate limits"""
        current_time = time.time()
        
        # Reset rate limit if time has passed
        if current_time > self.rate_limit_reset:
            self.rate_limit_remaining = 50
            self.rate_limit_reset = current_time + 60
            
        # Check if we have remaining calls
        if self.rate_limit_remaining <= 0:
            wait_time = self.rate_limit_reset - current_time
            if wait_time > 0:
                logger.warning(f"Rate limit exceeded, waiting {wait_time:.1f} seconds")
                await asyncio.sleep(wait_time)
                self.rate_limit_remaining = 50
                
        # Enforce minimum interval between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
            
        self.last_request_time = time.time()
        
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make authenticated API request with error handling"""
        await self._rate_limit_check()
        
        if not self.session:
            await self.connect()
            
        url = f"{self.base_url}/{endpoint}"
        
        try:
            async with self.session.get(url, params=params) as response:
                self.rate_limit_remaining -= 1
                
                if response.status == 200:
                    data = await response.json()
                    return data
                elif response.status == 429:
                    logger.warning("Rate limit hit, implementing exponential backoff")
                    await asyncio.sleep(60)  # Wait 1 minute
                    return await self._make_request(endpoint, params)
                elif response.status == 401:
                    logger.error("CoinGecko API key invalid or expired")
                    raise Exception("Invalid API key")
                elif response.status == 404:
                    logger.error(f"Endpoint not found: {endpoint}")
                    raise Exception("Endpoint not found")
                else:
                    logger.error(f"CoinGecko API error {response.status}: {await response.text()}")
                    raise Exception(f"API error {response.status}")
                    
        except aiohttp.ClientError as e:
            logger.error(f"CoinGecko network error: {e}")
            raise Exception(f"Network error: {e}")
        except Exception as e:
            logger.error(f"CoinGecko request error: {e}")
            raise
            
    async def get_coin_market_data(self, coin_id: str, vs_currency: str = 'usd') -> Optional[CoinGeckoMarketData]:
        """Get comprehensive market data for a specific coin"""
        try:
            params = {
                'ids': coin_id,
                'vs_currencies': vs_currency,
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true',
                'include_last_updated_at': 'true',
                'include_sparkline': 'true'
            }
            
            data = await self._make_request('coins/markets', params)
            
            if data and len(data) > 0:
                coin_data = data[0]
                return CoinGeckoMarketData(
                    id=coin_data.get('id', ''),
                    symbol=coin_data.get('symbol', '').upper(),
                    name=coin_data.get('name', ''),
                    current_price=coin_data.get('current_price', 0.0),
                    market_cap=coin_data.get('market_cap', 0.0),
                    market_cap_rank=coin_data.get('market_cap_rank', 0),
                    total_volume=coin_data.get('total_volume', 0.0),
                    high_24h=coin_data.get('high_24h', 0.0),
                    low_24h=coin_data.get('low_24h', 0.0),
                    price_change_24h=coin_data.get('price_change_24h', 0.0),
                    price_change_percentage_24h=coin_data.get('price_change_percentage_24h', 0.0),
                    market_cap_change_24h=coin_data.get('market_cap_change_24h', 0.0),
                    market_cap_change_percentage_24h=coin_data.get('market_cap_change_percentage_24h', 0.0),
                    circulating_supply=coin_data.get('circulating_supply', 0.0),
                    total_supply=coin_data.get('total_supply', 0.0),
                    max_supply=coin_data.get('max_supply'),
                    ath=coin_data.get('ath', 0.0),
                    ath_change_percentage=coin_data.get('ath_change_percentage', 0.0),
                    ath_date=coin_data.get('ath_date', ''),
                    atl=coin_data.get('atl', 0.0),
                    atl_change_percentage=coin_data.get('atl_change_percentage', 0.0),
                    atl_date=coin_data.get('atl_date', ''),
                    last_updated=coin_data.get('last_updated', ''),
                    sparkline_in_7d=coin_data.get('sparkline_in_7d')
                )
            return None
            
        except Exception as e:
            logger.error(f"Error getting market data for {coin_id}: {e}")
            return None
            
    async def get_trending_coins(self) -> List[Dict]:
        """Get trending coins in the last 24 hours"""
        try:
            data = await self._make_request('search/trending')
            
            if data and 'coins' in data:
                trending_coins = []
                for coin in data['coins']:
                    trending_coins.append({
                        'id': coin['item']['id'],
                        'name': coin['item']['name'],
                        'symbol': coin['item']['symbol'].upper(),
                        'market_cap_rank': coin['item']['market_cap_rank'],
                        'price_btc': coin['item']['price_btc'],
                        'score': coin['item']['score']
                    })
                return trending_coins
            return []
            
        except Exception as e:
            logger.error(f"Error getting trending coins: {e}")
            return []
            
    async def get_meme_coins(self, limit: int = 20) -> List[CoinGeckoMarketData]:
        """Get top meme coins by market cap"""
        try:
            # Get top coins and filter for meme coins
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': limit * 2,  # Get more to filter
                'page': 1,
                'sparkline': 'false'
            }
            
            data = await self._make_request('coins/markets', params)
            
            if data:
                # Filter for potential meme coins (lower market cap, higher volatility)
                meme_coins = []
                for coin in data:
                    if (coin.get('market_cap', 0) < 1e10 and  # Less than $10B market cap
                        abs(coin.get('price_change_percentage_24h', 0)) > 5):  # High volatility
                        meme_coins.append(CoinGeckoMarketData(
                            id=coin.get('id', ''),
                            symbol=coin.get('symbol', '').upper(),
                            name=coin.get('name', ''),
                            current_price=coin.get('current_price', 0.0),
                            market_cap=coin.get('market_cap', 0.0),
                            market_cap_rank=coin.get('market_cap_rank', 0),
                            total_volume=coin.get('total_volume', 0.0),
                            high_24h=coin.get('high_24h', 0.0),
                            low_24h=coin.get('low_24h', 0.0),
                            price_change_24h=coin.get('price_change_24h', 0.0),
                            price_change_percentage_24h=coin.get('price_change_percentage_24h', 0.0),
                            market_cap_change_24h=coin.get('market_cap_change_24h', 0.0),
                            market_cap_change_percentage_24h=coin.get('market_cap_change_percentage_24h', 0.0),
                            circulating_supply=coin.get('circulating_supply', 0.0),
                            total_supply=coin.get('total_supply', 0.0),
                            max_supply=coin.get('max_supply'),
                            ath=coin.get('ath', 0.0),
                            ath_change_percentage=coin.get('ath_change_percentage', 0.0),
                            ath_date=coin.get('ath_date', ''),
                            atl=coin.get('atl', 0.0),
                            atl_change_percentage=coin.get('atl_change_percentage', 0.0),
                            atl_date=coin.get('atl_date', ''),
                            last_updated=coin.get('last_updated', '')
                        ))
                        if len(meme_coins) >= limit:
                            break
                            
                return meme_coins
            return []
            
        except Exception as e:
            logger.error(f"Error getting meme coins: {e}")
            return []
            
    async def get_coin_price_history(self, coin_id: str, days: int = 30, vs_currency: str = 'usd') -> Optional[Dict]:
        """Get historical price data for a coin"""
        try:
            params = {
                'vs_currency': vs_currency,
                'days': days,
                'interval': 'daily'
            }
            
            data = await self._make_request(f'coins/{coin_id}/market_chart', params)
            
            if data and 'prices' in data:
                return {
                    'prices': data['prices'],
                    'market_caps': data.get('market_caps', []),
                    'total_volumes': data.get('total_volumes', [])
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting price history for {coin_id}: {e}")
            return None
            
    async def search_coins(self, query: str) -> List[Dict]:
        """Search for coins by name or symbol"""
        try:
            data = await self._make_request('search', {'query': query})
            
            if data and 'coins' in data:
                return data['coins'][:10]  # Return top 10 results
            return []
            
        except Exception as e:
            logger.error(f"Error searching coins: {e}")
            return []
            
    async def get_exchange_rates(self, base_currency: str = 'usd') -> Dict:
        """Get exchange rates for various cryptocurrencies"""
        try:
            data = await self._make_request('exchange_rates')
            
            if data and 'rates' in data:
                return data['rates']
            return {}
            
        except Exception as e:
            logger.error(f"Error getting exchange rates: {e}")
            return {}
            
    async def get_global_market_data(self) -> Optional[Dict]:
        """Get global cryptocurrency market data"""
        try:
            data = await self._make_request('global')
            
            if data and 'data' in data:
                return data['data']
            return None
            
        except Exception as e:
            logger.error(f"Error getting global market data: {e}")
            return None

# Factory function for easy instantiation
def get_coingecko_client(api_key: Optional[str] = None) -> CoinGeckoClient:
    """Get CoinGecko client instance"""
    return CoinGeckoClient(api_key=api_key) 