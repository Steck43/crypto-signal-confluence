"""
FRED API Client for Economic Data
Provides Federal Reserve Economic Data for macro regime analysis
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time
import json

logger = logging.getLogger(__name__)

@dataclass
class FREDSeriesData:
    """Structured FRED series data"""
    series_id: str
    title: str
    units: str
    frequency: str
    seasonal_adjustment: str
    last_updated: str
    observations: List[Dict]
    current_value: Optional[float] = None
    previous_value: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None

class FREDClient:
    """Professional FRED API client with rate limiting and error handling"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.stlouisfed.org/fred"):
        self.api_key = api_key
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit_remaining = 120  # FRED: 120 calls per minute
        self.rate_limit_reset = time.time() + 60
        self.last_request_time = 0
        self.min_request_interval = 0.5  # Minimum 0.5 seconds between requests
        
        # Key economic series for macro analysis
        self.key_series = {
            'federal_funds_rate': 'FEDFUNDS',
            'inflation_rate': 'CPIAUCSL',
            'unemployment_rate': 'UNRATE',
            'gdp_growth': 'GDP',
            'vix': 'VIXCLS',
            'real_gdp': 'GDPC1',
            'personal_consumption': 'PCE',
            'industrial_production': 'INDPRO',
            'housing_starts': 'HOUST',
            'consumer_confidence': 'UMCSENT',
            'retail_sales': 'RSAFS',
            'manufacturing_pmi': 'NAPM',
            'treasury_yield_10y': 'GS10',
            'treasury_yield_2y': 'GS2',
            'dollar_index': 'DTWEXBGS',
            'oil_prices': 'DCOILWTICO'
        }
        
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
                
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            )
            logger.info("✅ FRED client connected")
            
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("🔌 FRED client disconnected")
            
    async def _rate_limit_check(self):
        """Check and enforce rate limits"""
        current_time = time.time()
        
        # Reset rate limit if time has passed
        if current_time > self.rate_limit_reset:
            self.rate_limit_remaining = 120
            self.rate_limit_reset = current_time + 60
            
        # Check if we have remaining calls
        if self.rate_limit_remaining <= 0:
            wait_time = self.rate_limit_reset - current_time
            if wait_time > 0:
                logger.warning(f"Rate limit exceeded, waiting {wait_time:.1f} seconds")
                await asyncio.sleep(wait_time)
                self.rate_limit_remaining = 120
                
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
            
        # Add API key to all requests
        if params is None:
            params = {}
        params['api_key'] = self.api_key
        params['file_type'] = 'json'
        
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
                    logger.error("FRED API key invalid or expired")
                    raise Exception("Invalid API key")
                elif response.status == 404:
                    logger.error(f"Endpoint not found: {endpoint}")
                    raise Exception("Endpoint not found")
                else:
                    logger.error(f"FRED API error {response.status}: {await response.text()}")
                    raise Exception(f"API error {response.status}")
                    
        except aiohttp.ClientError as e:
            logger.error(f"FRED network error: {e}")
            raise Exception(f"Network error: {e}")
        except Exception as e:
            logger.error(f"FRED request error: {e}")
            raise
            
    async def get_series(self, series_id: str, limit: int = 100) -> Optional[FREDSeriesData]:
        """Get economic series data"""
        try:
            params = {
                'series_id': series_id,
                'limit': limit,
                'sort_order': 'desc'
            }
            
            data = await self._make_request('series/observations', params)
            
            if data and 'observations' in data:
                observations = data['observations']
                
                # Get series metadata
                metadata = await self.get_series_info(series_id)
                
                # Calculate current and previous values
                current_value = None
                previous_value = None
                change = None
                change_percent = None
                
                if len(observations) >= 2:
                    try:
                        current_value = float(observations[0]['value'])
                        previous_value = float(observations[1]['value'])
                        change = current_value - previous_value
                        if previous_value != 0:
                            change_percent = (change / previous_value) * 100
                    except (ValueError, KeyError):
                        pass
                
                return FREDSeriesData(
                    series_id=series_id,
                    title=metadata.get('title', series_id) if metadata else series_id,
                    units=metadata.get('units', '') if metadata else '',
                    frequency=metadata.get('frequency', '') if metadata else '',
                    seasonal_adjustment=metadata.get('seasonal_adjustment', '') if metadata else '',
                    last_updated=metadata.get('last_updated', '') if metadata else '',
                    observations=observations,
                    current_value=current_value,
                    previous_value=previous_value,
                    change=change,
                    change_percent=change_percent
                )
            return None
            
        except Exception as e:
            logger.error(f"Error getting series {series_id}: {e}")
            return None
            
    async def get_series_info(self, series_id: str) -> Optional[Dict]:
        """Get series metadata"""
        try:
            params = {'series_id': series_id}
            data = await self._make_request('series', params)
            
            if data and 'seriess' in data and len(data['seriess']) > 0:
                return data['seriess'][0]
            return None
            
        except Exception as e:
            logger.error(f"Error getting series info for {series_id}: {e}")
            return None
            
    async def get_federal_funds_rate(self) -> Optional[Dict]:
        """Get Federal Funds Rate data"""
        try:
            series_data = await self.get_series('FEDFUNDS', limit=50)
            
            if series_data:
                return {
                    'current': series_data.current_value,
                    'previous': series_data.previous_value,
                    'change': series_data.change,
                    'change_percent': series_data.change_percent,
                    'historical_avg': self._calculate_historical_average(series_data.observations, 12),
                    'last_updated': series_data.last_updated
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting Federal Funds Rate: {e}")
            return None
            
    async def get_inflation_data(self) -> Optional[Dict]:
        """Get inflation rate data (CPI)"""
        try:
            series_data = await self.get_series('CPIAUCSL', limit=50)
            
            if series_data:
                # Calculate year-over-year inflation
                yoy_inflation = self._calculate_yoy_change(series_data.observations)
                
                return {
                    'current': yoy_inflation,
                    'historical_avg': self._calculate_historical_average(series_data.observations, 12),
                    'last_updated': series_data.last_updated
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting inflation data: {e}")
            return None
            
    async def get_unemployment_rate(self) -> Optional[Dict]:
        """Get unemployment rate data"""
        try:
            series_data = await self.get_series('UNRATE', limit=50)
            
            if series_data:
                return {
                    'current': series_data.current_value,
                    'previous': series_data.previous_value,
                    'change': series_data.change,
                    'historical_avg': self._calculate_historical_average(series_data.observations, 12),
                    'last_updated': series_data.last_updated
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting unemployment rate: {e}")
            return None
            
    async def get_gdp_data(self) -> Optional[Dict]:
        """Get GDP growth data"""
        try:
            series_data = await self.get_series('GDP', limit=50)
            
            if series_data:
                # Calculate quarter-over-quarter growth
                qoq_growth = self._calculate_qoq_change(series_data.observations)
                
                return {
                    'current': qoq_growth,
                    'historical_avg': self._calculate_historical_average(series_data.observations, 8),
                    'last_updated': series_data.last_updated
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting GDP data: {e}")
            return None
            
    async def get_vix_data(self) -> Optional[Dict]:
        """Get VIX volatility index data"""
        try:
            series_data = await self.get_series('VIXCLS', limit=50)
            
            if series_data:
                return {
                    'current': series_data.current_value,
                    'previous': series_data.previous_value,
                    'change': series_data.change,
                    'historical_avg': self._calculate_historical_average(series_data.observations, 30),
                    'last_updated': series_data.last_updated
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting VIX data: {e}")
            return None
            
    async def get_comprehensive_economic_data(self) -> Dict:
        """Get all key economic indicators for macro regime analysis"""
        try:
            economic_data = {}
            
            # Get all key indicators concurrently
            tasks = [
                self.get_federal_funds_rate(),
                self.get_inflation_data(),
                self.get_unemployment_rate(),
                self.get_gdp_data(),
                self.get_vix_data()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Map results to economic data
            indicators = ['federal_funds_rate', 'inflation_rate', 'unemployment_rate', 'gdp_growth', 'vix']
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error getting {indicators[i]}: {result}")
                elif result:
                    economic_data[indicators[i]] = result
                    
            return economic_data
            
        except Exception as e:
            logger.error(f"Error getting comprehensive economic data: {e}")
            return {}
            
    def _calculate_historical_average(self, observations: List[Dict], periods: int) -> float:
        """Calculate historical average over specified periods"""
        try:
            values = []
            for obs in observations[:periods]:
                try:
                    value = float(obs['value'])
                    if value != 0:  # Skip zero values
                        values.append(value)
                except (ValueError, KeyError):
                    continue
                    
            if values:
                return sum(values) / len(values)
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating historical average: {e}")
            return 0.0
            
    def _calculate_yoy_change(self, observations: List[Dict]) -> float:
        """Calculate year-over-year percentage change"""
        try:
            if len(observations) < 12:
                return 0.0
                
            current = float(observations[0]['value'])
            year_ago = float(observations[11]['value'])
            
            if year_ago != 0:
                return ((current - year_ago) / year_ago) * 100
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating YoY change: {e}")
            return 0.0
            
    def _calculate_qoq_change(self, observations: List[Dict]) -> float:
        """Calculate quarter-over-quarter percentage change"""
        try:
            if len(observations) < 4:
                return 0.0
                
            current = float(observations[0]['value'])
            quarter_ago = float(observations[3]['value'])
            
            if quarter_ago != 0:
                return ((current - quarter_ago) / quarter_ago) * 100
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating QoQ change: {e}")
            return 0.0
            
    async def search_series(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for economic series"""
        try:
            params = {
                'search_text': query,
                'limit': limit
            }
            
            data = await self._make_request('series/search', params)
            
            if data and 'seriess' in data:
                return data['seriess']
            return []
            
        except Exception as e:
            logger.error(f"Error searching series: {e}")
            return []
            
    async def get_category_series(self, category_id: int, limit: int = 10) -> List[Dict]:
        """Get series in a specific category"""
        try:
            params = {
                'category_id': category_id,
                'limit': limit
            }
            
            data = await self._make_request('category/series', params)
            
            if data and 'seriess' in data:
                return data['seriess']
            return []
            
        except Exception as e:
            logger.error(f"Error getting category series: {e}")
            return []

# Factory function for easy instantiation
def get_fred_client(api_key: str) -> FREDClient:
    """Get FRED client instance"""
    return FREDClient(api_key=api_key) 