"""
RSS Feed Collector for Crypto News Sentiment Analysis

TIER 1: Essential Market-Moving Feeds (Priority)
TIER 2: Supplementary Feeds (Secondary)

Features:
- Real-time news collection from crypto RSS feeds
- Sentiment analysis using multiple approaches
- Crypto relevance scoring
- Caching to avoid rate limiting
- Error handling and graceful degradation
"""

import feedparser
import requests
import asyncio
import aiohttp
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from collections import Counter
import time

@dataclass
class NewsArticle:
    """News article data structure."""
    source: str
    title: str
    link: str
    published: str
    summary: str
    sentiment: str
    relevance_score: int
    crypto_mentions: List[str]
    timestamp: datetime
    article_hash: str

class CryptoRSSCollector:
    """
    Advanced RSS feed collector for cryptocurrency news.
    
    Features:
    - Prioritized feed collection (Tier 1 -> Tier 2)
    - Advanced sentiment analysis with multiple keywords
    - Crypto relevance scoring
    - Intelligent caching with TTL
    - Rate limiting and error handling
    - Duplicate detection
    """
    
    def __init__(self, cache_dir: str = "cache"):
        """Initialize the RSS collector."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Priority feeds - implement first
        self.priority_feeds = {
            'coindesk_main': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
            'coindesk_markets': 'https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml&tag=markets',
            'cointelegraph_main': 'https://cointelegraph.com/rss',
            'theblock': 'https://www.theblock.co/rss.xml'
        }
        
        # Secondary feeds - add after Tier 1 works
        self.secondary_feeds = {
            'decrypt': 'https://decrypt.co/feed',
            'cryptoslate': 'https://cryptoslate.com/feed/',
            'cointelegraph_markets': 'https://cointelegraph.com/rss/category/market-analysis',
            'beincrypto': 'https://beincrypto.com/feed/'
        }
        
        # Cache settings
        self.cache_duration = 300  # 5 minutes
        self.max_cache_size = 1000
        
        # Sentiment keywords
        self.bullish_keywords = {
            'strong': ['surge', 'rally', 'bullish', 'soar', 'skyrocket', 'moon', 'bull run', 'breakout'],
            'medium': ['gains', 'rises', 'up', 'green', 'positive', 'growth', 'increase', 'pump'],
            'weak': ['recovery', 'rebound', 'support', 'holding', 'stable']
        }
        
        self.bearish_keywords = {
            'strong': ['crash', 'plunge', 'collapse', 'dump', 'bloodbath', 'capitulation', 'bear market'],
            'medium': ['falls', 'drops', 'decline', 'down', 'red', 'negative', 'loss', 'sell-off'],
            'weak': ['correction', 'dip', 'pullback', 'resistance', 'consolidation']
        }
        
        # Crypto keywords for relevance scoring
        self.crypto_keywords = {
            'major': ['bitcoin', 'btc', 'ethereum', 'eth', 'solana', 'sol', 'crypto', 'cryptocurrency'],
            'defi': ['defi', 'decentralized finance', 'yield farming', 'liquidity', 'staking'],
            'trading': ['trading', 'exchange', 'binance', 'coinbase', 'okx', 'hyperliquid'],
            'meme': ['meme coin', 'pepe', 'shib', 'doge', 'hype', 'memecoin'],
            'technical': ['blockchain', 'smart contract', 'nft', 'web3', 'dao']
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Article cache
        self.article_cache = {}
        self.seen_articles = set()
        
        self.logger.info("📰 RSS Collector initialized with priority feeds")
    
    async def collect_latest_news(self, hours_back: int = 6, use_secondary: bool = False) -> List[NewsArticle]:
        """
        Collect latest news from RSS feeds.
        
        Args:
            hours_back: How many hours back to collect news
            use_secondary: Whether to include secondary feeds
            
        Returns:
            List of NewsArticle objects
        """
        start_time = time.time()
        all_articles = []
        
        # Start with priority feeds
        feeds_to_use = self.priority_feeds.copy()
        
        # Add secondary feeds if requested
        if use_secondary:
            feeds_to_use.update(self.secondary_feeds)
        
        self.logger.info(f"🔄 Collecting news from {len(feeds_to_use)} feeds ({hours_back}h back)")
        
        # Collect from all feeds concurrently
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            tasks = []
            for source, url in feeds_to_use.items():
                task = self._collect_from_feed(session, source, url, hours_back)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"❌ Feed collection error: {result}")
                continue
            
            if isinstance(result, list):
                all_articles.extend(result)
        
        # Remove duplicates
        unique_articles = self._remove_duplicates(all_articles)
        
        # Sort by relevance and timestamp
        unique_articles.sort(key=lambda x: (x.relevance_score, x.timestamp), reverse=True)
        
        collection_time = time.time() - start_time
        self.logger.info(f"✅ Collected {len(unique_articles)} unique articles in {collection_time:.2f}s")
        
        return unique_articles
    
    async def _collect_from_feed(self, session: aiohttp.ClientSession, source: str, 
                                url: str, hours_back: int) -> List[NewsArticle]:
        """Collect articles from a single RSS feed."""
        try:
            # Check cache first
            cached_articles = self._get_cached_articles(source)
            if cached_articles:
                self.logger.debug(f"📋 Using cached articles for {source}")
                return cached_articles
            
            # Fetch RSS feed
            async with session.get(url) as response:
                if response.status != 200:
                    self.logger.warning(f"⚠️ Feed {source} returned status {response.status}")
                    return []
                
                content = await response.text()
            
            # Parse RSS feed
            feed = feedparser.parse(content)
            
            if not feed.entries:
                self.logger.warning(f"⚠️ No entries found in feed {source}")
                return []
            
            articles = []
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
            
            for entry in feed.entries:
                try:
                    # Parse publication date
                    published_str = str(entry.get('published', ''))
                    pub_date = self._parse_date(published_str)
                    if pub_date and pub_date.tzinfo is None:
                        # Make naive datetime timezone-aware (assume UTC)
                        pub_date = pub_date.replace(tzinfo=timezone.utc)
                    if pub_date and pub_date < cutoff_time:
                        continue
                    
                    # Extract article data
                    title = str(entry.get('title', ''))
                    summary = str(entry.get('summary', entry.get('description', '')))
                    link = str(entry.get('link', ''))
                    
                    # Skip if missing essential data
                    if not title or not link:
                        continue
                    
                    # Create article hash for duplicate detection
                    article_hash = hashlib.md5(f"{title}{link}".encode()).hexdigest()
                    
                    # Skip if already seen
                    if article_hash in self.seen_articles:
                        continue
                    
                    # Analyze sentiment
                    sentiment = self._analyze_sentiment(title, summary)
                    
                    # Check crypto relevance
                    relevance_score, crypto_mentions = self._check_crypto_relevance(title, summary)
                    
                    # Create article object
                    article = NewsArticle(
                        source=source,
                        title=title,
                        link=link,
                        published=published_str,
                        summary=summary,
                        sentiment=sentiment,
                        relevance_score=relevance_score,
                        crypto_mentions=crypto_mentions,
                        timestamp=datetime.now(),
                        article_hash=article_hash
                    )
                    
                    articles.append(article)
                    self.seen_articles.add(article_hash)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error processing entry from {source}: {e}")
                    continue
            
            # Cache articles
            self._cache_articles(source, articles)
            
            self.logger.info(f"📰 Collected {len(articles)} articles from {source}")
            return articles
            
        except Exception as e:
            self.logger.error(f"❌ Error collecting from {source}: {e}")
            return []
    
    def _analyze_sentiment(self, title: str, summary: str) -> str:
        """
        Analyze sentiment using weighted keyword approach.
        
        Args:
            title: Article title
            summary: Article summary
            
        Returns:
            Sentiment: 'bullish', 'bearish', or 'neutral'
        """
        text = f"{title} {summary}".lower()
        
        # Calculate weighted sentiment scores
        bullish_score = 0
        bearish_score = 0
        
        # Strong keywords get higher weights
        for keyword in self.bullish_keywords['strong']:
            if keyword in text:
                bullish_score += 3
        
        for keyword in self.bullish_keywords['medium']:
            if keyword in text:
                bullish_score += 2
        
        for keyword in self.bullish_keywords['weak']:
            if keyword in text:
                bullish_score += 1
        
        # Same for bearish keywords
        for keyword in self.bearish_keywords['strong']:
            if keyword in text:
                bearish_score += 3
        
        for keyword in self.bearish_keywords['medium']:
            if keyword in text:
                bearish_score += 2
        
        for keyword in self.bearish_keywords['weak']:
            if keyword in text:
                bearish_score += 1
        
        # Determine sentiment
        if bullish_score > bearish_score and bullish_score > 0:
            return 'bullish'
        elif bearish_score > bullish_score and bearish_score > 0:
            return 'bearish'
        else:
            return 'neutral'
    
    def _check_crypto_relevance(self, title: str, summary: str) -> tuple[int, List[str]]:
        """
        Check crypto relevance and return score with mentioned cryptos.
        
        Args:
            title: Article title
            summary: Article summary
            
        Returns:
            Tuple of (relevance_score, crypto_mentions)
        """
        text = f"{title} {summary}".lower()
        relevance_score = 0
        crypto_mentions = []
        
        # Check for crypto mentions with different weights
        for category, keywords in self.crypto_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    if category == 'major':
                        relevance_score += 5
                    elif category == 'defi':
                        relevance_score += 3
                    elif category == 'trading':
                        relevance_score += 2
                    elif category == 'meme':
                        relevance_score += 4  # High relevance for meme coins
                    else:
                        relevance_score += 1
                    
                    crypto_mentions.append(keyword)
        
        # Bonus for SOL mentions (our primary target)
        if 'solana' in text or 'sol' in text:
            relevance_score += 10
        
        # Bonus for HYPE mentions
        if 'hype' in text:
            relevance_score += 8
        
        return relevance_score, list(set(crypto_mentions))
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse various date formats from RSS feeds."""
        if not date_str:
            return None
        
        # Common RSS date formats
        formats = [
            '%a, %d %b %Y %H:%M:%S %z',
            '%a, %d %b %Y %H:%M:%S %Z',
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M:%S'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # If all formats fail, return None
        return None
    
    def _remove_duplicates(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on hash."""
        seen_hashes = set()
        unique_articles = []
        
        for article in articles:
            if article.article_hash not in seen_hashes:
                unique_articles.append(article)
                seen_hashes.add(article.article_hash)
        
        return unique_articles
    
    def _get_cached_articles(self, source: str) -> Optional[List[NewsArticle]]:
        """Get cached articles for a source if still valid."""
        cache_file = self.cache_dir / f"{source}_cache.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache is still valid
            cache_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cache_time > timedelta(seconds=self.cache_duration):
                return None
            
            # Convert cached data back to NewsArticle objects
            articles = []
            for article_data in cache_data['articles']:
                article = NewsArticle(
                    source=article_data['source'],
                    title=article_data['title'],
                    link=article_data['link'],
                    published=article_data['published'],
                    summary=article_data['summary'],
                    sentiment=article_data['sentiment'],
                    relevance_score=article_data['relevance_score'],
                    crypto_mentions=article_data['crypto_mentions'],
                    timestamp=datetime.fromisoformat(article_data['timestamp']),
                    article_hash=article_data['article_hash']
                )
                articles.append(article)
            
            return articles
            
        except Exception as e:
            self.logger.error(f"❌ Error loading cache for {source}: {e}")
            return None
    
    def _cache_articles(self, source: str, articles: List[NewsArticle]):
        """Cache articles for a source."""
        cache_file = self.cache_dir / f"{source}_cache.json"
        
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'articles': [
                    {
                        'source': article.source,
                        'title': article.title,
                        'link': article.link,
                        'published': article.published,
                        'summary': article.summary,
                        'sentiment': article.sentiment,
                        'relevance_score': article.relevance_score,
                        'crypto_mentions': article.crypto_mentions,
                        'timestamp': article.timestamp.isoformat(),
                        'article_hash': article.article_hash
                    }
                    for article in articles
                ]
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"❌ Error caching articles for {source}: {e}")
    
    def get_sentiment_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get sentiment summary from recent articles."""
        try:
            # This would typically read from a database or cache
            # For now, return a mock summary
            return {
                'timeframe': f"{hours}h",
                'total_articles': 0,
                'sentiment_breakdown': {
                    'bullish': 0,
                    'bearish': 0,
                    'neutral': 0
                },
                'top_cryptos_mentioned': [],
                'average_relevance': 0.0,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"❌ Error getting sentiment summary: {e}")
            return {'error': str(e)}
    
    def cleanup_cache(self, max_age_days: int = 7):
        """Clean up old cache files."""
        try:
            cutoff_time = datetime.now() - timedelta(days=max_age_days)
            
            for cache_file in self.cache_dir.glob("*_cache.json"):
                try:
                    file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if file_time < cutoff_time:
                        cache_file.unlink()
                        self.logger.info(f"🧹 Cleaned up old cache file: {cache_file.name}")
                except Exception as e:
                    self.logger.error(f"❌ Error cleaning cache file {cache_file}: {e}")
                    
        except Exception as e:
            self.logger.error(f"❌ Error during cache cleanup: {e}")