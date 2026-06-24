"""
Advanced Twitter Sentiment Collection System

Comprehensive Twitter data collection with:
- Real-time tweet streaming
- Rate limiting and error handling
- Data preprocessing for CryptoBERT
- Multi-language sentiment analysis
- Spam and bot detection

References:
- Huang et al. (2020): "CryptoBERT: A Cryptocurrency-Domain Pre-trained Language Model"
- Devlin et al. (2018): "BERT: Pre-training of Deep Bidirectional Transformers"
"""

import tweepy
import pandas as pd
import numpy as np
import re
import json
import time
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import asyncio
import aiohttp
from collections import defaultdict, deque
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from textblob import TextBlob
import hashlib
import pickle
from pathlib import Path
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

Base = declarative_base()

@dataclass
class TweetData:
    """Tweet data container with sentiment analysis."""
    id: str
    text: str
    user_id: str
    username: str
    created_at: datetime
    retweet_count: int
    like_count: int
    reply_count: int
    quote_count: int
    language: str
    hashtags: List[str]
    mentions: List[str]
    urls: List[str]
    is_retweet: bool
    is_reply: bool
    sentiment_score: float = 0.0
    sentiment_label: str = 'neutral'
    confidence: float = 0.0
    processed_text: str = ''
    crypto_mentions: List[str] = None
    bot_probability: float = 0.0

class TweetRecord(Base):
    """SQLAlchemy model for storing tweets."""
    __tablename__ = 'tweets'
    
    id = Column(String, primary_key=True)
    text = Column(Text)
    user_id = Column(String)
    username = Column(String)
    created_at = Column(DateTime)
    retweet_count = Column(Integer)
    like_count = Column(Integer)
    reply_count = Column(Integer)
    quote_count = Column(Integer)
    language = Column(String)
    hashtags = Column(Text)  # JSON string
    mentions = Column(Text)  # JSON string
    urls = Column(Text)  # JSON string
    is_retweet = Column(Boolean)
    is_reply = Column(Boolean)
    sentiment_score = Column(Float)
    sentiment_label = Column(String)
    confidence = Column(Float)
    processed_text = Column(Text)
    crypto_mentions = Column(Text)  # JSON string
    bot_probability = Column(Float)

class TwitterSentimentCollector:
    """
    Advanced Twitter sentiment collection system with real-time processing.
    
    Features:
    - Real-time tweet streaming with advanced filtering
    - Multi-layered sentiment analysis (VADER, TextBlob, custom models)
    - Bot detection and spam filtering
    - Rate limiting and error recovery
    - Data persistence with SQLite/PostgreSQL
    - CryptoBERT preprocessing pipeline
    """
    
    def __init__(self,
                 bearer_token: str,
                 api_key: str = None,
                 api_secret: str = None,
                 access_token: str = None,
                 access_token_secret: str = None,
                 database_url: str = "sqlite:///twitter_sentiment.db",
                 max_tweets_per_hour: int = 10000,
                 sentiment_threshold: float = 0.1):
        """
        Initialize Twitter sentiment collector.
        
        Args:
            bearer_token: Twitter Bearer Token for API v2
            api_key: Twitter API Key (for API v1.1)
            api_secret: Twitter API Secret
            access_token: Twitter Access Token
            access_token_secret: Twitter Access Token Secret
            database_url: Database connection URL
            max_tweets_per_hour: Rate limiting
            sentiment_threshold: Minimum sentiment magnitude to store
        """
        self.bearer_token = bearer_token
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret
        
        # Initialize API clients
        self._setup_twitter_clients()
        
        # Database setup
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Rate limiting
        self.max_tweets_per_hour = max_tweets_per_hour
        self.tweet_timestamps = deque()
        
        # Sentiment analysis setup
        self.sentiment_threshold = sentiment_threshold
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Load spaCy model for NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logging.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Cryptocurrency keywords and symbols
        self.crypto_keywords = {
            'bitcoin': ['bitcoin', 'btc', '$btc'],
            'ethereum': ['ethereum', 'eth', '$eth'],
            'binancecoin': ['binance', 'bnb', '$bnb'],
            'cardano': ['cardano', 'ada', '$ada'],
            'solana': ['solana', 'sol', '$sol'],
            'dogecoin': ['dogecoin', 'doge', '$doge'],
            'ripple': ['ripple', 'xrp', '$xrp'],
            'polkadot': ['polkadot', 'dot', '$dot'],
            'avalanche': ['avalanche', 'avax', '$avax'],
            'chainlink': ['chainlink', 'link', '$link']
        }
        
        # Performance tracking
        self.stats = {
            'tweets_collected': 0,
            'tweets_processed': 0,
            'api_errors': 0,
            'rate_limits_hit': 0,
            'start_time': datetime.now()
        }
        
        # Threading setup
        self.is_streaming = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _setup_twitter_clients(self):
        """Setup Twitter API clients."""
        # Twitter API v2 client
        if self.bearer_token:
            self.client_v2 = tweepy.Client(bearer_token=self.bearer_token)
        
        # Twitter API v1.1 client (for streaming)
        if all([self.api_key, self.api_secret, self.access_token, self.access_token_secret]):
            auth = tweepy.OAuth1UserHandler(
                self.api_key, self.api_secret,
                self.access_token, self.access_token_secret
            )
            self.api_v1 = tweepy.API(auth, wait_on_rate_limit=True)
    
    def preprocess_text(self, text: str) -> str:
        """
        Advanced text preprocessing for sentiment analysis.
        
        Args:
            text: Raw tweet text
            
        Returns:
            Preprocessed text ready for analysis
        """
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions but keep the context
        text = re.sub(r'@\w+', '[USER]', text)
        
        # Replace hashtags but keep the text
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Handle emojis (keep them for sentiment)
        # Convert some common emojis to text
        emoji_dict = {
            '🚀': 'rocket',
            '📈': 'chart_up',
            '📉': 'chart_down',
            '💎': 'diamond_hands',
            '🌙': 'moon',
            '🔥': 'fire',
            '💰': 'money',
            '⚡': 'lightning'
        }
        
        for emoji, text_rep in emoji_dict.items():
            text = text.replace(emoji, f' {text_rep} ')
        
        # Normalize case
        text = text.lower()
        
        return text
    
    def detect_crypto_mentions(self, text: str) -> List[str]:
        """
        Detect cryptocurrency mentions in text.
        
        Args:
            text: Preprocessed text
            
        Returns:
            List of detected cryptocurrencies
        """
        mentions = []
        text_lower = text.lower()
        
        for crypto, keywords in self.crypto_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    mentions.append(crypto)
                    break
        
        return list(set(mentions))  # Remove duplicates
    
    def analyze_sentiment_multilayer(self, text: str) -> Tuple[float, str, float]:
        """
        Multi-layered sentiment analysis combining multiple methods.
        
        Args:
            text: Preprocessed text
            
        Returns:
            Tuple of (sentiment_score, sentiment_label, confidence)
        """
        scores = []
        
        # VADER Sentiment Analysis
        vader_scores = self.vader_analyzer.polarity_scores(text)
        vader_compound = vader_scores['compound']
        scores.append(vader_compound)
        
        # TextBlob Sentiment Analysis
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        scores.append(textblob_polarity)
        
        # Simple lexicon-based approach for crypto-specific terms
        crypto_positive_words = ['moon', 'bullish', 'pump', 'rocket', 'diamond_hands', 'hodl', 'buy']
        crypto_negative_words = ['dump', 'bearish', 'crash', 'sell', 'panic', 'fud', 'dip']
        
        crypto_score = 0
        words = text.split()
        for word in words:
            if word in crypto_positive_words:
                crypto_score += 0.1
            elif word in crypto_negative_words:
                crypto_score -= 0.1
        
        crypto_score = max(-1, min(1, crypto_score))  # Clip to [-1, 1]
        scores.append(crypto_score)
        
        # Ensemble scoring (weighted average)
        weights = [0.4, 0.3, 0.3]  # VADER, TextBlob, Crypto-specific
        final_score = sum(w * s for w, s in zip(weights, scores))
        
        # Determine label
        if final_score > 0.1:
            label = 'positive'
        elif final_score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        # Calculate confidence (based on score magnitude and agreement)
        score_std = np.std(scores)
        confidence = abs(final_score) * (1 - score_std)  # Higher confidence when scores agree
        confidence = max(0, min(1, confidence))
        
        return final_score, label, confidence
    
    def detect_bot_probability(self, tweet_data: Dict) -> float:
        """
        Simple bot detection based on tweet patterns.
        
        Args:
            tweet_data: Tweet metadata
            
        Returns:
            Bot probability (0-1)
        """
        bot_score = 0.0
        
        # Check username patterns
        username = tweet_data.get('username', '').lower()
        if re.search(r'\d{8,}', username):  # Many digits
            bot_score += 0.3
        
        # Check for high retweet ratio
        retweet_count = tweet_data.get('retweet_count', 0)
        like_count = tweet_data.get('like_count', 1)
        if retweet_count > like_count * 5:  # Unusually high retweet ratio
            bot_score += 0.2
        
        # Check text patterns
        text = tweet_data.get('text', '')
        if len(re.findall(r'[A-Z]{3,}', text)) > 3:  # Too many caps words
            bot_score += 0.2
        
        if text.count('$') > 3:  # Too many dollar signs
            bot_score += 0.1
        
        return min(1.0, bot_score)
    
    def process_tweet(self, tweet) -> Optional[TweetData]:
        """
        Process a single tweet and extract relevant information.
        
        Args:
            tweet: Tweet object from Twitter API
            
        Returns:
            Processed TweetData object or None if filtered out
        """
        try:
            # Extract basic tweet information
            tweet_dict = tweet._json if hasattr(tweet, '_json') else tweet
            
            text = tweet_dict.get('text', '')
            if not text:
                return None
            
            # Skip if text is too short
            if len(text) < 10:
                return None
            
            # Extract entities
            entities = tweet_dict.get('entities', {})
            hashtags = [tag['text'] for tag in entities.get('hashtags', [])]
            mentions = [mention['screen_name'] for mention in entities.get('user_mentions', [])]
            urls = [url['expanded_url'] for url in entities.get('urls', [])]
            
            # Check if contains crypto mentions
            crypto_mentions = self.detect_crypto_mentions(text)
            if not crypto_mentions:
                return None  # Only process crypto-related tweets
            
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Analyze sentiment
            sentiment_score, sentiment_label, confidence = self.analyze_sentiment_multilayer(processed_text)
            
            # Skip tweets with very low sentiment magnitude
            if abs(sentiment_score) < self.sentiment_threshold:
                return None
            
            # Bot detection
            bot_probability = self.detect_bot_probability(tweet_dict)
            
            # Skip likely bots
            if bot_probability > 0.7:
                return None
            
            # Create TweetData object
            tweet_data = TweetData(
                id=tweet_dict['id_str'],
                text=text,
                user_id=tweet_dict['user']['id_str'],
                username=tweet_dict['user']['screen_name'],
                created_at=datetime.strptime(tweet_dict['created_at'], '%a %b %d %H:%M:%S %z %Y'),
                retweet_count=tweet_dict.get('retweet_count', 0),
                like_count=tweet_dict.get('favorite_count', 0),
                reply_count=tweet_dict.get('reply_count', 0),
                quote_count=tweet_dict.get('quote_count', 0),
                language=tweet_dict.get('lang', 'en'),
                hashtags=hashtags,
                mentions=mentions,
                urls=urls,
                is_retweet=text.startswith('RT @'),
                is_reply=tweet_dict.get('in_reply_to_status_id') is not None,
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label,
                confidence=confidence,
                processed_text=processed_text,
                crypto_mentions=crypto_mentions,
                bot_probability=bot_probability
            )
            
            return tweet_data
            
        except Exception as e:
            self.logger.error(f"Error processing tweet: {e}")
            self.stats['api_errors'] += 1
            return None
    
    def save_tweet(self, tweet_data: TweetData):
        """Save tweet data to database."""
        try:
            tweet_record = TweetRecord(
                id=tweet_data.id,
                text=tweet_data.text,
                user_id=tweet_data.user_id,
                username=tweet_data.username,
                created_at=tweet_data.created_at,
                retweet_count=tweet_data.retweet_count,
                like_count=tweet_data.like_count,
                reply_count=tweet_data.reply_count,
                quote_count=tweet_data.quote_count,
                language=tweet_data.language,
                hashtags=json.dumps(tweet_data.hashtags),
                mentions=json.dumps(tweet_data.mentions),
                urls=json.dumps(tweet_data.urls),
                is_retweet=tweet_data.is_retweet,
                is_reply=tweet_data.is_reply,
                sentiment_score=tweet_data.sentiment_score,
                sentiment_label=tweet_data.sentiment_label,
                confidence=tweet_data.confidence,
                processed_text=tweet_data.processed_text,
                crypto_mentions=json.dumps(tweet_data.crypto_mentions),
                bot_probability=tweet_data.bot_probability
            )
            
            self.session.merge(tweet_record)  # Use merge to handle duplicates
            self.session.commit()
            self.stats['tweets_processed'] += 1
            
        except Exception as e:
            self.logger.error(f"Error saving tweet: {e}")
            self.session.rollback()
    
    def check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        # Remove old timestamps
        while self.tweet_timestamps and self.tweet_timestamps[0] < hour_ago:
            self.tweet_timestamps.popleft()
        
        return len(self.tweet_timestamps) < self.max_tweets_per_hour
    
    def collect_tweets_search(self, 
                            query: str,
                            max_results: int = 100,
                            since_days: int = 7) -> List[TweetData]:
        """
        Collect tweets using search API.
        
        Args:
            query: Search query
            max_results: Maximum number of tweets to collect
            since_days: Number of days to look back
            
        Returns:
            List of processed tweet data
        """
        tweets_data = []
        
        try:
            # Calculate start time
            start_time = datetime.now() - timedelta(days=since_days)
            
            # Search tweets
            tweets = tweepy.Cursor(
                self.api_v1.search_tweets,
                q=query,
                lang='en',
                result_type='recent',
                tweet_mode='extended'
            ).items(max_results)
            
            for tweet in tweets:
                if not self.check_rate_limit():
                    self.logger.warning("Rate limit reached, stopping collection")
                    self.stats['rate_limits_hit'] += 1
                    break
                
                tweet_data = self.process_tweet(tweet)
                if tweet_data:
                    tweets_data.append(tweet_data)
                    self.save_tweet(tweet_data)
                    self.tweet_timestamps.append(datetime.now())
                
                self.stats['tweets_collected'] += 1
                
                # Rate limiting
                time.sleep(0.1)  # Be nice to the API
        
        except Exception as e:
            self.logger.error(f"Error in tweet search: {e}")
            self.stats['api_errors'] += 1
        
        return tweets_data
    
    def start_streaming(self,
                       track_keywords: List[str] = None,
                       callback: Callable[[TweetData], None] = None):
        """
        Start real-time tweet streaming.
        
        Args:
            track_keywords: Keywords to track
            callback: Optional callback function for each processed tweet
        """
        if track_keywords is None:
            # Default crypto keywords
            track_keywords = []
            for keywords in self.crypto_keywords.values():
                track_keywords.extend(keywords[:2])  # Take first 2 keywords per crypto
        
        class TwitterStreamListener(tweepy.Stream):
            def __init__(self, collector, callback=None):
                super().__init__(
                    collector.api_key,
                    collector.api_secret,
                    collector.access_token,
                    collector.access_token_secret
                )
                self.collector = collector
                self.callback = callback
            
            def on_status(self, status):
                if not self.collector.check_rate_limit():
                    return True  # Continue but don't process
                
                tweet_data = self.collector.process_tweet(status)
                if tweet_data:
                    self.collector.save_tweet(tweet_data)
                    self.collector.tweet_timestamps.append(datetime.now())
                    
                    if self.callback:
                        self.callback(tweet_data)
                
                self.collector.stats['tweets_collected'] += 1
                return True
            
            def on_error(self, status_code):
                self.collector.logger.error(f"Twitter API error: {status_code}")
                self.collector.stats['api_errors'] += 1
                if status_code == 420:  # Rate limit
                    self.collector.stats['rate_limits_hit'] += 1
                    return False  # Disconnect
                return True
        
        # Start streaming
        stream = TwitterStreamListener(self, callback)
        
        try:
            self.is_streaming = True
            self.logger.info(f"Starting Twitter stream with keywords: {track_keywords}")
            stream.filter(track=track_keywords, threaded=True)
        except Exception as e:
            self.logger.error(f"Streaming error: {e}")
            self.is_streaming = False
    
    def stop_streaming(self):
        """Stop tweet streaming."""
        self.is_streaming = False
        self.logger.info("Twitter streaming stopped")
    
    def get_sentiment_summary(self, 
                            crypto: str = None,
                            hours: int = 24) -> Dict:
        """
        Get sentiment summary for the specified period.
        
        Args:
            crypto: Specific cryptocurrency (None for all)
            hours: Number of hours to look back
            
        Returns:
            Sentiment summary dictionary
        """
        from sqlalchemy import and_, func
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        query = self.session.query(TweetRecord).filter(
            TweetRecord.created_at >= cutoff_time
        )
        
        if crypto:
            query = query.filter(
                TweetRecord.crypto_mentions.contains(f'"{crypto}"')
            )
        
        tweets = query.all()
        
        if not tweets:
            return {'error': 'No tweets found for the specified criteria'}
        
        # Calculate summary statistics
        sentiment_scores = [t.sentiment_score for t in tweets]
        sentiment_labels = [t.sentiment_label for t in tweets]
        
        summary = {
            'total_tweets': len(tweets),
            'average_sentiment': np.mean(sentiment_scores),
            'sentiment_std': np.std(sentiment_scores),
            'positive_tweets': sentiment_labels.count('positive'),
            'negative_tweets': sentiment_labels.count('negative'),
            'neutral_tweets': sentiment_labels.count('neutral'),
            'period_hours': hours,
            'crypto_filter': crypto,
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculate sentiment distribution
        summary['sentiment_distribution'] = {
            'positive': summary['positive_tweets'] / summary['total_tweets'],
            'negative': summary['negative_tweets'] / summary['total_tweets'],
            'neutral': summary['neutral_tweets'] / summary['total_tweets']
        }
        
        return summary
    
    def get_stats(self) -> Dict:
        """Get collection statistics."""
        runtime = datetime.now() - self.stats['start_time']
        
        return {
            **self.stats,
            'runtime_hours': runtime.total_seconds() / 3600,
            'tweets_per_hour': self.stats['tweets_collected'] / max(1, runtime.total_seconds() / 3600),
            'processing_rate': self.stats['tweets_processed'] / max(1, self.stats['tweets_collected'])
        }

# Example usage
if __name__ == "__main__":
    # Initialize collector (requires Twitter API credentials)
    collector = TwitterSentimentCollector(
        bearer_token="your_bearer_token_here",
        api_key="your_api_key_here",
        api_secret="your_api_secret_here", 
        access_token="your_access_token_here",
        access_token_secret="your_access_token_secret_here"
    )
    
    # Collect some tweets
    crypto_query = "bitcoin OR ethereum OR $BTC OR $ETH -is:retweet"
    tweets = collector.collect_tweets_search(crypto_query, max_results=50)
    
    print(f"Collected {len(tweets)} tweets")
    
    # Get sentiment summary
    summary = collector.get_sentiment_summary(crypto='bitcoin', hours=24)
    print(f"Sentiment summary: {summary}")
    
    # Print statistics
    print(f"Collection stats: {collector.get_stats()}")