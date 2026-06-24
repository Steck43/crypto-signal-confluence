"""
Sentiment analyzer for news and RSS feeds
Primary sources: RSS feeds (CoinDesk, Cointelegraph) + Alpha Vantage news sentiment
"""

import numpy as np
from datetime import datetime
import re

class SentimentAnalyzer:
    def __init__(self):
        self.model_loaded = False
        
        # Enhanced keyword dictionaries for crypto news
        self.bullish_keywords = {
            'strong': ['bullish', 'surge', 'rally', 'breakout', 'moon', 'pump', 'positive', 'gain', 'rise', 'climb'],
            'medium': ['up', 'higher', 'growth', 'adoption', 'institutional', 'partnership', 'upgrade'],
            'weak': ['stable', 'steady', 'maintain', 'hold', 'consolidate']
        }
        
        self.bearish_keywords = {
            'strong': ['bearish', 'crash', 'dump', 'sell-off', 'negative', 'decline', 'drop', 'plunge', 'tank'],
            'medium': ['down', 'lower', 'fall', 'correction', 'volatility', 'uncertainty', 'risk'],
            'weak': ['stable', 'steady', 'maintain', 'hold', 'consolidate']
        }
        
        # Crypto-specific terms
        self.crypto_bullish = ['bitcoin', 'btc', 'ethereum', 'eth', 'solana', 'sol', 'defi', 'nft', 'web3', 'blockchain']
        self.crypto_bearish = ['regulation', 'ban', 'crackdown', 'sec', 'tax', 'hack', 'scam', 'ponzi']
        
    def analyze_news_sentiment(self, news_data):
        """
        Analyze sentiment from news data (RSS feeds + Alpha Vantage)
        Returns sentiment score between -1.0 (bearish) and 1.0 (bullish)
        """
        try:
            if not news_data or len(news_data) == 0:
                return 0.0  # Neutral if no data
                
            total_score = 0.0
            article_count = 0
            
            for article in news_data:
                if isinstance(article, dict):
                    # Extract text content
                    text = self._extract_text(article)
                    if not text:
                        continue
                        
                    # Calculate article sentiment
                    article_score = self._calculate_article_sentiment(text)
                    
                    # Weight by source reliability
                    source_weight = self._get_source_weight(article.get('source', ''))
                    weighted_score = article_score * source_weight
                    
                    total_score += weighted_score
                    article_count += 1
            
            # Return average sentiment
            return total_score / article_count if article_count > 0 else 0.0
            
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return 0.0  # Neutral on error
    
    def _extract_text(self, article):
        """Extract text content from article"""
        text = ""
        
        # Combine title and content
        if 'title' in article:
            text += article['title'] + " "
        if 'content' in article:
            text += article['content'] + " "
        if 'summary' in article:
            text += article['summary'] + " "
            
        return text.lower().strip()
    
    def _calculate_article_sentiment(self, text):
        """Calculate sentiment score for article text"""
        # Count keyword occurrences by strength
        bullish_scores = {
            'strong': sum(1 for word in self.bullish_keywords['strong'] if word in text) * 0.3,
            'medium': sum(1 for word in self.bullish_keywords['medium'] if word in text) * 0.2,
            'weak': sum(1 for word in self.bullish_keywords['weak'] if word in text) * 0.1
        }
        
        bearish_scores = {
            'strong': sum(1 for word in self.bearish_keywords['strong'] if word in text) * 0.3,
            'medium': sum(1 for word in self.bearish_keywords['medium'] if word in text) * 0.2,
            'weak': sum(1 for word in self.bearish_keywords['weak'] if word in text) * 0.1
        }
        
        # Calculate total scores
        total_bullish = sum(bullish_scores.values())
        total_bearish = sum(bearish_scores.values())
        
        # Crypto-specific adjustments
        crypto_bullish_count = sum(1 for term in self.crypto_bullish if term in text)
        crypto_bearish_count = sum(1 for term in self.crypto_bearish if term in text)
        
        # Adjust scores based on crypto context
        if crypto_bullish_count > crypto_bearish_count:
            total_bullish *= 1.2  # Boost bullish sentiment
        elif crypto_bearish_count > crypto_bullish_count:
            total_bearish *= 1.2  # Boost bearish sentiment
        
        # Calculate final sentiment score
        if total_bullish > total_bearish:
            return min(1.0, (total_bullish - total_bearish) * 0.5)
        elif total_bearish > total_bullish:
            return -min(1.0, (total_bearish - total_bullish) * 0.5)
        else:
            return 0.0
    
    def _get_source_weight(self, source):
        """Get reliability weight for news source"""
        source_weights = {
            'coindesk': 1.0,
            'cointelegraph': 1.0,
            'alphavantage': 0.9,
            'reuters': 0.8,
            'bloomberg': 0.8,
            'cnbc': 0.7,
            'default': 0.5
        }
        
        source_lower = source.lower() if source else 'default'
        return source_weights.get(source_lower, source_weights['default'])
    
    def analyze_rss_sentiment(self, rss_data):
        """
        Analyze sentiment specifically from RSS feeds
        """
        return self.analyze_news_sentiment(rss_data)
    
    def analyze_alpha_vantage_sentiment(self, alpha_vantage_data):
        """
        Analyze sentiment from Alpha Vantage news API
        """
        try:
            if not alpha_vantage_data or len(alpha_vantage_data) == 0:
                return 0.0
                
            # Alpha Vantage might provide pre-calculated sentiment scores
            if isinstance(alpha_vantage_data, list) and len(alpha_vantage_data) > 0:
                if 'sentiment_score' in alpha_vantage_data[0]:
                    # Use pre-calculated scores
                    scores = [item.get('sentiment_score', 0) for item in alpha_vantage_data]
                    return np.mean(scores)
                else:
                    # Fall back to text analysis
                    return self.analyze_news_sentiment(alpha_vantage_data)
            
            return 0.0
            
        except Exception as e:
            print(f"Alpha Vantage sentiment analysis error: {e}")
            return 0.0
    
    def combine_sentiment_scores(self, rss_score, alpha_vantage_score, weights=None):
        """
        Combine sentiment scores from different sources
        """
        if weights is None:
            weights = {'rss': 0.6, 'alpha_vantage': 0.4}  # RSS feeds more weight
            
        combined_score = (
            rss_score * weights['rss'] + 
            alpha_vantage_score * weights['alpha_vantage']
        )
        
        return np.clip(combined_score, -1.0, 1.0)  # Ensure within bounds 