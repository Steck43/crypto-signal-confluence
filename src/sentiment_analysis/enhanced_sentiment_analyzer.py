"""
ML-Enhanced Sentiment Analyzer for Crypto News

Integrates with your ML ensemble to provide sophisticated sentiment analysis
beyond simple keyword matching.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from collections import deque
import asyncio
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SentimentScore:
    """Enhanced sentiment score with confidence and context"""
    score: float  # -1 to 1
    confidence: float  # 0 to 1
    volume: int  # Number of sources
    momentum: float  # Rate of change
    sources: Dict[str, float]  # Source-specific scores
    keywords: List[str]  # Important keywords found
    timestamp: datetime


class TransformerSentimentModel(nn.Module):
    """
    Lightweight transformer for crypto sentiment analysis
    Designed to run efficiently on RTX 4090
    """
    
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 256, 
                 num_heads: int = 8, num_layers: int = 4):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 512, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output heads
        self.sentiment_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3)  # Negative, Neutral, Positive
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask=None):
        # Embedding
        x = self.embedding(input_ids)
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]
        
        # Transformer encoding
        if attention_mask is not None:
            x = self.transformer(x, src_key_padding_mask=~attention_mask)
        else:
            x = self.transformer(x)
        
        # Global pooling
        x = x.mean(dim=1)
        
        # Predictions
        sentiment = self.sentiment_head(x)
        confidence = self.confidence_head(x)
        
        return sentiment, confidence


class EnhancedSentimentAnalyzer:
    """
    Enhanced sentiment analyzer with ML capabilities
    Integrates with your existing sentiment analyzer
    """
    
    def __init__(self, device: str = 'auto', use_ml: bool = True):
        # Device configuration
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.use_ml = use_ml and torch.cuda.is_available()
        
        # Original keyword-based analyzer (fallback)
        from .sentiment_analyzer import SentimentAnalyzer
        self.keyword_analyzer = SentimentAnalyzer()  # Your original
        
        # ML model
        if self.use_ml:
            self.model = TransformerSentimentModel().to(self.device)
            self.model.eval()  # Default to eval mode
            logger.info(f"ML Sentiment analyzer initialized on {self.device}")
        
        # Enhanced keyword dictionaries with weights
        self.crypto_entities = {
            'btc': 1.2, 'bitcoin': 1.2, 'eth': 1.1, 'ethereum': 1.1,
            'sol': 1.0, 'solana': 1.0, 'defi': 0.9, 'nft': 0.8
        }
        
        self.bullish_patterns = {
            'breakout': 0.8, 'accumulation': 0.7, 'support': 0.6,
            'adoption': 0.9, 'institutional': 0.85, 'etf': 0.9
        }
        
        self.bearish_patterns = {
            'breakdown': -0.8, 'distribution': -0.7, 'resistance': -0.6,
            'regulation': -0.7, 'crackdown': -0.9, 'hack': -0.95
        }
        
        # Sentiment history for momentum calculation
        self.sentiment_history = deque(maxlen=100)
        
        # Source reliability weights
        self.source_weights = {
            'coindesk': 1.0,
            'cointelegraph': 0.95,
            'reuters': 0.9,
            'bloomberg': 0.9,
            'twitter': 0.7,
            'reddit': 0.6,
            'unknown': 0.5
        }
        
        # Cache for performance
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def analyze_news_sentiment(self, news_data: List[Dict]) -> SentimentScore:
        """
        Analyze sentiment from multiple news sources with ML enhancement
        """
        if not news_data:
            return self._create_neutral_sentiment()
        
        # Check cache
        cache_key = self._create_cache_key(news_data)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        # Analyze each article
        article_sentiments = []
        source_scores = {}
        all_keywords = []
        
        for article in news_data:
            # Extract and clean text
            text = self._extract_and_clean_text(article)
            if not text:
                continue
            
            # Get sentiment score
            if self.use_ml:
                sentiment, confidence = await self._ml_analyze(text)
            else:
                sentiment, confidence = self._keyword_analyze(text)
            
            # Weight by source
            source = article.get('source', 'unknown').lower()
            weight = self.source_weights.get(source, 0.5)
            weighted_sentiment = sentiment * weight * confidence
            
            article_sentiments.append(weighted_sentiment)
            
            # Track by source
            if source not in source_scores:
                source_scores[source] = []
            source_scores[source].append(sentiment)
            
            # Extract keywords
            keywords = self._extract_keywords(text)
            all_keywords.extend(keywords)
        
        # Calculate aggregate sentiment
        if not article_sentiments:
            return self._create_neutral_sentiment()
        
        # Weighted average sentiment
        avg_sentiment = np.mean(article_sentiments)
        
        # Calculate confidence based on agreement
        sentiment_std = np.std(article_sentiments)
        confidence = 1.0 - min(sentiment_std, 1.0)  # High std = low confidence
        
        # Calculate momentum
        momentum = self._calculate_sentiment_momentum(avg_sentiment)
        
        # Create result
        result = SentimentScore(
            score=float(np.clip(avg_sentiment, -1, 1)),
            confidence=float(confidence),
            volume=len(article_sentiments),
            momentum=float(momentum),
            sources={k: float(np.mean(v)) for k, v in source_scores.items()},
            keywords=list(set(all_keywords))[:10],  # Top 10 unique keywords
            timestamp=datetime.now()
        )
        
        # Update history
        self.sentiment_history.append(result)
        
        # Cache result
        self._cache_result(cache_key, result)
        
        return result
    
    async def _ml_analyze(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment using ML model
        """
        try:
            # Tokenize (simplified - you'd use a proper tokenizer)
            tokens = self._simple_tokenize(text)
            input_ids = torch.tensor([tokens], device=self.device)
            
            # Run model
            with torch.no_grad():
                sentiment_logits, confidence = self.model(input_ids)
            
            # Convert to sentiment score
            probs = torch.softmax(sentiment_logits, dim=-1)
            sentiment_score = (probs[0, 2] - probs[0, 0]).item()  # Positive - Negative
            confidence_score = confidence.item()
            
            return sentiment_score, confidence_score
            
        except Exception as e:
            logger.warning(f"ML sentiment analysis failed: {e}, falling back to keywords")
            return self._keyword_analyze(text)
    
    def _keyword_analyze(self, text: str) -> Tuple[float, float]:
        """
        Fallback keyword-based analysis with enhancements
        """
        text_lower = text.lower()
        
        # Entity detection boosts
        entity_boost = 0.0
        for entity, weight in self.crypto_entities.items():
            if entity in text_lower:
                entity_boost += weight * 0.1
        
        # Pattern matching
        bullish_score = sum(
            weight for pattern, weight in self.bullish_patterns.items()
            if pattern in text_lower
        )
        
        bearish_score = sum(
            weight for pattern, weight in self.bearish_patterns.items()
            if pattern in text_lower
        )
        
        # Use original analyzer for additional signals
        basic_score = self.keyword_analyzer._calculate_article_sentiment(text)
        
        # Combine scores
        total_score = (bullish_score + bearish_score + basic_score + entity_boost) / 4
        
        # Confidence based on signal strength
        confidence = min(abs(total_score) * 2, 1.0)
        
        return np.clip(total_score, -1, 1), confidence
    
    def _calculate_sentiment_momentum(self, current_sentiment: float) -> float:
        """
        Calculate sentiment momentum (rate of change)
        """
        if len(self.sentiment_history) < 3:
            return 0.0
        
        recent_sentiments = [s.score for s in list(self.sentiment_history)[-10:]]
        
        # Calculate trend
        if len(recent_sentiments) >= 2:
            momentum = current_sentiment - np.mean(recent_sentiments)
            return np.clip(momentum * 2, -1, 1)  # Scale and clip
        
        return 0.0
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract important keywords from text
        """
        keywords = []
        text_lower = text.lower()
        
        # Check for entities
        for entity in self.crypto_entities:
            if entity in text_lower:
                keywords.append(entity)
        
        # Check for patterns
        for pattern in {**self.bullish_patterns, **self.bearish_patterns}:
            if pattern in text_lower:
                keywords.append(pattern)
        
        return keywords
    
    def _extract_and_clean_text(self, article: Dict) -> str:
        """
        Extract and clean text from article
        """
        text_parts = []
        
        for field in ['title', 'content', 'summary', 'description']:
            if field in article and article[field]:
                text_parts.append(str(article[field]))
        
        text = ' '.join(text_parts)
        
        # Basic cleaning
        text = ' '.join(text.split())  # Normalize whitespace
        
        return text[:2000]  # Limit length for performance
    
    def _simple_tokenize(self, text: str, max_length: int = 512) -> List[int]:
        """
        Simple tokenization for demo (replace with proper tokenizer)
        """
        # This is a placeholder - use a proper tokenizer in production
        words = text.lower().split()[:max_length]
        
        # Simple word to ID mapping (you'd use a real vocabulary)
        word_to_id = {word: i % 10000 for i, word in enumerate(set(words))}
        
        return [word_to_id.get(word, 0) for word in words]
    
    def _create_cache_key(self, news_data: List[Dict]) -> str:
        """Create cache key from news data"""
        # Simple hash of titles
        titles = [article.get('title', '') for article in news_data[:10]]
        return str(hash(''.join(titles)))
    
    def _get_cached_result(self, cache_key: str) -> Optional[SentimentScore]:
        """Get cached result if still valid"""
        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_ttl:
                return result
        return None
    
    def _cache_result(self, cache_key: str, result: SentimentScore):
        """Cache result with timestamp"""
        self.cache[cache_key] = (result, datetime.now())
        
        # Clean old cache entries
        if len(self.cache) > 100:
            oldest_keys = sorted(self.cache.keys())[:50]
            for key in oldest_keys:
                del self.cache[key]
    
    def _create_neutral_sentiment(self) -> SentimentScore:
        """Create neutral sentiment when no data available"""
        return SentimentScore(
            score=0.0,
            confidence=0.0,
            volume=0,
            momentum=0.0,
            sources={},
            keywords=[],
            timestamp=datetime.now()
        )
    
    def get_sentiment_trend(self, hours: int = 24) -> Dict[str, float]:
        """
        Get sentiment trend over specified hours
        """
        if not self.sentiment_history:
            return {'trend': 0.0, 'volatility': 0.0}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_sentiments = [
            s for s in self.sentiment_history 
            if s.timestamp > cutoff_time
        ]
        
        if len(recent_sentiments) < 2:
            return {'trend': 0.0, 'volatility': 0.0}
        
        scores = [s.score for s in recent_sentiments]
        
        # Calculate trend (linear regression slope)
        x = np.arange(len(scores))
        slope, _ = np.polyfit(x, scores, 1)
        
        # Calculate volatility
        volatility = np.std(scores)
        
        return {
            'trend': float(slope),
            'volatility': float(volatility),
            'current': float(scores[-1]),
            'average': float(np.mean(scores))
        }
    
    async def analyze_social_sentiment(self, social_data: List[Dict]) -> SentimentScore:
        """
        Analyze sentiment from social media (Twitter, Reddit, etc.)
        Apply different weights and processing for social sources
        """
        # Social media specific processing
        # Higher weight on viral content, trending topics
        # Consider engagement metrics (likes, retweets, etc.)
        
        # For now, use standard analysis with lower confidence
        result = await self.analyze_news_sentiment(social_data)
        result.confidence *= 0.7  # Social media is less reliable
        
        return result
    
    def combine_sentiments(self, news_sentiment: SentimentScore, 
                         social_sentiment: Optional[SentimentScore] = None,
                         weights: Dict[str, float] = None) -> SentimentScore:
        """
        Combine multiple sentiment sources
        """
        if weights is None:
            weights = {'news': 0.7, 'social': 0.3}
        
        if not social_sentiment:
            return news_sentiment
        
        # Weighted combination
        combined_score = (
            news_sentiment.score * weights['news'] +
            social_sentiment.score * weights['social']
        )
        
        combined_confidence = (
            news_sentiment.confidence * weights['news'] +
            social_sentiment.confidence * weights['social']
        )
        
        # Combine keywords
        all_keywords = news_sentiment.keywords + social_sentiment.keywords
        unique_keywords = list(set(all_keywords))[:15]
        
        return SentimentScore(
            score=float(np.clip(combined_score, -1, 1)),
            confidence=float(combined_confidence),
            volume=news_sentiment.volume + social_sentiment.volume,
            momentum=(news_sentiment.momentum + social_sentiment.momentum) / 2,
            sources={**news_sentiment.sources, **social_sentiment.sources},
            keywords=unique_keywords,
            timestamp=datetime.now()
        )


# Integration with your existing system
class SentimentAnalysisIntegration:
    """
    Integrates enhanced sentiment analysis with your trading system
    """
    
    def __init__(self, ml_ensemble_manager, sentiment_analyzer: EnhancedSentimentAnalyzer):
        self.ml_ensemble = ml_ensemble_manager
        self.sentiment_analyzer = sentiment_analyzer
        
    async def get_trading_sentiment(self, symbol: str, news_data: List[Dict], 
                                  social_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Get comprehensive sentiment analysis for trading decisions
        """
        # Analyze news sentiment
        news_sentiment = await self.sentiment_analyzer.analyze_news_sentiment(news_data)
        
        # Analyze social sentiment if available
        social_sentiment = None
        if social_data:
            social_sentiment = await self.sentiment_analyzer.analyze_social_sentiment(social_data)
        
        # Combine sentiments
        combined_sentiment = self.sentiment_analyzer.combine_sentiments(
            news_sentiment, social_sentiment
        )
        
        # Get sentiment trend
        trend_data = self.sentiment_analyzer.get_sentiment_trend(hours=24)
        
        # Create sentiment signal
        sentiment_signal = self._create_sentiment_signal(combined_sentiment, trend_data)
        
        return {
            'symbol': symbol,
            'sentiment_score': combined_sentiment.score,
            'confidence': combined_sentiment.confidence,
            'momentum': combined_sentiment.momentum,
            'volume': combined_sentiment.volume,
            'trend': trend_data,
            'signal': sentiment_signal,
            'keywords': combined_sentiment.keywords,
            'sources': combined_sentiment.sources
        }
    
    def _create_sentiment_signal(self, sentiment: SentimentScore, 
                               trend: Dict[str, float]) -> Dict[str, Any]:
        """
        Create trading signal from sentiment analysis
        """
        # Strong bullish signal
        if sentiment.score > 0.5 and sentiment.confidence > 0.7 and trend['trend'] > 0:
            return {
                'type': 'BUY',
                'strength': min(sentiment.score * sentiment.confidence, 1.0),
                'reason': 'Strong positive sentiment with upward trend'
            }
        
        # Strong bearish signal
        elif sentiment.score < -0.5 and sentiment.confidence > 0.7 and trend['trend'] < 0:
            return {
                'type': 'SELL',
                'strength': min(abs(sentiment.score) * sentiment.confidence, 1.0),
                'reason': 'Strong negative sentiment with downward trend'
            }
        
        # Neutral or uncertain
        else:
            return {
                'type': 'HOLD',
                'strength': sentiment.confidence * 0.5,
                'reason': 'Neutral or uncertain sentiment'
            } 