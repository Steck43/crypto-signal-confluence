"""
Social Data Collection Module

Handles collection of social media data and sentiment analysis
for the institutional AI crypto trading system.
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass

from ..api.rss_collector import CryptoRSSCollector
from ..sentiment_analysis.sentiment_analyzer import SentimentAnalyzer


@dataclass
class SocialDataPoint:
    """Social data point structure."""
    timestamp: datetime
    source: str
    content: str
    sentiment_score: float
    sentiment_label: str
    engagement_score: float
    topic: str
    url: str


class SocialDataCollector:
    """Social media data collector."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize social data collector."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Initialize components
        self.rss_collector = CryptoRSSCollector()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Data storage
        self.social_data: List[SocialDataPoint] = []
        
        self.logger.info("Social data collector initialized")
    
    async def collect_social_data(self, hours_back: int = 24) -> List[SocialDataPoint]:
        """Collect social media data."""
        try:
            # Collect RSS news
            articles = await self.rss_collector.collect_latest_news(hours_back)
            
            # Analyze sentiment
            social_data = []
            for article in articles:
                sentiment = await self.sentiment_analyzer.analyze_text(article['title'] + " " + article['summary'])
                
                data_point = SocialDataPoint(
                    timestamp=article['timestamp'],
                    source=article['source'],
                    content=article['title'] + " " + article['summary'],
                    sentiment_score=sentiment['compound'],
                    sentiment_label=sentiment['label'],
                    engagement_score=0.0,  # Placeholder
                    topic=article.get('topic', 'crypto'),
                    url=article['link']
                )
                social_data.append(data_point)
            
            self.social_data.extend(social_data)
            return social_data
            
        except Exception as e:
            self.logger.error(f"Error collecting social data: {e}")
            return []
    
    def get_sentiment_summary(self) -> Dict[str, Any]:
        """Get sentiment summary statistics."""
        if not self.social_data:
            return {}
        
        df = pd.DataFrame([{
            'sentiment_score': point.sentiment_score,
            'sentiment_label': point.sentiment_label,
            'source': point.source
        } for point in self.social_data])
        
        return {
            'avg_sentiment': df['sentiment_score'].mean(),
            'sentiment_distribution': df['sentiment_label'].value_counts().to_dict(),
            'source_distribution': df['source'].value_counts().to_dict(),
            'total_posts': len(df)
        }


# Factory function
def create_social_data_collector(config: Optional[Dict[str, Any]] = None) -> SocialDataCollector:
    """Create a social data collector."""
    return SocialDataCollector(config) 