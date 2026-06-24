"""
Sentiment Analysis Module

Advanced NLP and sentiment analysis including:
- Twitter sentiment collection with CryptoBERT (optional)
- Multi-head attention mechanisms
- Cross-asset sentiment propagation
- Real-time sentiment fusion
"""

# Import core sentiment analyzer (always available)
from .sentiment_analyzer import SentimentAnalyzer

# Optional imports - only if dependencies are available
try:
    from .twitter_collector import TwitterSentimentCollector
    TWITTER_AVAILABLE = True
except ImportError:
    print("Twitter sentiment collector not available (tweepy not installed)")
    TWITTER_AVAILABLE = False

try:
    from .transformer_analyzer import TransformerSentimentAnalyzer
    TRANSFORMER_AVAILABLE = True
except ImportError:
    print("Transformer sentiment analyzer not available")
    TRANSFORMER_AVAILABLE = False

try:
    from .cryptobert_finetuner import CryptoBERTFineTuner
    CRYPTOBERT_AVAILABLE = True
except ImportError:
    print("CryptoBERT fine-tuner not available")
    CRYPTOBERT_AVAILABLE = False

try:
    from .sentiment_fusion import SentimentFusionEngine
    FUSION_AVAILABLE = True
except ImportError:
    print("Sentiment fusion engine not available")
    FUSION_AVAILABLE = False

__all__ = [
    "SentimentAnalyzer",  # Always available
]

# Add optional components if available
if TWITTER_AVAILABLE:
    __all__.append("TwitterSentimentCollector")
if TRANSFORMER_AVAILABLE:
    __all__.append("TransformerSentimentAnalyzer")
if CRYPTOBERT_AVAILABLE:
    __all__.append("CryptoBERTFineTuner")
if FUSION_AVAILABLE:
    __all__.append("SentimentFusionEngine")