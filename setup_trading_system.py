#!/usr/bin/env python3
"""
AI-Driven Crypto Trading System Setup Script

This script sets up the complete trading system with:
- Secure API credential management
- All required API integrations (Alpha Vantage, CoinGecko, FRED, Binance, OKX, Hyperliquid)
- Real-time data collection
- Volume anomaly detection
- Sentiment analysis
- Risk management with Kelly Criterion
- Order execution capabilities

SECURITY NOTE: This script stores API credentials securely using encryption.
The credentials provided will be encrypted and stored safely.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from api.secure_manager import get_secure_api_manager
from api.alpha_vantage import get_alpha_vantage_client
from api.coingecko import get_coingecko_client
from api.fred import get_fred_client
from api.exchanges.binance import BinanceClient
from api.exchanges.okx import OKXClient
from api.exchanges.hyperliquid import HyperliquidClient
from api.rss_collector import CryptoRSSCollector
from analysis.volume_anomaly_detection import VolumeAnomalyDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system_setup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_api_credentials():
    """Set up all API credentials securely from environment variables."""
    print("🔐 Setting up API credentials from environment...")

    api_manager = get_secure_api_manager()

    def env(name: str) -> str:
        return os.getenv(name, "") or ""

    credentials = {
        'alpha_vantage': {
            # Set ALPHA_VANTAGE_API_KEY in your .env file
            'api_key': env('ALPHA_VANTAGE_API_KEY'),
            'rate_limit': int(os.getenv('ALPHA_VANTAGE_RATE_LIMIT', '75')),
        },
        'binance': {
            # Set BINANCE_API_KEY and BINANCE_API_SECRET in your .env file
            'api_key': env('BINANCE_API_KEY'),
            'api_secret': env('BINANCE_API_SECRET'),
            'rate_limit': int(os.getenv('BINANCE_RATE_LIMIT', '1200')),
        },
        'okx': {
            # Set OKX_API_KEY and OKX_API_SECRET in your .env file
            'api_key': env('OKX_API_KEY'),
            'api_secret': env('OKX_API_SECRET'),
            'rate_limit': int(os.getenv('OKX_RATE_LIMIT', '600')),
        },
        'hyperliquid': {
            # Set HYPERLIQUID_PRIVATE_KEY and HYPERLIQUID_WALLET_ADDRESS in your .env file
            'api_key': env('HYPERLIQUID_PRIVATE_KEY'),
            'wallet_address': env('HYPERLIQUID_WALLET_ADDRESS'),
            'rate_limit': int(os.getenv('HYPERLIQUID_RATE_LIMIT', '300')),
        },
        'coingecko': {
            # Set COINGECKO_API_KEY in your .env file
            'api_key': env('COINGECKO_API_KEY'),
            'rate_limit': int(os.getenv('COINGECKO_RATE_LIMIT', '30')),
        },
        'telegram': {
            # Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your .env file
            'bot_token': env('TELEGRAM_BOT_TOKEN'),
            'chat_id': env('TELEGRAM_CHAT_ID'),
            'rate_limit': int(os.getenv('TELEGRAM_RATE_LIMIT', '30')),
        },
        'fred': {
            # Set FRED_API_KEY in your .env file
            'api_key': env('FRED_API_KEY'),
            'rate_limit': int(os.getenv('FRED_RATE_LIMIT', '120')),
        },
    }

    for api_name, creds in credentials.items():
        api_key = creds.get('api_key') or creds.get('bot_token')
        if not api_key:
            continue
        try:
            api_manager.add_api_credentials(
                name=api_name,
                api_key=api_key,
                api_secret=creds.get('api_secret'),
                passphrase=creds.get('passphrase'),
                rate_limit=creds.get('rate_limit', 60),
            )
            print(f"✅ {api_name} credentials added securely")
        except Exception as e:
            print(f"❌ Error adding {api_name} credentials: {e}")

    print("🔐 API credentials setup complete!")
    return api_manager

async def test_api_connections():
    """Test all API connections."""
    print("\n🔍 Testing API connections...")
    
    api_manager = get_secure_api_manager()
    
    # Test Alpha Vantage
    try:
        alpha_client = get_alpha_vantage_client()
        sol_data = await alpha_client.get_crypto_price('SOL')
        if sol_data:
            print(f"✅ Alpha Vantage: SOL price = ${sol_data.price:.2f}")
        else:
            print("⚠️ Alpha Vantage: No data received")
    except Exception as e:
        print(f"❌ Alpha Vantage error: {e}")
    
    # Test CoinGecko
    try:
        coingecko_client = api_manager.get_coingecko_client()
        if coingecko_client:
            global_data = await coingecko_client.get_global_market_data()
            if global_data:
                print(f"✅ CoinGecko: Global market cap = ${global_data.get('total_market_cap', {}).get('usd', 0):,.0f}")
            else:
                print("⚠️ CoinGecko: No data received")
        else:
            print("⚠️ CoinGecko: Client not available")
    except Exception as e:
        print(f"❌ CoinGecko error: {e}")
    
    # Test FRED
    try:
        fred_client = api_manager.get_fred_client()
        if fred_client:
            fed_funds = await fred_client.get_federal_funds_rate()
            if fed_funds and fed_funds.get('current'):
                print(f"✅ FRED: Federal Funds Rate = {fed_funds['current']:.2f}%")
            else:
                print("⚠️ FRED: No data received")
        else:
            print("⚠️ FRED: Client not available")
    except Exception as e:
        print(f"❌ FRED error: {e}")
    
    # Test Binance
    try:
        binance_client = api_manager.get_binance_client()
        if binance_client:
            await binance_client.connect()
            if binance_client.connected:
                sol_data = await binance_client.get_sol_data()
                print(f"✅ Binance: SOL price = ${sol_data['price']:.2f}")
            else:
                print("⚠️ Binance: Connection failed")
        else:
            print("⚠️ Binance: Client not available")
    except Exception as e:
        print(f"❌ Binance error: {e}")
    
    # Test OKX
    try:
        okx_client = api_manager.get_okx_client()
        if okx_client:
            await okx_client.connect()
            if okx_client.connected:
                market_data = await okx_client.get_market_data('SOL-USDT')
                if market_data and market_data.get('price'):
                    print(f"✅ OKX: SOL price = ${market_data['price']:.2f}")
                else:
                    print("⚠️ OKX: No market data received")
            else:
                print("⚠️ OKX: Connection failed")
        else:
            print("⚠️ OKX: Client not available")
    except Exception as e:
        print(f"❌ OKX error: {e}")
    
    # Test Hyperliquid
    try:
        hyperliquid_client = api_manager.get_hyperliquid_client()
        if hyperliquid_client:
            await hyperliquid_client.connect()
            if hyperliquid_client.connected:
                hype_data = await hyperliquid_client.get_hype_data()
                if hype_data and hype_data.get('ticker', {}).get('price'):
                    print(f"✅ Hyperliquid: HYPE price = ${hype_data['ticker']['price']:.4f}")
                else:
                    print("⚠️ Hyperliquid: No HYPE data received")
            else:
                print("⚠️ Hyperliquid: Connection failed")
        else:
            print("⚠️ Hyperliquid: Client not available")
    except Exception as e:
        print(f"❌ Hyperliquid error: {e}")
    
    # Test RSS feeds
    try:
        rss_collector = CryptoRSSCollector()
        articles = await rss_collector.collect_latest_news(hours_back=6)
        if articles:
            print(f"✅ RSS Feeds: Collected {len(articles)} articles")
            bullish_count = sum(1 for a in articles if a.sentiment == 'bullish')
            bearish_count = sum(1 for a in articles if a.sentiment == 'bearish')
            print(f"   📈 Bullish: {bullish_count} | 📉 Bearish: {bearish_count}")
        else:
            print("⚠️ RSS Feeds: No articles collected")
    except Exception as e:
        print(f"❌ RSS Feeds error: {e}")
    
    # Test all API validations
    active_apis = api_manager.get_active_apis()
    for api_name in active_apis:
        try:
            is_valid = await api_manager.validate_api_connection(api_name)
            status = "✅" if is_valid else "❌"
            print(f"{status} {api_name}: {'Connected' if is_valid else 'Failed'}")
        except Exception as e:
            print(f"❌ {api_name} validation error: {e}")
    
    print("🔍 API connection tests complete!")

async def test_exchange_functionality():
    """Test exchange-specific functionality."""
    print("\n🏦 Testing exchange functionality...")
    
    api_manager = get_secure_api_manager()
    
    # Test Binance functionality
    try:
        binance_client = api_manager.get_binance_client()
        if binance_client:
            # Test market data
            sol_data = await binance_client.get_sol_data()
            print(f"✅ Binance SOL Data: Price=${sol_data['price']:.2f}, Volume={sol_data['volume']:,.0f}")
            
            # Test order book
            orderbook = await binance_client.get_order_book('SOLUSDT', limit=5)
            if orderbook['bids'] and orderbook['asks']:
                print(f"✅ Binance Order Book: Top bid=${orderbook['bids'][0][0]:.2f}, Top ask=${orderbook['asks'][0][0]:.2f}")
            
            # Test recent trades
            trades = await binance_client.get_recent_trades('SOLUSDT', limit=5)
            if trades:
                print(f"✅ Binance Recent Trades: {len(trades)} trades retrieved")
    except Exception as e:
        print(f"❌ Binance functionality test error: {e}")
    
    # Test OKX functionality
    try:
        okx_client = api_manager.get_okx_client()
        if okx_client:
            # Test market data
            market_data = await okx_client.get_market_data('SOL-USDT')
            print(f"✅ OKX SOL Data: Price=${market_data['price']:.2f}, Volume={market_data['volume']:,.0f}")
            
            # Test order book
            orderbook = await okx_client.get_order_book('SOL-USDT', depth=5)
            if orderbook['bids'] and orderbook['asks']:
                print(f"✅ OKX Order Book: Top bid=${orderbook['bids'][0][0]:.2f}, Top ask=${orderbook['asks'][0][0]:.2f}")
    except Exception as e:
        print(f"❌ OKX functionality test error: {e}")
    
    # Test Hyperliquid functionality
    try:
        hyperliquid_client = api_manager.get_hyperliquid_client()
        if hyperliquid_client:
            # Test HYPE data
            hype_data = await hyperliquid_client.get_hype_data()
            if hype_data.get('ticker'):
                ticker = hype_data['ticker']
                print(f"✅ Hyperliquid HYPE Data: Price=${ticker['price']:.4f}, Volume={ticker['volume']:,.0f}")
            
            # Test market status
            status = await hyperliquid_client.get_market_status()
            print(f"✅ Hyperliquid Market Status: {status['status']}")
    except Exception as e:
        print(f"❌ Hyperliquid functionality test error: {e}")
    
    print("🏦 Exchange functionality tests complete!")

async def test_economic_data():
    """Test economic data functionality."""
    print("\n📊 Testing economic data functionality...")
    
    api_manager = get_secure_api_manager()
    
    # Test FRED economic data
    try:
        fred_client = api_manager.get_fred_client()
        if fred_client:
            # Test comprehensive economic data
            economic_data = await fred_client.get_comprehensive_economic_data()
            
            if economic_data:
                print("✅ FRED Economic Data Retrieved:")
                for indicator, data in economic_data.items():
                    if data and data.get('current') is not None:
                        print(f"   📈 {indicator}: {data['current']:.2f}")
                    else:
                        print(f"   ⚠️ {indicator}: No data")
            else:
                print("⚠️ FRED: No economic data received")
        else:
            print("⚠️ FRED: Client not available")
    except Exception as e:
        print(f"❌ FRED economic data error: {e}")
    
    # Test CoinGecko market data
    try:
        coingecko_client = api_manager.get_coingecko_client()
        if coingecko_client:
            # Test trending coins
            trending = await coingecko_client.get_trending_coins()
            if trending:
                print(f"✅ CoinGecko Trending Coins: {len(trending)} coins")
                for coin in trending[:3]:  # Show top 3
                    print(f"   🚀 {coin['name']} ({coin['symbol']}) - Score: {coin['score']:.2f}")
            
            # Test meme coins
            meme_coins = await coingecko_client.get_meme_coins(limit=5)
            if meme_coins:
                print(f"✅ CoinGecko Meme Coins: {len(meme_coins)} coins")
                for coin in meme_coins[:3]:  # Show top 3
                    print(f"   🎭 {coin.name} ({coin.symbol}) - ${coin.current_price:.6f}")
        else:
            print("⚠️ CoinGecko: Client not available")
    except Exception as e:
        print(f"❌ CoinGecko market data error: {e}")
    
    print("📊 Economic data tests complete!")

async def setup_volume_anomaly_detection():
    """Set up volume anomaly detection with sample data."""
    print("\n🔍 Setting up volume anomaly detection...")
    
    try:
        # Create volume anomaly detector
        detector = VolumeAnomalyDetector(
            contamination=0.05,
            enable_feature_selection=True
        )
        
        # For demonstration, we'll create sample data
        # In production, this would use real market data
        import pandas as pd
        import numpy as np
        
        # Generate sample SOL trading data
        dates = pd.date_range(start='2024-01-01', periods=500, freq='H')
        np.random.seed(42)
        
        # Create realistic-looking volume and price data
        base_volume = 1000000
        base_price = 100
        
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'symbol': 'SOL',
            'volume': base_volume * (1 + np.random.normal(0, 0.5, 500)).clip(0.1, 10),
            'price': base_price * np.cumprod(1 + np.random.normal(0, 0.02, 500)),
            'high': base_price * np.cumprod(1 + np.random.normal(0, 0.02, 500)) * 1.02,
            'low': base_price * np.cumprod(1 + np.random.normal(0, 0.02, 500)) * 0.98
        })
        
        # Add some artificial anomalies
        anomaly_indices = np.random.choice(len(sample_data), 25, replace=False)
        sample_data.loc[anomaly_indices, 'volume'] *= np.random.uniform(5, 15, 25)
        
        # Fit the detector
        detector.fit(sample_data)
        
        # Test predictions
        recent_data = sample_data.tail(50)
        anomaly_results = detector.predict(recent_data)
        
        detected_anomalies = [r for r in anomaly_results if r.is_anomaly]
        
        print(f"✅ Volume Anomaly Detection setup complete!")
        print(f"   📊 Trained on {len(sample_data)} samples")
        print(f"   🔍 Detected {len(detected_anomalies)} anomalies in recent data")
        
        # Show feature importance
        importance = detector.get_feature_importance()
        if importance:
            print("   📈 Top 5 features:")
            for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"      • {feature}: {score:.3f}")
        
        return detector
        
    except Exception as e:
        print(f"❌ Volume anomaly detection setup error: {e}")
        return None

async def demonstrate_system_capabilities():
    """Demonstrate the full system capabilities."""
    print("\n🚀 Demonstrating system capabilities...")
    
    api_manager = get_secure_api_manager()
    
    # Demonstrate multi-exchange price comparison
    print("\n💰 Multi-Exchange Price Comparison:")
    try:
        # Get SOL prices from different sources
        prices = {}
        
        # Alpha Vantage
        try:
            alpha_client = get_alpha_vantage_client()
            sol_data = await alpha_client.get_crypto_price('SOL')
            if sol_data:
                prices['Alpha Vantage'] = sol_data.price
        except:
            pass
        
        # Binance
        try:
            binance_client = api_manager.get_binance_client()
            if binance_client:
                sol_data = await binance_client.get_sol_data()
                prices['Binance'] = sol_data['price']
        except:
            pass
        
        # OKX
        try:
            okx_client = api_manager.get_okx_client()
            if okx_client:
                market_data = await okx_client.get_market_data('SOL-USDT')
                prices['OKX'] = market_data['price']
        except:
            pass
        
        # Display price comparison
        if prices:
            print("   📊 SOL/USDT Prices:")
            for exchange, price in prices.items():
                print(f"      • {exchange}: ${price:.2f}")
            
            # Calculate spread
            if len(prices) > 1:
                min_price = min(prices.values())
                max_price = max(prices.values())
                spread = ((max_price - min_price) / min_price) * 100
                print(f"      📈 Spread: {spread:.2f}%")
        else:
            print("   ⚠️ No price data available")
            
    except Exception as e:
        print(f"   ❌ Price comparison error: {e}")
    
    # Demonstrate economic data integration
    print("\n📊 Economic Data Integration:")
    try:
        fred_client = api_manager.get_fred_client()
        if fred_client:
            # Get key economic indicators
            indicators = {
                'Federal Funds Rate': await fred_client.get_federal_funds_rate(),
                'Inflation Rate': await fred_client.get_inflation_data(),
                'Unemployment Rate': await fred_client.get_unemployment_rate(),
                'VIX': await fred_client.get_vix_data()
            }
            
            print("   📈 Key Economic Indicators:")
            for name, data in indicators.items():
                if data and data.get('current') is not None:
                    print(f"      • {name}: {data['current']:.2f}")
                else:
                    print(f"      • {name}: No data")
        else:
            print("   ⚠️ FRED client not available")
            
    except Exception as e:
        print(f"   ❌ Economic data error: {e}")
    
    # Demonstrate market sentiment
    print("\n📰 Market Sentiment Analysis:")
    try:
        rss_collector = CryptoRSSCollector()
        articles = await rss_collector.collect_latest_news(hours_back=12)
        
        if articles:
            sentiment_counts = {}
            for article in articles:
                sentiment = article.sentiment
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            
            print("   📊 Recent News Sentiment:")
            for sentiment, count in sentiment_counts.items():
                emoji = "📈" if sentiment == "bullish" else "📉" if sentiment == "bearish" else "➡️"
                print(f"      {emoji} {sentiment.capitalize()}: {count} articles")
        else:
            print("   ⚠️ No recent news articles")
            
    except Exception as e:
        print(f"   ❌ Sentiment analysis error: {e}")
    
    print("\n🚀 System capabilities demonstration complete!")

def create_env_file():
    """Create environment configuration file from .env.example template."""
    print("\n📝 Creating environment configuration...")

    example_path = Path(".env.example")
    if not example_path.exists():
        print("❌ .env.example not found; cannot create .env")
        return

    try:
        env_content = example_path.read_text()
        env_content = env_content.replace(
            "# Environment Configuration Template",
            "# Local environment — copy values from .env.example and fill in secrets",
        )
        Path(".env").write_text(env_content)
        print("✅ Environment configuration file created (.env) from .env.example")
        print("   Set ENCRYPTION_PASSWORD and API keys before running credential setup.")
    except Exception as e:
        print(f"❌ Error creating environment file: {e}")

def print_security_summary():
    """Print security features summary."""
    print("\n🔒 Security Features Summary:")
    print("   ✅ Fernet encryption for all API credentials")
    print("   ✅ PBKDF2 key derivation with 100,000 iterations")
    print("   ✅ Zero hardcoded credentials in source code")
    print("   ✅ Secure credential rotation capabilities")
    print("   ✅ Comprehensive logging without exposing keys")
    print("   ✅ Rate limiting and error handling")
    print("   ✅ Automatic usage tracking and health monitoring")
    print("   ✅ Encrypted credential storage")
    print("   ✅ API connection validation")
    print("   ✅ Graceful error handling and fallbacks")

def print_system_overview():
    """Print comprehensive system overview."""
    print("\n🏛️ Institutional Trading System Overview:")
    print("\n📊 Data Sources:")
    print("   • Alpha Vantage: Technical indicators, crypto data, news sentiment")
    print("   • CoinGecko: Meme coin data, market validation")
    print("   • FRED: Macro economic indicators")
    print("   • RSS Feeds: Real-time news sentiment analysis")
    print("   • Telegram: Notifications and alerts")
    
    print("\n🏦 Exchange Integrations:")
    print("   • Binance: SOL/USDT trading, market data")
    print("   • OKX: Alternative market data, order execution")
    print("   • Hyperliquid: HYPE trading, low-fee execution")
    
    print("\n🧠 AI & Machine Learning:")
    print("   • Volume Anomaly Detection (5 algorithms)")
    print("   • Multi-source Sentiment Analysis")
    print("   • Technical Analysis (50+ indicators)")
    print("   • Macro Regime Detection")
    print("   • Ensemble Learning with Adaptive Weights")
    
    print("\n🛡️ Risk Management:")
    print("   • Kelly Criterion position sizing")
    print("   • Dynamic position sizing (5-15% per trade)")
    print("   • Maximum 3 concurrent positions")
    print("   • Stop-loss and take-profit automation")
    print("   • Real-time drawdown monitoring")
    
    print("\n📈 Performance Targets:")
    print("   • Monthly Return: 25-40%")
    print("   • Sharpe Ratio: >2.0")
    print("   • Maximum Drawdown: <15%")
    print("   • Win Rate: >60%")
    print("   • Risk-Adjusted Returns: Optimized")

async def main():
    """Main setup function."""
    print("🚀 AI-Driven Crypto Trading System Setup")
    print("=" * 50)
    
    try:
        # Setup API credentials
        api_manager = setup_api_credentials()
        
        # Test API connections
        await test_api_connections()
        
        # Test exchange functionality
        await test_exchange_functionality()
        
        # Test economic data
        await test_economic_data()
        
        # Setup volume anomaly detection
        detector = await setup_volume_anomaly_detection()
        
        # Demonstrate system capabilities
        await demonstrate_system_capabilities()
        
        # Create environment file
        create_env_file()
        
        # Print summaries
        print_security_summary()
        print_system_overview()
        
        print("\n🎉 Setup Complete!")
        print("\n📋 Next Steps:")
        print("   1. Review the .env file for configuration")
        print("   2. Run: python test_simplified_system.py")
        print("   3. Start paper trading: python paper_trading_system.py")
        print("   4. Monitor performance and adjust parameters")
        
        print("\n🔒 Security Note:")
        print("   All API credentials are encrypted and stored securely.")
        print("   Never share your encryption password or API keys.")
        print("   The system includes comprehensive error handling and fallbacks.")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        logger.error(f"Setup failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n✅ Trading system setup completed successfully!")
    else:
        print("\n❌ Trading system setup failed. Check logs for details.")
        sys.exit(1)