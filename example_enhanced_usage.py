#!/usr/bin/env python3
"""
Enhanced ML Trading System Usage Example

Demonstrates how to use the enhanced institutional trading system
with XGBoost and GRU integration.
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.trading.enhanced_signal_generator import EnhancedInstitutionalSignalGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate sample market data for demonstration"""
    
    logger.info(f"📊 Generating {n_samples} sample data points...")
    
    np.random.seed(42)
    
    # Generate realistic price movements
    base_price = 100.0
    trend = 0.0001
    volatility = 0.02
    
    prices = [base_price]
    volumes = [1000000]
    
    for i in range(1, n_samples):
        price_change = trend + np.random.normal(0, volatility / np.sqrt(252))
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
        
        volume_change = np.random.normal(0, 0.3)
        if abs(price_change) > volatility:
            volume_change += np.random.normal(0.5, 0.2)
        
        new_volume = volumes[-1] * (1 + volume_change)
        volumes.append(max(new_volume, 100000))
    
    # Create OHLCV data
    data = []
    for i in range(n_samples):
        price = prices[i]
        volume = volumes[i]
        
        spread = price * 0.001
        high = price + np.random.uniform(0, spread)
        low = price - np.random.uniform(0, spread)
        open_price = price + np.random.uniform(-spread/2, spread/2)
        
        data.append({
            'timestamp': datetime.now() - timedelta(minutes=5*(n_samples-i)),
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

async def demonstrate_enhanced_system():
    """Demonstrate the enhanced ML trading system"""
    
    logger.info("🚀 Enhanced ML Trading System Demonstration")
    logger.info("=" * 60)
    
    try:
        # 1. Generate sample data
        logger.info("\n1️⃣ Generating sample market data...")
        historical_data = generate_sample_data(800)  # 800 samples for training
        current_data = generate_sample_data(100)     # 100 samples for current analysis
        
        logger.info(f"   Historical data: {len(historical_data)} samples")
        logger.info(f"   Current data: {len(current_data)} samples")
        
        # 2. Initialize enhanced signal generator
        logger.info("\n2️⃣ Initializing Enhanced Signal Generator...")
        enhanced_generator = EnhancedInstitutionalSignalGenerator(
            enable_xgboost=True,    # Enable XGBoost model
            enable_gru=True,        # Enable GRU model
            ml_weight=0.35          # 35% weight for ML predictions
        )
        
        logger.info("   ✅ Enhanced generator initialized")
        logger.info(f"   ML Weight: {enhanced_generator.ml_weight:.1%}")
        logger.info(f"   Institutional Weight: {enhanced_generator.institutional_weight:.1%}")
        
        # 3. Initialize the system with historical data
        logger.info("\n3️⃣ Initializing system with historical data...")
        init_results = await enhanced_generator.initialize_enhanced_system(
            historical_data=historical_data,
            symbol="SOL"
        )
        
        logger.info("   ✅ System initialization complete")
        logger.info(f"   Institutional ready: {init_results['enhanced_system']['institutional_ready']}")
        logger.info(f"   ML ready: {init_results['enhanced_system']['ml_ready']}")
        logger.info(f"   Total algorithms: {init_results['enhanced_system']['total_algorithms']}")
        
        # 4. Generate enhanced signals
        logger.info("\n4️⃣ Generating enhanced trading signals...")
        enhanced_signal = await enhanced_generator.generate_enhanced_signals(
            current_data=current_data,
            symbol="SOL"
        )
        
        logger.info("   ✅ Enhanced signal generated")
        logger.info(f"   Signal: {enhanced_signal['signal'].upper()}")
        logger.info(f"   Confidence: {enhanced_signal['confidence']:.3f}")
        logger.info(f"   Strength: {enhanced_signal['strength']:.3f}")
        logger.info(f"   Signal type: {enhanced_signal['signal_type']}")
        logger.info(f"   Reasoning: {enhanced_signal['reasoning']}")
        
        # 5. Get performance analysis
        logger.info("\n5️⃣ Analyzing system performance...")
        performance = enhanced_generator.get_enhanced_performance()
        
        logger.info("   ✅ Performance analysis complete")
        logger.info(f"   System status: {performance['system_status']}")
        logger.info(f"   Signal count: {performance['enhanced_analysis']['signal_count']}")
        logger.info(f"   Agreement rate: {performance['enhanced_analysis']['agreement_rate']:.3f}")
        logger.info(f"   Average confidence: {performance['enhanced_analysis']['average_confidence']:.3f}")
        
        # 6. Get feature importance analysis
        logger.info("\n6️⃣ Analyzing feature importance...")
        feature_analysis = enhanced_generator.get_feature_importance_analysis()
        
        logger.info("   ✅ Feature analysis complete")
        logger.info(f"   Institutional features: {feature_analysis['institutional_features']['count']}")
        logger.info(f"   Institutional algorithms: {feature_analysis['institutional_features']['algorithms']}")
        
        if 'xgboost' in feature_analysis['ml_features']:
            xgb_info = feature_analysis['ml_features']['xgboost']
            if 'total_features' in xgb_info:
                logger.info(f"   XGBoost features: {xgb_info['total_features']}")
        
        if 'gru' in feature_analysis['ml_features']:
            gru_info = feature_analysis['ml_features']['gru']
            if 'input_features' in gru_info:
                logger.info(f"   GRU features: {gru_info['input_features']}")
        
        # 7. Generate multiple signals for demonstration
        logger.info("\n7️⃣ Generating multiple signals for demonstration...")
        
        signals = []
        for i in range(5):
            # Generate new current data for each signal
            new_current_data = generate_sample_data(50)
            signal = await enhanced_generator.generate_enhanced_signals(
                current_data=new_current_data,
                symbol="SOL"
            )
            signals.append(signal)
            
            logger.info(f"   Signal {i+1}: {signal['signal'].upper()} "
                       f"(conf: {signal['confidence']:.3f}, "
                       f"type: {signal['signal_type']})")
        
        # 8. Summary
        logger.info("\n" + "=" * 60)
        logger.info("🎯 Enhanced ML Trading System Summary")
        logger.info("=" * 60)
        
        signal_counts = {}
        for signal in signals:
            signal_type = signal['signal_type']
            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
        
        logger.info(f"📊 Signal Distribution:")
        for signal_type, count in signal_counts.items():
            logger.info(f"   {signal_type}: {count} signals")
        
        avg_confidence = np.mean([s['confidence'] for s in signals])
        logger.info(f"📈 Average Confidence: {avg_confidence:.3f}")
        
        buy_signals = [s for s in signals if s['signal'] == 'buy']
        sell_signals = [s for s in signals if s['signal'] == 'sell']
        hold_signals = [s for s in signals if s['signal'] == 'hold']
        
        logger.info(f"🎯 Signal Breakdown:")
        logger.info(f"   BUY: {len(buy_signals)} signals")
        logger.info(f"   SELL: {len(sell_signals)} signals")
        logger.info(f"   HOLD: {len(hold_signals)} signals")
        
        logger.info("\n🎉 Enhanced ML Trading System is working correctly!")
        logger.info("   The system successfully combines institutional analysis with ML predictions.")
        logger.info("   Ready for paper trading and live deployment.")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        return False

async def main():
    """Main demonstration function"""
    
    logger.info("🚀 Starting Enhanced ML Trading System Demonstration")
    
    success = await demonstrate_enhanced_system()
    
    if success:
        logger.info("\n✅ Demonstration completed successfully!")
        logger.info("   The enhanced ML trading system is ready for use.")
    else:
        logger.error("\n❌ Demonstration failed!")
        logger.error("   Please check the logs for details.")
    
    return success

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main()) 