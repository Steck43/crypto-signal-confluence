#!/usr/bin/env python3
"""
Test script to verify utils modules are working correctly
Run this to check if all imports are working
"""

import sys
import os

# Add src to path
sys.path.append('src')

def test_utils_import():
    """Test importing utils modules"""
    print("🔧 Testing utils module imports...")
    
    try:
        # Test basic import
        import utils
        print("✅ Basic utils import successful")
        
        # Check module status
        status = utils.get_utils_status()
        print(f"\n📊 Module Status:")
        for module, available in status.items():
            if module != 'total_modules':
                status_icon = "✅" if available else "❌"
                print(f"   {status_icon} {module}")
        
        print(f"\n📈 Total modules available: {status['total_modules']}/5")
        
        # Test available modules
        if status['logging_config']:
            print("\n🔧 Testing logging configuration...")
            logger = utils.setup_logging(log_level="INFO", enable_console=True)
            logger.info("Logging system working correctly")
            print("✅ Logging system initialized")
        
        if status['telegram_notifier']:
            print("\n📱 Testing Telegram notifier...")
            notifier = utils.get_telegram_notifier()
            print("✅ Telegram notifier initialized (disabled without tokens)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing utils: {e}")
        return False

def test_paper_trading_system():
    """Test if paper trading system can start"""
    print("\n🎯 Testing paper trading system startup...")
    
    try:
        # Import your main components
        from trading.signal_generator import SignalGenerator
        from trading.paper_trading import PaperTrader
        
        print("✅ Trading modules imported successfully")
        
        # Test signal generator
        generator = SignalGenerator()
        print("✅ Signal generator initialized")
        
        # Test paper trader
        trader = PaperTrader(initial_balance=10000)
        print("✅ Paper trader initialized")
        
        print("\n🚀 Paper trading system ready to start!")
        return True
        
    except Exception as e:
        print(f"❌ Error with paper trading system: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 UTILS MODULE & PAPER TRADING SYSTEM TEST")
    print("=" * 60)
    
    # Test utils
    utils_ok = test_utils_import()
    
    # Test paper trading
    paper_trading_ok = test_paper_trading_system()
    
    print("\n" + "=" * 60)
    print("📋 TEST RESULTS")
    print("=" * 60)
    
    utils_status = "✅ PASS" if utils_ok else "❌ FAIL"
    paper_status = "✅ PASS" if paper_trading_ok else "❌ FAIL"
    
    print(f"Utils Module Test:        {utils_status}")
    print(f"Paper Trading Test:       {paper_status}")
    
    if utils_ok and paper_trading_ok:
        print(f"\n🎉 ALL TESTS PASSED - System is ready!")
        print("📝 Next steps:")
        print("   1. Run your paper trading system")
        print("   2. Add exchange integrations")
        print("   3. Test with live data")
    else:
        print(f"\n⚠️  Some tests failed - check error messages above")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 