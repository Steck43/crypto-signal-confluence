# 🏛️ Institutional Trading System Setup Guide

## 🚀 Quick Start

### 1. **Install Dependencies**
```bash
py -m pip install numpy pandas scikit-learn requests
```

### 2. **Test the System**
```bash
py test_simplified_system.py
```

### 3. **Run Paper Trading (24/7)**
```bash
py paper_trading_system.py
```

---

## 📱 Telegram Setup (Optional but Recommended)

### Step 1: Create Telegram Bot
1. Open Telegram and search for `@BotFather`
2. Send `/newbot`
3. Choose a name for your bot (e.g., "Institutional Trading Bot")
4. Choose a username (e.g., "my_trading_bot")
5. Save the bot token (looks like: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

### Step 2: Get Your Chat ID
1. Start a chat with your bot
2. Send any message to the bot
3. Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
4. Find your `chat_id` in the response

### Step 3: Set Environment Variables
**Windows (PowerShell):**
```powershell
$env:TELEGRAM_BOT_TOKEN="your_bot_token_here"
$env:TELEGRAM_CHAT_ID="your_chat_id_here"
```

**Windows (Command Prompt):**
```cmd
set TELEGRAM_BOT_TOKEN=your_bot_token_here
set TELEGRAM_CHAT_ID=your_chat_id_here
```

**Or create a `.env` file:**
```
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

---

## 🎯 Running the System

### **Option 1: Quick Test**
```bash
py test_simplified_system.py
```
- Tests the institutional system
- Shows all 5 algorithms and 66 features
- Verifies competitive advantage

### **Option 2: Continuous Paper Trading**
```bash
py paper_trading_system.py
```
- Runs 24/7 paper trading
- Generates signals every 5 minutes
- Sends Telegram alerts for strong signals
- Tracks performance and retrains models

### **Option 3: VSCode Integration**
1. Open the project in VSCode
2. Install Python extension
3. Create launch configuration:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Paper Trading",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/paper_trading_system.py",
            "console": "integratedTerminal",
            "python": "py"
        }
    ]
}
```

---

## 📊 What You'll Get

### **Telegram Notifications:**
- 🚨 **Trading Signals**: BUY/SELL alerts with strength and confidence
- 🔍 **Volume Anomalies**: Institutional-grade anomaly detection
- 📊 **Performance Updates**: Win rate, balance, drawdown
- 🏛️ **System Status**: Model health and algorithm weights

### **Paper Trading Features:**
- 💰 **Starting Balance**: $10,000
- 📈 **Real-time Signals**: Every 5 minutes
- 🔄 **Model Retraining**: Every 24 hours
- 📊 **Performance Tracking**: Win rate, PnL, drawdown
- 🛡️ **Risk Management**: Position sizing and stop-loss

### **Institutional Advantages:**
- 🏛️ **5 ML Algorithms**: Isolation Forest, LOF, Mahalanobis, SPC, One-Class SVM
- 🔧 **66 Engineered Features**: Entropy, mutual information, volume profiles
- ⚖️ **Ensemble Weighting**: Adaptive algorithm weights
- 🎯 **Market Regime Detection**: High volatility, trending, normal

---

## 🔧 Configuration Options

### **Trading Parameters**
Edit `paper_trading_system.py`:
```python
SYMBOL = "SOL"              # Trading symbol
INTERVAL_MINUTES = 5        # Signal interval
STARTING_BALANCE = 10000    # Paper trading balance
```

### **Signal Thresholds**
Edit `src/trading/signal_generator.py`:
```python
self.signal_threshold = 0.7  # Minimum signal strength
```

### **Telegram Alerts**
Edit `paper_trading_system.py`:
```python
# Send alerts for signals with strength > 0.7 and confidence > 0.6
if signal_strength > 0.7 and signal_confidence > 0.6:
    self.telegram.send_signal_alert(signal, self.symbol)
```

---

## 📈 Performance Monitoring

### **Log Files**
- `paper_trading_SOL_YYYYMMDD.log` - Daily trading logs
- Console output - Real-time signal generation

### **Telegram Reports**
- 📊 Performance updates every 50 trades
- 🔄 Model retraining notifications
- 🚨 Strong signal alerts
- 🔚 Session summary on shutdown

### **Key Metrics**
- **Win Rate**: Percentage of profitable trades
- **Total PnL**: Cumulative profit/loss
- **Max Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted returns

---

## 🛠️ Troubleshooting

### **Common Issues:**

1. **Import Errors**
   ```bash
   py -m pip install numpy pandas scikit-learn requests
   ```

2. **Telegram Not Working**
   - Check bot token and chat ID
   - Ensure bot is started in Telegram
   - Test with: `py src/utils/telegram_notifier.py`

3. **System Not Initializing**
   - Check Python version: `py --version`
   - Verify dependencies: `py -m pip list`

4. **Performance Issues**
   - Reduce signal interval (increase `INTERVAL_MINUTES`)
   - Disable Telegram for faster execution
   - Use smaller historical data window

---

## 🎯 Next Steps

### **Immediate Actions:**
1. ✅ Test the system: `py test_simplified_system.py`
2. 📱 Set up Telegram (optional)
3. 🚀 Start paper trading: `py paper_trading_system.py`
4. 📊 Monitor performance for 24-48 hours

### **Advanced Features:**
- 🔗 Connect real market data feeds
- 💰 Implement real trading execution
- 📊 Add more sophisticated risk management
- 🤖 Integrate with additional exchanges

---

## 🏆 Your Competitive Advantage

You now have:
- ✅ **5 Advanced ML Algorithms** working in ensemble
- ✅ **66 Institutional Features** (entropy, mutual information, volume profiles)
- ✅ **Professional Error Handling** with fallbacks
- ✅ **Market Regime Detection** and adaptive learning
- ✅ **Real-time Telegram Notifications**
- ✅ **Continuous Paper Trading** with performance tracking

This is **exactly** the kind of system that separates institutional traders from retail traders! 🎯 