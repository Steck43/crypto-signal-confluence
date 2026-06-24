# 🚀 AI Trading System - Quick Start Guide

## ⚡ 5-Minute Setup

### 1. Clone and Setup
```bash
git clone <repository-url>
cd ai-crypto-trading

# Copy environment template
cp .env.example .env
```

### 2. Configure API Keys

Edit `.env` with your credentials:
```bash
# Twitter API (for sentiment analysis)
TWITTER_BEARER_TOKEN=your_bearer_token_here
TWITTER_API_KEY=your_api_key_here
TWITTER_API_SECRET=your_api_secret_here

# Exchange APIs (start with sandbox/testnet)
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here
BINANCE_SANDBOX=true
```

### 3. Launch System
```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f trading-app
```

### 4. Access Services

- **📊 Trading Dashboard**: http://localhost:8000
- **📈 Grafana Monitoring**: http://localhost:3000 (credentials from `GF_SECURITY_ADMIN_PASSWORD` in `.env`)
- **🔬 Jupyter Analysis**: http://localhost:8888 (token from `JUPYTER_TOKEN` in `.env`)
- **📊 Prometheus Metrics**: http://localhost:9090

## 🎯 System Status Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### System Metrics
```bash
curl http://localhost:8000/metrics
```

### Volume Anomaly Status
```bash
curl http://localhost:8000/models/volume-anomaly/status
```

### Sentiment Summary
```bash
curl http://localhost:8000/models/sentiment/summary
```

## 🔧 Configuration Options

### Trading Parameters
```bash
# In .env file
MAX_OPEN_TRADES=5
STAKE_AMOUNT=100
STAKE_CURRENCY=USDT
DRY_RUN=true  # Set to false for live trading
```

### ML Model Settings
```bash
# Volume Anomaly Detection
VOLUME_CONTAMINATION=0.05
VOLUME_N_ESTIMATORS=100

# MAML Configuration
MAML_INNER_LR=0.01
MAML_META_LR=0.001
MAML_INNER_STEPS=5
```

## 📱 API Usage Examples

### Start System
```bash
curl -X POST http://localhost:8000/start
```

### Stop System
```bash
curl -X POST http://localhost:8000/stop
```

### Trigger Model Retraining
```bash
curl -X POST http://localhost:8000/models/retrain
```

## 🛠️ Development Mode

For development with hot reloading:
```bash
# Start in development mode
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Or run locally
pip install -r requirements.txt
python -m src.main --debug
```

## 📊 Monitoring & Alerts

### Grafana Dashboards
Access Grafana at http://localhost:3000:
- Trading Performance Dashboard
- System Health Monitoring
- Model Performance Metrics

### Log Analysis
```bash
# View real-time logs
docker-compose logs -f trading-app

# Search for errors
docker-compose logs trading-app | grep ERROR

# View specific service logs
docker-compose logs sentiment-worker
docker-compose logs volume-worker
```

## 🚨 Troubleshooting

### Common Issues

**1. Services not starting:**
```bash
# Check service status
docker-compose ps

# Restart specific service
docker-compose restart trading-app
```

**2. Database connection issues:**
```bash
# Check PostgreSQL status
docker-compose logs postgres

# Reset database
docker-compose down -v
docker-compose up -d
```

**3. API credentials not working:**
```bash
# Validate configuration
curl http://localhost:8000/status
```

### Performance Optimization

**For production deployment:**
```bash
# Use production compose file
docker-compose -f docker-compose.prod.yml up -d

# Scale workers
docker-compose up -d --scale sentiment-worker=3
```

## 📚 Next Steps

1. **📈 Backtesting**: Use Freqtrade for strategy backtesting
2. **🔐 Security**: Configure SSL certificates and firewall
3. **📊 Monitoring**: Set up alerts and notifications
4. **🔄 CI/CD**: Implement automated deployment pipeline

## ⚠️ Important Notes

- **Start with DRY_RUN=true** for testing
- **Use sandbox APIs** initially
- **Monitor system resources** during operation
- **Keep API keys secure** and rotate regularly

---

**🎯 The system is now ready for AI-driven cryptocurrency trading!**