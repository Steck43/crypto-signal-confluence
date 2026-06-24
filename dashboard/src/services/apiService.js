import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

class ApiService {
  constructor() {
    this.api = axios.create({
      baseURL: API_BASE_URL,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.api.interceptors.request.use(
      (config) => {
        console.log(`🌐 API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('❌ API Request Error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.api.interceptors.response.use(
      (response) => {
        console.log(`✅ API Response: ${response.status} ${response.config.url}`);
        return response;
      },
      (error) => {
        console.error('❌ API Response Error:', error.response?.status, error.response?.data);
        return Promise.reject(error);
      }
    );
  }

  // System endpoints
  async getSystemStatus() {
    try {
      const response = await this.api.get('/status');
      return response.data;
    } catch (error) {
      console.error('Failed to get system status:', error);
      throw error;
    }
  }

  async getHealthCheck() {
    try {
      const response = await this.api.get('/health');
      return response.data;
    } catch (error) {
      console.error('Failed to get health check:', error);
      throw error;
    }
  }

  async getMetrics() {
    try {
      const response = await this.api.get('/metrics');
      return response.data;
    } catch (error) {
      console.error('Failed to get metrics:', error);
      throw error;
    }
  }

  // Trading control
  async startTrading() {
    try {
      const response = await this.api.post('/start');
      return response.data;
    } catch (error) {
      console.error('Failed to start trading:', error);
      throw error;
    }
  }

  async stopTrading() {
    try {
      const response = await this.api.post('/stop');
      return response.data;
    } catch (error) {
      console.error('Failed to stop trading:', error);
      throw error;
    }
  }

  // Model endpoints
  async getVolumeAnomalyStatus() {
    try {
      const response = await this.api.get('/models/volume-anomaly/status');
      return response.data;
    } catch (error) {
      console.error('Failed to get volume anomaly status:', error);
      throw error;
    }
  }

  async getSentimentSummary() {
    try {
      const response = await this.api.get('/models/sentiment/summary');
      return response.data;
    } catch (error) {
      console.error('Failed to get sentiment summary:', error);
      throw error;
    }
  }

  async retrainModels() {
    try {
      const response = await this.api.post('/models/retrain');
      return response.data;
    } catch (error) {
      console.error('Failed to retrain models:', error);
      throw error;
    }
  }

  // Market data
  async getMarketData(symbols = ['BTC', 'ETH', 'SOL']) {
    try {
      const response = await this.api.get('/market-data', {
        params: { symbols: symbols.join(',') }
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get market data:', error);
      throw error;
    }
  }

  async getPriceHistory(symbol, timeframe = '1h', limit = 100) {
    try {
      const response = await this.api.get(`/price-history/${symbol}`, {
        params: { timeframe, limit }
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get price history:', error);
      throw error;
    }
  }

  // Sentiment data
  async getSentimentData(symbol, hours = 24) {
    try {
      const response = await this.api.get(`/sentiment/${symbol}`, {
        params: { hours }
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get sentiment data:', error);
      throw error;
    }
  }

  // Portfolio data
  async getPortfolioStatus() {
    try {
      const response = await this.api.get('/portfolio/status');
      return response.data;
    } catch (error) {
      console.error('Failed to get portfolio status:', error);
      throw error;
    }
  }

  async getPositions() {
    try {
      const response = await this.api.get('/portfolio/positions');
      return response.data;
    } catch (error) {
      console.error('Failed to get positions:', error);
      throw error;
    }
  }

  async getPerformance(period = '24h') {
    try {
      const response = await this.api.get('/portfolio/performance', {
        params: { period }
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get performance:', error);
      throw error;
    }
  }

  // Alerts and notifications
  async getAlerts() {
    try {
      const response = await this.api.get('/alerts');
      return response.data;
    } catch (error) {
      console.error('Failed to get alerts:', error);
      throw error;
    }
  }

  async createAlert(alertData) {
    try {
      const response = await this.api.post('/alerts', alertData);
      return response.data;
    } catch (error) {
      console.error('Failed to create alert:', error);
      throw error;
    }
  }

  async deleteAlert(alertId) {
    try {
      const response = await this.api.delete(`/alerts/${alertId}`);
      return response.data;
    } catch (error) {
      console.error('Failed to delete alert:', error);
      throw error;
    }
  }

  // Configuration
  async getConfig() {
    try {
      const response = await this.api.get('/config');
      return response.data;
    } catch (error) {
      console.error('Failed to get config:', error);
      throw error;
    }
  }

  async updateConfig(configData) {
    try {
      const response = await this.api.put('/config', configData);
      return response.data;
    } catch (error) {
      console.error('Failed to update config:', error);
      throw error;
    }
  }

  // Logs
  async getLogs(level = 'INFO', limit = 100) {
    try {
      const response = await this.api.get('/logs', {
        params: { level, limit }
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get logs:', error);
      throw error;
    }
  }

  // Error handling utility
  handleError(error) {
    if (error.response) {
      // Server responded with error status
      const { status, data } = error.response;
      console.error(`HTTP ${status}:`, data);
      return {
        error: true,
        status,
        message: data?.detail || data?.message || 'Unknown error',
        data
      };
    } else if (error.request) {
      // Request made but no response
      console.error('No response received:', error.request);
      return {
        error: true,
        status: 0,
        message: 'No response from server',
        data: null
      };
    } else {
      // Something else happened
      console.error('Request setup error:', error.message);
      return {
        error: true,
        status: 0,
        message: error.message,
        data: null
      };
    }
  }

  // Connection test
  async testConnection() {
    try {
      const response = await this.api.get('/', { timeout: 5000 });
      return {
        connected: true,
        version: response.data?.version,
        status: response.data?.status
      };
    } catch (error) {
      return {
        connected: false,
        error: this.handleError(error)
      };
    }
  }
}

// Create singleton instance
const apiService = new ApiService();

export default apiService; 