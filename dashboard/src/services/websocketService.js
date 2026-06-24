import io from 'socket.io-client';

class WebSocketService {
  constructor() {
    this.socket = null;
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000;
    this.listeners = new Map();
  }

  connect(url = 'http://localhost:8000') {
    try {
      this.socket = io(url, {
        transports: ['websocket', 'polling'],
        timeout: 20000,
        reconnection: true,
        reconnectionAttempts: this.maxReconnectAttempts,
        reconnectionDelay: this.reconnectDelay,
      });

      this.socket.on('connect', () => {
        console.log('🔗 Connected to trading system');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.emit('dashboard_connected', { timestamp: Date.now() });
      });

      this.socket.on('disconnect', (reason) => {
        console.log('❌ Disconnected from trading system:', reason);
        this.isConnected = false;
      });

      this.socket.on('connect_error', (error) => {
        console.error('🔌 Connection error:', error);
        this.isConnected = false;
      });

      this.socket.on('reconnect', (attemptNumber) => {
        console.log(`🔄 Reconnected after ${attemptNumber} attempts`);
        this.isConnected = true;
      });

      this.socket.on('reconnect_failed', () => {
        console.error('❌ Failed to reconnect after maximum attempts');
        this.isConnected = false;
      });

      // Trading system events
      this.socket.on('price_update', this.handlePriceUpdate.bind(this));
      this.socket.on('sentiment_update', this.handleSentimentUpdate.bind(this));
      this.socket.on('signal_generated', this.handleSignalGenerated.bind(this));
      this.socket.on('position_update', this.handlePositionUpdate.bind(this));
      this.socket.on('alert_generated', this.handleAlertGenerated.bind(this));
      this.socket.on('system_status', this.handleSystemStatus.bind(this));
      this.socket.on('model_performance', this.handleModelPerformance.bind(this));

    } catch (error) {
      console.error('Failed to initialize WebSocket connection:', error);
    }
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
      this.isConnected = false;
    }
  }

  emit(event, data) {
    if (this.socket && this.isConnected) {
      this.socket.emit(event, data);
    } else {
      console.warn('Socket not connected, cannot emit:', event);
    }
  }

  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);
  }

  off(event, callback) {
    if (this.listeners.has(event)) {
      const callbacks = this.listeners.get(event);
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  // Event handlers
  handlePriceUpdate(data) {
    this.notifyListeners('price_update', data);
  }

  handleSentimentUpdate(data) {
    this.notifyListeners('sentiment_update', data);
  }

  handleSignalGenerated(data) {
    this.notifyListeners('signal_generated', data);
  }

  handlePositionUpdate(data) {
    this.notifyListeners('position_update', data);
  }

  handleAlertGenerated(data) {
    this.notifyListeners('alert_generated', data);
  }

  handleSystemStatus(data) {
    this.notifyListeners('system_status', data);
  }

  handleModelPerformance(data) {
    this.notifyListeners('model_performance', data);
  }

  notifyListeners(event, data) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in ${event} listener:`, error);
        }
      });
    }
  }

  // Trading system commands
  startTrading() {
    this.emit('start_trading', { timestamp: Date.now() });
  }

  stopTrading() {
    this.emit('stop_trading', { timestamp: Date.now() });
  }

  setAlert(symbol, price, type) {
    this.emit('set_alert', { symbol, price, type, timestamp: Date.now() });
  }

  getSystemStatus() {
    this.emit('get_system_status', { timestamp: Date.now() });
  }

  getModelPerformance() {
    this.emit('get_model_performance', { timestamp: Date.now() });
  }

  // Connection status
  getConnectionStatus() {
    return {
      isConnected: this.isConnected,
      reconnectAttempts: this.reconnectAttempts,
    };
  }
}

// Create singleton instance
const websocketService = new WebSocketService();

export default websocketService; 