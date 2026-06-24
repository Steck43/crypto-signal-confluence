"""
Institutional-Grade Performance Monitoring System
Monitors both system performance (CPU, memory, latency) and trading performance
(returns, risk metrics, signal quality) in real-time for production trading systems.
"""

import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import deque
import pandas as pd
import numpy as np
from contextlib import contextmanager

# Setup logger
logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available, system monitoring will be limited")


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    disk_io_read: float
    disk_io_write: float
    network_bytes_sent: float
    network_bytes_recv: float
    active_threads: int
    open_files: int


@dataclass
class TradingMetrics:
    """Trading performance metrics"""
    timestamp: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    unrealized_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    current_positions: int
    portfolio_value: float


@dataclass
class SignalMetrics:
    """Signal generation and quality metrics"""
    timestamp: datetime
    signals_generated: int
    signal_latency_ms: float
    signal_accuracy: float
    false_positive_rate: float
    signal_strength_avg: float
    processing_time_ms: float


class PerformanceMonitor:
    """
    Comprehensive performance monitoring for institutional trading systems
    """
    
    def __init__(self, monitoring_interval: int = 5, history_size: int = 1440):
        """
        Initialize performance monitor
        
        Args:
            monitoring_interval: Seconds between measurements
            history_size: Number of historical measurements to keep (1440 = 24h at 1min intervals)
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        
        # Performance history
        self.system_history = deque(maxlen=history_size)
        self.trading_history = deque(maxlen=history_size)
        self.signal_history = deque(maxlen=history_size)
        
        # Current metrics
        self.current_system_metrics: Optional[SystemMetrics] = None
        self.current_trading_metrics: Optional[TradingMetrics] = None
        self.current_signal_metrics: Optional[SignalMetrics] = None
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Alert thresholds
        self.alert_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'signal_latency_ms': 500.0,
            'max_drawdown': -0.15,  # 15% max drawdown
            'win_rate_min': 0.45    # Minimum 45% win rate
        }
        
        # Performance tracking
        self.start_time = datetime.now()
        self.trade_log: List[Dict] = []
        self.signal_log: List[Dict] = []
        
        # Baseline measurements
        self._baseline_metrics: Optional[SystemMetrics] = None
        
    def start_monitoring(self):
        """Start continuous performance monitoring"""
        if self._monitoring:
            logger.warning("Performance monitoring already running")
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        # Take baseline measurement
        self._baseline_metrics = self._collect_system_metrics()
        
        logger.info("🔍 Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("🛑 Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.current_system_metrics = system_metrics
                self.system_history.append(system_metrics)
                
                # Check for alerts
                self._check_alerts(system_metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics"""
        if not PSUTIL_AVAILABLE:
            # Return dummy metrics if psutil not available
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_gb=0.0,
                disk_io_read=0.0,
                disk_io_write=0.0,
                network_bytes_sent=0.0,
                network_bytes_recv=0.0,
                active_threads=threading.active_count(),
                open_files=0
            )
        
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_read = disk_io.read_bytes if disk_io else 0
        disk_write = disk_io.write_bytes if disk_io else 0
        
        # Network I/O
        network_io = psutil.net_io_counters()
        network_sent = network_io.bytes_sent if network_io else 0
        network_recv = network_io.bytes_recv if network_io else 0
        
        # Process info
        current_process = psutil.Process()
        active_threads = current_process.num_threads()
        open_files = len(current_process.open_files())
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            disk_io_read=disk_read,
            disk_io_write=disk_write,
            network_bytes_sent=network_sent,
            network_bytes_recv=network_recv,
            active_threads=active_threads,
            open_files=open_files
        )
    
    def update_trading_metrics(self, total_trades: int, winning_trades: int, 
                             total_pnl: float, unrealized_pnl: float,
                             max_drawdown: float, current_positions: int,
                             portfolio_value: float):
        """Update trading performance metrics"""
        
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Calculate Sharpe ratio from trade history
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        trading_metrics = TradingMetrics(
            timestamp=datetime.now(),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            unrealized_pnl=unrealized_pnl,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            current_positions=current_positions,
            portfolio_value=portfolio_value
        )
        
        self.current_trading_metrics = trading_metrics
        self.trading_history.append(trading_metrics)
        
        # Check trading alerts
        self._check_trading_alerts(trading_metrics)
    
    def update_signal_metrics(self, signals_generated: int, signal_latency_ms: float,
                            signal_accuracy: float, false_positive_rate: float,
                            signal_strength_avg: float, processing_time_ms: float):
        """Update signal generation metrics"""
        
        signal_metrics = SignalMetrics(
            timestamp=datetime.now(),
            signals_generated=signals_generated,
            signal_latency_ms=signal_latency_ms,
            signal_accuracy=signal_accuracy,
            false_positive_rate=false_positive_rate,
            signal_strength_avg=signal_strength_avg,
            processing_time_ms=processing_time_ms
        )
        
        self.current_signal_metrics = signal_metrics
        self.signal_history.append(signal_metrics)
        
        # Check signal alerts
        self._check_signal_alerts(signal_metrics)
    
    def log_trade(self, trade_type: str, symbol: str, entry_price: float,
                  exit_price: Optional[float], quantity: float, pnl: Optional[float],
                  signal_strength: float, execution_time_ms: float):
        """Log individual trade for performance analysis"""
        
        trade_record = {
            'timestamp': datetime.now(),
            'trade_type': trade_type,
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'pnl': pnl,
            'signal_strength': signal_strength,
            'execution_time_ms': execution_time_ms
        }
        
        self.trade_log.append(trade_record)
        
        # Keep only recent trades (last 1000)
        if len(self.trade_log) > 1000:
            self.trade_log = self.trade_log[-1000:]
    
    def log_signal(self, signal_type: str, symbol: str, strength: float,
                   processing_time_ms: float, confidence: float):
        """Log signal generation for analysis"""
        
        signal_record = {
            'timestamp': datetime.now(),
            'signal_type': signal_type,
            'symbol': symbol,
            'strength': strength,
            'processing_time_ms': processing_time_ms,
            'confidence': confidence
        }
        
        self.signal_log.append(signal_record)
        
        # Keep only recent signals (last 1000)
        if len(self.signal_log) > 1000:
            self.signal_log = self.signal_log[-1000:]
    
    @contextmanager
    def measure_execution_time(self, operation_name: str):
        """Context manager to measure execution time"""
        start_time = time.time()
        try:
            yield
        finally:
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            logger.debug(f"⏱️ {operation_name}: {execution_time:.2f}ms")
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from recent trades"""
        if len(self.trade_log) < 10:
            return 0.0
        
        recent_trades = self.trade_log[-100:]  # Last 100 trades
        returns = [trade['pnl'] for trade in recent_trades if trade['pnl'] is not None]
        
        if len(returns) < 5:
            return 0.0
        
        returns_series = pd.Series(returns)
        return returns_series.mean() / returns_series.std() if returns_series.std() > 0 else 0.0
    
    def _check_alerts(self, metrics: SystemMetrics):
        """Check system performance alerts"""
        alerts = []
        
        if metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            alerts.append(f"🚨 High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.alert_thresholds['memory_percent']:
            alerts.append(f"🚨 High memory usage: {metrics.memory_percent:.1f}%")
        
        for alert in alerts:
            logger.warning(alert)
    
    def _check_trading_alerts(self, metrics: TradingMetrics):
        """Check trading performance alerts"""
        alerts = []
        
        if metrics.max_drawdown < self.alert_thresholds['max_drawdown']:
            alerts.append(f"🚨 High drawdown: {metrics.max_drawdown:.2%}")
        
        if metrics.win_rate < self.alert_thresholds['win_rate_min'] and metrics.total_trades > 20:
            alerts.append(f"🚨 Low win rate: {metrics.win_rate:.2%}")
        
        for alert in alerts:
            logger.warning(alert)
    
    def _check_signal_alerts(self, metrics: SignalMetrics):
        """Check signal performance alerts"""
        alerts = []
        
        if metrics.signal_latency_ms > self.alert_thresholds['signal_latency_ms']:
            alerts.append(f"🚨 High signal latency: {metrics.signal_latency_ms:.1f}ms")
        
        for alert in alerts:
            logger.warning(alert)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        uptime = datetime.now() - self.start_time
        
        summary = {
            'uptime_hours': uptime.total_seconds() / 3600,
            'system_metrics': self.current_system_metrics.__dict__ if self.current_system_metrics else None,
            'trading_metrics': self.current_trading_metrics.__dict__ if self.current_trading_metrics else None,
            'signal_metrics': self.current_signal_metrics.__dict__ if self.current_signal_metrics else None,
            'total_trades_logged': len(self.trade_log),
            'total_signals_logged': len(self.signal_log)
        }
        
        # Add system performance statistics
        if self.system_history:
            cpu_values = [m.cpu_percent for m in self.system_history]
            memory_values = [m.memory_percent for m in self.system_history]
            
            summary['system_stats'] = {
                'avg_cpu_percent': np.mean(cpu_values),
                'max_cpu_percent': np.max(cpu_values),
                'avg_memory_percent': np.mean(memory_values),
                'max_memory_percent': np.max(memory_values)
            }
        
        # Add trading performance statistics
        if self.trading_history:
            win_rates = [m.win_rate for m in self.trading_history if m.total_trades > 0]
            pnls = [m.total_pnl for m in self.trading_history]
            
            if win_rates and pnls:
                summary['trading_stats'] = {
                    'avg_win_rate': np.mean(win_rates),
                    'current_pnl': pnls[-1] if pnls else 0.0,
                    'pnl_trend': 'positive' if len(pnls) > 1 and pnls[-1] > pnls[0] else 'negative'
                }
        
        return summary
    
    def export_metrics_to_csv(self, filepath: str):
        """Export performance metrics to CSV for analysis"""
        try:
            # System metrics
            if self.system_history:
                system_df = pd.DataFrame([m.__dict__ for m in self.system_history])
                system_df.to_csv(f"{filepath}_system_metrics.csv", index=False)
            
            # Trading metrics
            if self.trading_history:
                trading_df = pd.DataFrame([m.__dict__ for m in self.trading_history])
                trading_df.to_csv(f"{filepath}_trading_metrics.csv", index=False)
            
            # Signal metrics
            if self.signal_history:
                signal_df = pd.DataFrame([m.__dict__ for m in self.signal_history])
                signal_df.to_csv(f"{filepath}_signal_metrics.csv", index=False)
            
            # Trade log
            if self.trade_log:
                trades_df = pd.DataFrame(self.trade_log)
                trades_df.to_csv(f"{filepath}_trades.csv", index=False)
            
            logger.info(f"📊 Performance metrics exported to {filepath}_*.csv")
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
    
    def print_live_dashboard(self):
        """Print live performance dashboard to console"""
        if not self.current_system_metrics:
            print("⏳ Collecting performance data...")
            return
        
        # Clear screen (works on most terminals)
        print("\033[2J\033[H")
        
        print("=" * 80)
        print(f"🚀 INSTITUTIONAL TRADING SYSTEM DASHBOARD - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # System Performance
        sys_metrics = self.current_system_metrics
        print(f"\n📊 SYSTEM PERFORMANCE:")
        print(f"   CPU Usage:      {sys_metrics.cpu_percent:6.1f}%")
        print(f"   Memory Usage:   {sys_metrics.memory_percent:6.1f}% ({sys_metrics.memory_used_gb:.1f} GB)")
        print(f"   Active Threads: {sys_metrics.active_threads:6d}")
        print(f"   Open Files:     {sys_metrics.open_files:6d}")
        
        # Trading Performance
        if self.current_trading_metrics:
            trade_metrics = self.current_trading_metrics
            print(f"\n💰 TRADING PERFORMANCE:")
            print(f"   Total Trades:   {trade_metrics.total_trades:6d}")
            print(f"   Win Rate:       {trade_metrics.win_rate:6.1%}")
            print(f"   Total P&L:      ${trade_metrics.total_pnl:8.2f}")
            print(f"   Max Drawdown:   {trade_metrics.max_drawdown:6.2%}")
            print(f"   Sharpe Ratio:   {trade_metrics.sharpe_ratio:6.2f}")
            print(f"   Positions:      {trade_metrics.current_positions:6d}")
        
        # Signal Performance
        if self.current_signal_metrics:
            sig_metrics = self.current_signal_metrics
            print(f"\n🎯 SIGNAL PERFORMANCE:")
            print(f"   Signals Generated: {sig_metrics.signals_generated:6d}")
            print(f"   Signal Latency:    {sig_metrics.signal_latency_ms:6.1f}ms")
            print(f"   Signal Accuracy:   {sig_metrics.signal_accuracy:6.1%}")
            print(f"   False Positive:    {sig_metrics.false_positive_rate:6.1%}")
            print(f"   Avg Strength:      {sig_metrics.signal_strength_avg:6.2f}")
        
        # Uptime
        uptime = datetime.now() - self.start_time
        print(f"\n⏰ UPTIME: {uptime}")
        
        print("=" * 80)


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor

def start_monitoring():
    """Start global performance monitoring"""
    monitor = get_performance_monitor()
    monitor.start_monitoring()

def stop_monitoring():
    """Stop global performance monitoring"""
    monitor = get_performance_monitor()
    monitor.stop_monitoring()

# Convenience functions
def log_trade(trade_type: str, symbol: str, entry_price: float, exit_price: Optional[float],
              quantity: float, pnl: Optional[float], signal_strength: float, execution_time_ms: float):
    """Log trade to global monitor"""
    monitor = get_performance_monitor()
    monitor.log_trade(trade_type, symbol, entry_price, exit_price, quantity, pnl, signal_strength, execution_time_ms)

def log_signal(signal_type: str, symbol: str, strength: float, processing_time_ms: float, confidence: float):
    """Log signal to global monitor"""
    monitor = get_performance_monitor()
    monitor.log_signal(signal_type, symbol, strength, processing_time_ms, confidence)

def measure_execution_time(operation_name: str):
    """Measure execution time context manager"""
    monitor = get_performance_monitor()
    return monitor.measure_execution_time(operation_name) 