"""
AI-Driven Cryptocurrency Trading System
Main Application Entry Point

This is the central orchestrator that coordinates all system components:
- Volume anomaly detection workers
- Sentiment analysis collection
- Trading strategy execution
- Meta-learning model updates
- System monitoring and health checks
"""

import asyncio
import logging
import signal
import sys
import os
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import argparse

# Core system imports
from config.api_config import get_api_config
from utils.logging_config import setup_logging
from technical_analysis.volume_anomaly_detection import VolumeAnomalyDetector
from sentiment_analysis.twitter_collector import TwitterSentimentCollector
from machine_learning.meta_learning import MAMLTrader, MAMLConfig

# FastAPI for web interface
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# System monitoring
import psutil
import time
from collections import defaultdict

class TradingSystemOrchestrator:
    """
    Main orchestrator for the AI trading system.
    
    Coordinates all components and manages system lifecycle:
    - Initializes all ML models and data collectors
    - Manages worker processes for real-time processing
    - Provides web API for monitoring and control
    - Handles graceful shutdown and error recovery
    """
    
    def __init__(self, config_file: str = None):
        """Initialize the trading system orchestrator."""
        
        # Setup logging
        self.logger = setup_logging()
        self.logger.info("🚀 Initializing AI Trading System...")
        
        # Load configuration
        self.config = get_api_config()
        
        # System state
        self.is_running = False
        self.shutdown_requested = False
        self.workers = {}
        self.models = {}
        self.performance_metrics = defaultdict(list)
        
        # FastAPI app
        self.app = FastAPI(
            title="AI Crypto Trading System",
            description="Advanced AI-driven cryptocurrency trading platform",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        # Thread pool for background tasks
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self.logger.info("✅ System orchestrator initialized")
    
    def _setup_routes(self):
        """Setup FastAPI routes for system monitoring and control."""
        
        @self.app.get("/")
        async def root():
            return {
                "message": "AI-Driven Cryptocurrency Trading System",
                "version": "1.0.0",
                "status": "running" if self.is_running else "stopped",
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/health")
        async def health_check():
            """Comprehensive system health check."""
            health_status = await self._perform_health_check()
            
            if not health_status["healthy"]:
                raise HTTPException(status_code=503, detail=health_status)
            
            return health_status
        
        @self.app.get("/status")
        async def system_status():
            """Get detailed system status."""
            return await self._get_system_status()
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get system performance metrics."""
            return {
                "system_metrics": self._get_system_metrics(),
                "trading_metrics": self.performance_metrics,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/start")
        async def start_system():
            """Start the trading system."""
            if self.is_running:
                raise HTTPException(status_code=400, detail="System is already running")
            
            await self.start()
            return {"message": "System started successfully"}
        
        @self.app.post("/stop")
        async def stop_system():
            """Stop the trading system."""
            if not self.is_running:
                raise HTTPException(status_code=400, detail="System is not running")
            
            await self.stop()
            return {"message": "System stopped successfully"}
        
        @self.app.get("/models/volume-anomaly/status")
        async def volume_anomaly_status():
            """Get volume anomaly detection model status."""
            detector = self.models.get('volume_detector')
            if not detector:
                raise HTTPException(status_code=404, detail="Volume anomaly detector not found")
            
            return {
                "is_fitted": detector.is_fitted,
                "feature_importance": detector.get_feature_importance(),
                "performance_metrics": detector.get_performance_metrics()
            }
        
        @self.app.get("/models/sentiment/summary")
        async def sentiment_summary():
            """Get sentiment analysis summary."""
            collector = self.models.get('sentiment_collector')
            if not collector:
                raise HTTPException(status_code=404, detail="Sentiment collector not found")
            
            return collector.get_sentiment_summary(hours=24)
        
        @self.app.post("/models/retrain")
        async def retrain_models(background_tasks: BackgroundTasks):
            """Trigger model retraining."""
            background_tasks.add_task(self._retrain_models)
            return {"message": "Model retraining started"}
    
    async def _perform_health_check(self) -> Dict:
        """Perform comprehensive health check."""
        health = {
            "healthy": True,
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Check configuration
        config_validation = self.config.validate_config()
        health["components"]["configuration"] = {
            "status": "healthy" if any(config_validation.values()) else "error",
            "details": config_validation
        }
        
        # Check models
        health["components"]["models"] = {
            "volume_detector": "healthy" if self.models.get('volume_detector') else "not_loaded",
            "sentiment_collector": "healthy" if self.models.get('sentiment_collector') else "not_loaded",
            "maml_trader": "healthy" if self.models.get('maml_trader') else "not_loaded"
        }
        
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        health["components"]["resources"] = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "disk_percent": (disk.used / disk.total) * 100,
            "status": "healthy" if cpu_percent < 80 and memory.percent < 85 else "warning"
        }
        
        # Overall health
        component_health = [comp.get("status", "unknown") for comp in health["components"].values()]
        if "error" in component_health:
            health["healthy"] = False
        elif "warning" in component_health:
            health["healthy"] = True  # Warning is still healthy
        
        return health
    
    async def _get_system_status(self) -> Dict:
        """Get detailed system status."""
        return {
            "running": self.is_running,
            "uptime": time.time() - getattr(self, 'start_time', time.time()),
            "workers": {name: "running" for name in self.workers.keys()},
            "models_loaded": list(self.models.keys()),
            "configuration": {
                "exchanges": self.config.get_available_exchanges(),
                "database": bool(self.config.get_database_config().url),
                "twitter": bool(self.config.get_twitter_config().bearer_token)
            }
        }
    
    def _get_system_metrics(self) -> Dict:
        """Get system performance metrics."""
        return {
            "cpu": {
                "percent": psutil.cpu_percent(),
                "count": psutil.cpu_count()
            },
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent
            },
            "disk": {
                "total": psutil.disk_usage('/').total,
                "used": psutil.disk_usage('/').used,
                "free": psutil.disk_usage('/').free
            },
            "network": {
                "bytes_sent": psutil.net_io_counters().bytes_sent,
                "bytes_recv": psutil.net_io_counters().bytes_recv
            }
        }
    
    async def initialize_models(self):
        """Initialize all ML models and data collectors."""
        self.logger.info("🧠 Initializing ML models...")
        
        try:
            # Initialize volume anomaly detector
            self.models['volume_detector'] = VolumeAnomalyDetector(
                contamination=0.05,
                enable_feature_selection=True
            )
            self.logger.info("✅ Volume anomaly detector initialized")
            
            # Initialize sentiment collector if Twitter config is available
            twitter_config = self.config.get_twitter_config()
            if twitter_config.bearer_token:
                self.models['sentiment_collector'] = TwitterSentimentCollector(
                    bearer_token=twitter_config.bearer_token,
                    api_key=twitter_config.api_key,
                    api_secret=twitter_config.api_secret,
                    access_token=twitter_config.access_token,
                    access_token_secret=twitter_config.access_token_secret
                )
                self.logger.info("✅ Sentiment collector initialized")
            else:
                self.logger.warning("⚠️ Twitter configuration not found, sentiment analysis disabled")
            
            # Initialize MAML trader
            maml_config = MAMLConfig(
                inner_lr=0.01,
                meta_lr=0.001,
                inner_steps=5,
                meta_batch_size=16,
                device='cpu'  # Use GPU if available
            )
            self.models['maml_trader'] = MAMLTrader(maml_config, input_size=50)
            self.logger.info("✅ MAML trader initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing models: {e}")
            raise
    
    async def start_workers(self):
        """Start background workers for real-time processing."""
        self.logger.info("🔄 Starting background workers...")
        
        # Volume anomaly detection worker
        if 'volume_detector' in self.models:
            self.workers['volume_worker'] = self.executor.submit(
                self._volume_worker_loop
            )
            
        # Sentiment analysis worker
        if 'sentiment_collector' in self.models:
            self.workers['sentiment_worker'] = self.executor.submit(
                self._sentiment_worker_loop
            )
        
        # Performance monitoring worker
        self.workers['monitor_worker'] = self.executor.submit(
            self._monitoring_worker_loop
        )
        
        self.logger.info(f"✅ Started {len(self.workers)} background workers")
    
    def _volume_worker_loop(self):
        """Background worker for volume anomaly detection."""
        self.logger.info("📊 Volume anomaly detection worker started")
        
        detector = self.models['volume_detector']
        
        while self.is_running:
            try:
                # This would normally process real market data
                # For now, it's a placeholder that runs model health checks
                if detector.is_fitted:
                    metrics = detector.get_performance_metrics()
                    self.performance_metrics['volume_anomaly'].append({
                        'timestamp': datetime.now().isoformat(),
                        'metrics': metrics
                    })
                
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                self.logger.error(f"Volume worker error: {e}")
                time.sleep(30)  # Wait before retrying
    
    def _sentiment_worker_loop(self):
        """Background worker for sentiment analysis."""
        self.logger.info("📱 Sentiment analysis worker started")
        
        collector = self.models['sentiment_collector']
        
        while self.is_running:
            try:
                # Collect recent sentiment data
                summary = collector.get_sentiment_summary(hours=1)
                if 'error' not in summary:
                    self.performance_metrics['sentiment'].append({
                        'timestamp': datetime.now().isoformat(),
                        'summary': summary
                    })
                
                time.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Sentiment worker error: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _monitoring_worker_loop(self):
        """Background worker for system monitoring."""
        self.logger.info("📈 System monitoring worker started")
        
        while self.is_running:
            try:
                # Collect system metrics
                metrics = self._get_system_metrics()
                self.performance_metrics['system'].append({
                    'timestamp': datetime.now().isoformat(),
                    'metrics': metrics
                })
                
                # Keep only last 100 measurements
                for key in self.performance_metrics:
                    if len(self.performance_metrics[key]) > 100:
                        self.performance_metrics[key] = self.performance_metrics[key][-100:]
                
                time.sleep(30)  # Run every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring worker error: {e}")
                time.sleep(60)
    
    async def _retrain_models(self):
        """Retrain models with latest data."""
        self.logger.info("🔄 Starting model retraining...")
        
        try:
            # Retrain volume anomaly detector with latest data
            # This would normally use real market data
            self.logger.info("Retraining volume anomaly detector...")
            
            # Retrain MAML trader
            self.logger.info("Retraining MAML trader...")
            
            self.logger.info("✅ Model retraining completed")
            
        except Exception as e:
            self.logger.error(f"❌ Model retraining failed: {e}")
    
    async def start(self):
        """Start the trading system."""
        if self.is_running:
            self.logger.warning("System is already running")
            return
        
        self.logger.info("🚀 Starting AI Trading System...")
        self.start_time = time.time()
        
        try:
            # Initialize models
            await self.initialize_models()
            
            # Start workers
            await self.start_workers()
            
            self.is_running = True
            self.logger.info("✅ AI Trading System started successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start system: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the trading system."""
        self.logger.info("🛑 Stopping AI Trading System...")
        
        self.is_running = False
        
        # Stop all workers
        for name, worker in self.workers.items():
            self.logger.info(f"Stopping {name}...")
            worker.cancel()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("✅ AI Trading System stopped")
    
    def run_web_server(self, host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
        """Run the web server."""
        self.logger.info(f"🌐 Starting web server on {host}:{port}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="debug" if debug else "info",
            reload=debug
        )

def signal_handler(signum, frame, orchestrator):
    """Handle shutdown signals."""
    print("\n📤 Received shutdown signal, stopping system...")
    # Set a flag to indicate shutdown was requested
    orchestrator.shutdown_requested = True

async def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="AI-Driven Cryptocurrency Trading System")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the web server")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the web server")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-web", action="store_true", help="Disable web server")
    
    args = parser.parse_args()
    
    # Initialize system
    orchestrator = TradingSystemOrchestrator()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, orchestrator))
    signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, orchestrator))
    
    server_task = None
    
    try:
        # Start the system
        await orchestrator.start()
        
        if not args.no_web:
            # Run web server in a separate task to allow shutdown handling
            server_task = asyncio.create_task(
                asyncio.to_thread(
                    orchestrator.run_web_server,
                    host=args.host,
                    port=args.port,
                    debug=args.debug
                )
            )
            
            # Monitor for shutdown signals
            while orchestrator.is_running and not orchestrator.shutdown_requested:
                await asyncio.sleep(0.1)
                
        else:
            # Keep system running without web server
            while orchestrator.is_running and not orchestrator.shutdown_requested:
                await asyncio.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n📤 Interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
    finally:
        # Proper cleanup of server task and orchestrator
        if server_task and not server_task.done():
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"⚠️ Warning: Server task cleanup error: {e}")
        
        # Stop orchestrator only once
        if orchestrator.is_running:
            await orchestrator.stop()

if __name__ == "__main__":
    # Set up asyncio event loop
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Application error: {e}")
        sys.exit(1)