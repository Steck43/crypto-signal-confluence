#!/bin/bash
set -e

# AI-Driven Cryptocurrency Trading System
# Docker Container Entrypoint Script

echo "🚀 Starting AI-Driven Cryptocurrency Trading System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_debug() {
    if [ "$DEBUG" = "true" ]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Wait for dependent services
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3
    local timeout=${4:-30}
    
    print_status "Waiting for $service ($host:$port)..."
    
    for i in $(seq 1 $timeout); do
        if timeout 1 bash -c "cat < /dev/null > /dev/tcp/$host/$port" 2>/dev/null; then
            print_status "$service is ready!"
            return 0
        fi
        sleep 1
    done
    
    print_error "$service is not available after $timeout seconds"
    return 1
}

# Database migration function
run_migrations() {
    print_status "Running database migrations..."
    
    if [ -f "/app/alembic/versions" ]; then
        python -m alembic upgrade head
    else
        print_warning "No migrations found, creating tables..."
        python -c "
from src.config.api_config import get_api_config
from sqlalchemy import create_engine
from src.sentiment_analysis.twitter_collector import Base

config = get_api_config()
db_config = config.get_database_config()
engine = create_engine(db_config.url)
Base.metadata.create_all(engine)
print('Database tables created successfully!')
"
    fi
}

# Health check function
health_check() {
    print_status "Performing health checks..."
    
    # Check Python imports
    python -c "
import sys
try:
    from src.technical_analysis.volume_anomaly_detection import VolumeAnomalyDetector
    from src.sentiment_analysis.twitter_collector import TwitterSentimentCollector
    from src.trading_strategies.ensemble_strategy import EnsembleStrategy
    from src.machine_learning.meta_learning import MAMLTrader
    print('✅ All core modules imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"
    
    # Check configuration
    python -c "
import os
from src.config.api_config import get_api_config

config = get_api_config()
validation = config.validate_config()

print('Configuration validation:')
for service, valid in validation.items():
    status = '✅' if valid else '❌'
    print(f'  {status} {service}: {valid}')

# Warn about missing critical configs
if not validation.get('database', False):
    print('⚠️  Database configuration missing')
if not validation.get('twitter', False):
    print('⚠️  Twitter API configuration missing - sentiment analysis will be disabled')
"
}

# Setup logging directory
setup_logging() {
    print_status "Setting up logging..."
    
    mkdir -p /app/logs
    chmod 755 /app/logs
    
    # Rotate old logs if they exist
    if [ -f "/app/logs/trading_app.log" ]; then
        mv /app/logs/trading_app.log "/app/logs/trading_app.log.$(date +%Y%m%d_%H%M%S)"
    fi
    
    print_status "Logging directory ready"
}

# Download required data/models
download_models() {
    print_status "Checking for required models and data..."
    
    # Create models directory
    mkdir -p /app/models
    
    # Download spaCy model if not exists
    if ! python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
        print_status "Downloading spaCy English model..."
        python -m spacy download en_core_web_sm
    fi
    
    # Download NLTK data if not exists
    python -c "
import nltk
import os
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('vader_lexicon')
    print('✅ NLTK data already available')
except LookupError:
    print('📥 Downloading NLTK data...')
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    print('✅ NLTK data downloaded')
"
}

# Initialize system
initialize_system() {
    print_status "Initializing AI Trading System..."
    
    # Set timezone
    if [ ! -z "$TZ" ]; then
        export TZ=$TZ
        print_debug "Timezone set to: $TZ"
    fi
    
    # Set Python path
    export PYTHONPATH=/app/src:$PYTHONPATH
    print_debug "Python path: $PYTHONPATH"
    
    # Create required directories
    mkdir -p /app/data/market_data
    mkdir -p /app/data/sentiment_data
    mkdir -p /app/data/models
    mkdir -p /app/config
    
    # Generate default configuration if not exists
    if [ ! -f "/app/config/api_config.json" ]; then
        print_status "Generating default configuration..."
        python -c "
from src.config.api_config import get_api_config
config = get_api_config()
print('Default configuration generated')
"
    fi
}

# Main execution
main() {
    print_status "🤖 AI Crypto Trading System Entrypoint"
    print_status "Environment: ${ENVIRONMENT:-development}"
    print_status "Debug mode: ${DEBUG:-false}"
    
    # Setup
    setup_logging
    initialize_system
    download_models
    
    # Wait for dependencies based on environment
    if [ "$ENVIRONMENT" = "production" ] || [ "$ENVIRONMENT" = "staging" ]; then
        # In production, wait for external services
        if [ ! -z "$DATABASE_URL" ] && [[ "$DATABASE_URL" == *"postgres"* ]]; then
            DB_HOST=$(echo $DATABASE_URL | sed -E 's/.*@([^:]+):.*/\1/')
            DB_PORT=$(echo $DATABASE_URL | sed -E 's/.*:([0-9]+)\/.*/\1/')
            wait_for_service "$DB_HOST" "$DB_PORT" "PostgreSQL" 60
        fi
        
        if [ ! -z "$REDIS_URL" ]; then
            REDIS_HOST=$(echo $REDIS_URL | sed -E 's/.*@([^:]+):.*/\1/' | sed 's/.*\/\///')
            REDIS_PORT=$(echo $REDIS_URL | sed -E 's/.*:([0-9]+).*/\1/')
            wait_for_service "$REDIS_HOST" "$REDIS_PORT" "Redis" 30
        fi
        
        # Run migrations
        run_migrations
    fi
    
    # Health checks
    health_check
    
    print_status "✅ System initialization complete!"
    
    # Execute the main command
    if [ $# -eq 0 ]; then
        print_status "No command provided, starting default application..."
        exec python -m src.main
    else
        print_status "Executing command: $@"
        exec "$@"
    fi
}

# Error handling
trap 'print_error "Script failed at line $LINENO"' ERR

# Ensure we're in the right directory
cd /app

# Run main function
main "$@"