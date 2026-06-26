"""
GRU Time Series Forecaster for Institutional Trading System

EXPLORATORY: GRU overlay research. Not on the validated ablation path. Not validated.

Provides time series forecasting capabilities to complement the volume anomaly detection.
Focuses on price prediction and trend forecasting using GRU neural networks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

@dataclass
class GRUPrediction:
    price_change_prediction: float      # Predicted price change (%)
    volatility_prediction: float        # Predicted volatility  
    trend_direction: int                # -1 (bearish), 0 (neutral), 1 (bullish)
    confidence: float                   # Prediction confidence
    signal_strength: float              # Signal strength for trading
    reasoning: str                      # Human-readable explanation

class GRUPriceForecaster(nn.Module):
    """
    GRU Neural Network for Price Forecasting
    
    Architecture optimized for financial time series:
    - Multi-layer GRU with dropout for regularization
    - Fully connected layers for final prediction
    - Multiple outputs: price change, volatility, trend
    """
    
    def __init__(self, 
                 input_size: int = 10, 
                 hidden_size: int = 128, 
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 output_size: int = 3):
        """
        Initialize GRU architecture
        
        Args:
            input_size: Number of input features per timestep
            hidden_size: Hidden state size (128 for good capacity)
            num_layers: Number of GRU layers (3 for depth)
            dropout: Dropout rate for regularization
            output_size: Number of outputs (3: price_change, volatility, trend)
        """
        super(GRUPriceForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        
        # GRU layers with dropout
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # Unidirectional for causal prediction
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, output_size)
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size // 2)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 4)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        if x.is_cuda:
            h0 = h0.cuda()
        
        # GRU forward pass
        gru_out, _ = self.gru(x, h0)
        
        # Use the last time step output
        out = gru_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Fully connected layers with normalization and dropout
        out = self.fc1(out)
        out = self.batch_norm1(out) if out.size(0) > 1 else out  # Skip batch norm for single samples
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.batch_norm2(out) if out.size(0) > 1 else out
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        
        return out

class GRUTradingPredictor:
    """
    GRU-based trading predictor integrated with institutional system
    
    Features:
    - Time series forecasting for price movements
    - Integration with 66-feature volume system
    - Professional training with validation
    - Real-time prediction capabilities
    """
    
    def __init__(self, 
                 sequence_length: int = 30,
                 hidden_size: int = 128,
                 num_layers: int = 3,
                 learning_rate: float = 0.001,
                 device: str = 'cpu'):
        """
        Initialize GRU trading predictor
        
        Args:
            sequence_length: Length of input sequences (30 = 2.5 hours with 5-min data)
            hidden_size: GRU hidden state size
            num_layers: Number of GRU layers
            learning_rate: Learning rate for training
            device: Device for computation ('cpu' or 'cuda')
        """
        
        self.sequence_length = sequence_length
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        
        # Model will be initialized after seeing input size
        self.model = None
        self.input_size = None
        
        # Data preprocessing
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
        
        # Training state
        self.is_fitted = False
        self.training_history = []
        self.feature_names = []
        
        # Model parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("🧠 GRU Trading Predictor initialized")
    
    def prepare_sequences(self, 
                         market_data: pd.DataFrame, 
                         volume_features: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare sequential data for GRU training
        
        Combines market data with selected volume features for optimal prediction
        
        Args:
            market_data: OHLCV market data
            volume_features: Volume anomaly features from institutional detector
            
        Returns:
            Tuple of (features_tensor, targets_tensor)
        """
        
        # Select key features for time series prediction
        market_features = market_data[['close', 'volume', 'high', 'low']].copy()
        
        # Add derived market features
        market_features['returns'] = market_features['close'].pct_change()
        market_features['log_volume'] = np.log1p(market_features['volume'])
        market_features['hl_ratio'] = (market_features['high'] - market_features['low']) / market_features['close']
        market_features['price_position'] = (market_features['close'] - market_features['low']) / (market_features['high'] - market_features['low'])
        
        # Select top volume features (based on typical importance)
        top_volume_features = [
            'volume_zscore', 'volume_entropy', 'mutual_info', 
            'volume_price_correlation', 'price_volatility',
            'volume_momentum', 'price_momentum', 'vwap',
            'volume_normalized', 'volume_percentile'
        ]
        
        # Filter available features
        available_volume_features = [f for f in top_volume_features if f in volume_features.columns]
        
        if available_volume_features:
            selected_volume_features = volume_features[available_volume_features]
        else:
            # Fallback to first 10 volume features if specific ones not found
            selected_volume_features = volume_features.iloc[:, :min(10, volume_features.shape[1])]
        
        # Combine all features
        combined_features = pd.concat([market_features, selected_volume_features], axis=1)
        
        # Forward fill missing values
        combined_features = combined_features.fillna(method='ffill').fillna(0)
        
        # Store feature names
        self.feature_names = combined_features.columns.tolist()
        self.input_size = len(self.feature_names)
        
        self.logger.info(f"📊 Prepared {self.input_size} features for GRU: {self.feature_names[:5]}...")
        
        # Scale features
        scaled_features = self.scaler.fit_transform(combined_features.values)
        
        # Create sequences and targets
        X, y = [], []
        
        for i in range(self.sequence_length, len(scaled_features) - 5):  # Leave 5 for target calculation
            # Input sequence
            X.append(scaled_features[i-self.sequence_length:i])
            
            # Targets: next period price change, volatility, trend
            current_price = market_data['close'].iloc[i]
            future_price = market_data['close'].iloc[i+5]  # 5 periods ahead
            
            # Price change (percentage)
            price_change = (future_price - current_price) / current_price
            
            # Volatility (rolling std of returns)
            volatility = market_data['close'].pct_change().rolling(window=10).std().iloc[i]
            if pd.isna(volatility):
                volatility = 0.02  # Default volatility
            
            # Trend direction
            if price_change > 0.01:       # > 1% gain
                trend = 1
            elif price_change < -0.01:    # < -1% loss
                trend = -1
            else:                         # Between -1% and 1%
                trend = 0
            
            y.append([price_change, volatility, trend])
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(np.array(X))
        y_tensor = torch.FloatTensor(np.array(y))
        
        # Scale targets
        y_scaled = self.target_scaler.fit_transform(y_tensor.numpy())
        y_tensor = torch.FloatTensor(y_scaled)
        
        self.logger.info(f"✅ Created {len(X)} sequences of length {self.sequence_length}")
        
        return X_tensor.to(self.device), y_tensor.to(self.device)
    
    def fit(self, 
            market_data: pd.DataFrame, 
            volume_features: pd.DataFrame,
            epochs: int = 200,
            batch_size: int = 32,
            validation_split: float = 0.2,
            patience: int = 20,
            min_delta: float = 1e-4) -> 'GRUTradingPredictor':
        """
        Train GRU model with professional validation and early stopping
        
        Args:
            market_data: Market OHLCV data
            volume_features: Volume anomaly features
            epochs: Maximum training epochs
            batch_size: Training batch size
            validation_split: Fraction of data for validation
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
        """
        
        # Prepare data
        X, y = self.prepare_sequences(market_data, volume_features)
        
        # Initialize model now that we know input size
        self.model = GRUPriceForecaster(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(self.device)
        
        # Split data for validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        # Training variables
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.logger.info(f"🚀 Starting GRU training: {epochs} epochs, batch size {batch_size}")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_gru_model.pth')
            else:
                patience_counter += 1
            
            # Logging
            if epoch % 20 == 0 or epoch == epochs - 1:
                avg_train_loss = train_loss / len(train_loader)
                self.logger.info(f"   Epoch {epoch+1:3d}: Train Loss={avg_train_loss:.6f}, "
                               f"Val Loss={val_loss:.6f}, LR={optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping
            if patience_counter >= patience:
                self.logger.info(f"🛑 Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_gru_model.pth'))
        
        # Calculate final metrics
        self.model.eval()
        with torch.no_grad():
            final_predictions = self.model(X_val)
            final_val_loss = criterion(final_predictions, y_val).item()
            
            # Unscale predictions for analysis
            pred_unscaled = self.target_scaler.inverse_transform(final_predictions.cpu().numpy())
            y_unscaled = self.target_scaler.inverse_transform(y_val.cpu().numpy())
            
            # Calculate metrics
            mse = mean_squared_error(y_unscaled, pred_unscaled)
            mae = mean_absolute_error(y_unscaled, pred_unscaled)
        
        self.is_fitted = True
        
        self.logger.info("✅ GRU training completed:")
        self.logger.info(f"   📊 Final Validation Loss: {final_val_loss:.6f}")
        self.logger.info(f"   📊 MSE: {mse:.6f}")
        self.logger.info(f"   📊 MAE: {mae:.6f}")
        
        return self
    
    def predict(self, recent_data: pd.DataFrame, recent_volume_features: pd.DataFrame) -> GRUPrediction:
        """
        Generate prediction for next periods
        
        Args:
            recent_data: Recent market data (at least sequence_length periods)
            recent_volume_features: Recent volume features
            
        Returns:
            GRUPrediction object with forecasts and trading signals
        """
        
        if not self.is_fitted:
            raise ValueError("❌ GRU model must be fitted before prediction")
        
        try:
            # Prepare input features (same as training)
            market_features = recent_data[['close', 'volume', 'high', 'low']].copy()
            market_features['returns'] = market_features['close'].pct_change()
            market_features['log_volume'] = np.log1p(market_features['volume'])
            market_features['hl_ratio'] = (market_features['high'] - market_features['low']) / market_features['close']
            market_features['price_position'] = (market_features['close'] - market_features['low']) / (market_features['high'] - market_features['low'])
            
            # Select same volume features as training
            available_volume_features = [f for f in self.feature_names if f in recent_volume_features.columns]
            
            if available_volume_features:
                selected_volume_features = recent_volume_features[available_volume_features]
            else:
                # Create zero features if not available
                selected_volume_features = pd.DataFrame(
                    0, index=recent_data.index, 
                    columns=[f for f in self.feature_names if f not in market_features.columns]
                )
            
            # Combine features
            combined_features = pd.concat([market_features, selected_volume_features], axis=1)
            combined_features = combined_features.fillna(method='ffill').fillna(0)
            
            # Ensure feature order matches training
            combined_features = combined_features.reindex(columns=self.feature_names, fill_value=0)
            
            # Get last sequence
            if len(combined_features) < self.sequence_length:
                raise ValueError(f"Need at least {self.sequence_length} data points for prediction")
            
            last_sequence = combined_features.tail(self.sequence_length).values
            
            # Scale input
            scaled_sequence = self.scaler.transform(last_sequence)
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(scaled_sequence).unsqueeze(0).to(self.device)  # Add batch dimension
            
            # Generate prediction
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(input_tensor).cpu().numpy()[0]
            
            # Unscale prediction
            prediction_unscaled = self.target_scaler.inverse_transform(prediction.reshape(1, -1))[0]
            
            # Extract predictions
            price_change_pred = float(prediction_unscaled[0])
            volatility_pred = float(abs(prediction_unscaled[1]))  # Ensure positive
            trend_pred = int(np.round(prediction_unscaled[2]))
            
            # Calculate confidence based on prediction strength
            signal_strength = abs(price_change_pred)
            confidence = min(signal_strength * 10, 1.0)  # Scale to [0, 1]
            
            # Generate reasoning
            reasoning = self._generate_gru_reasoning(price_change_pred, volatility_pred, trend_pred, confidence)
            
            return GRUPrediction(
                price_change_prediction=price_change_pred,
                volatility_prediction=volatility_pred,
                trend_direction=trend_pred,
                confidence=confidence,
                signal_strength=signal_strength,
                reasoning=reasoning
            )
            
        except Exception as e:
            self.logger.error(f"❌ GRU prediction error: {e}")
            
            # Return neutral prediction on error
            return GRUPrediction(
                price_change_prediction=0.0,
                volatility_prediction=0.02,
                trend_direction=0,
                confidence=0.0,
                signal_strength=0.0,
                reasoning=f"GRU prediction failed: {str(e)}"
            )
    
    def _generate_gru_reasoning(self, 
                               price_change: float, 
                               volatility: float, 
                               trend: int, 
                               confidence: float) -> str:
        """Generate human-readable reasoning for GRU prediction"""
        
        direction = "bullish" if price_change > 0 else "bearish" if price_change < 0 else "neutral"
        vol_desc = "high" if volatility > 0.03 else "moderate" if volatility > 0.015 else "low"
        
        reasoning = f"GRU forecasts {price_change*100:.1f}% price change ({direction} trend) "
        reasoning += f"with {vol_desc} volatility ({volatility*100:.1f}%) "
        reasoning += f"and {confidence:.1%} confidence"
        
        return reasoning
    
    def get_model_summary(self) -> Dict:
        """Get comprehensive model information"""
        
        if not self.is_fitted:
            return {"error": "Model not fitted"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_architecture": {
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "sequence_length": self.sequence_length
            },
            "parameters": {
                "total": total_params,
                "trainable": trainable_params
            },
            "features": {
                "count": len(self.feature_names),
                "names": self.feature_names
            },
            "training_info": {
                "device": str(self.device),
                "learning_rate": self.learning_rate,
                "is_fitted": self.is_fitted
            }
        }
    
    def save_model(self, filepath: str):
        """Save complete GRU model"""
        if self.is_fitted:
            save_dict = {
                'model_state_dict': self.model.state_dict(),
                'model_params': {
                    'input_size': self.input_size,
                    'hidden_size': self.hidden_size,
                    'num_layers': self.num_layers
                },
                'scaler': self.scaler,
                'target_scaler': self.target_scaler,
                'feature_names': self.feature_names,
                'sequence_length': self.sequence_length
            }
            torch.save(save_dict, filepath)
            self.logger.info(f"💾 GRU model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load complete GRU model"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Restore model parameters
            self.input_size = checkpoint['model_params']['input_size']
            self.hidden_size = checkpoint['model_params']['hidden_size']
            self.num_layers = checkpoint['model_params']['num_layers']
            
            # Recreate model
            self.model = GRUPriceForecaster(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers
            ).to(self.device)
            
            # Load state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.scaler = checkpoint['scaler']
            self.target_scaler = checkpoint['target_scaler']
            self.feature_names = checkpoint['feature_names']
            self.sequence_length = checkpoint['sequence_length']
            
            self.is_fitted = True
            self.logger.info(f"📂 GRU model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load GRU model: {e}") 