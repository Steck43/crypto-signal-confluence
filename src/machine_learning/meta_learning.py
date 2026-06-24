"""
Model-Agnostic Meta-Learning (MAML) for Trading

Implementation of MAML for rapid adaptation to new market conditions:
- Fast adaptation to regime changes
- Cross-asset learning transfer
- Online meta-learning updates
- Gradient-based optimization

Mathematical Foundation:
MAML objective: min Σ L(θ - α∇θL(θ,D^tr), D^val)
where θ are meta-parameters, α is learning rate, D^tr is training data, D^val is validation data

References:
- Finn et al. (2017): "Model-Agnostic Meta-Learning for Fast Adaptation"
- Nichol et al. (2018): "On First-Order Meta-Learning Algorithms"
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict
import logging
from copy import deepcopy
import warnings

warnings.filterwarnings('ignore')

@dataclass
class Task:
    """A meta-learning task with support/query sets."""
    support_X: torch.Tensor
    support_y: torch.Tensor
    query_X: torch.Tensor
    query_y: torch.Tensor
    metadata: Dict = None

@dataclass
class MAMLConfig:
    """MAML configuration parameters."""
    inner_lr: float = 0.01
    meta_lr: float = 0.001
    inner_steps: int = 5
    meta_batch_size: int = 16
    max_epochs: int = 1000
    early_stopping_patience: int = 50
    adaptation_steps: int = 5
    device: str = 'cpu'

class TradingNetwork(nn.Module):
    """
    Neural network for trading signal prediction.
    
    Architecture designed for financial time series:
    - Input features (technical indicators, volume, sentiment)
    - LSTM layers for temporal modeling
    - Attention mechanism for feature importance
    - Output: trading signal confidence
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 lstm_layers: int = 2, dropout: float = 0.1):
        """
        Initialize trading network.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            lstm_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(TradingNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Feature extraction layers
        self.feature_norm = nn.BatchNorm1d(input_size)
        self.feature_proj = nn.Linear(input_size, hidden_size)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 3)  # [bearish, neutral, bullish]
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor [batch_size, sequence_length, input_size]
            
        Returns:
            Trading signal predictions [batch_size, 3]
        """
        batch_size, seq_len, input_size = x.shape
        
        # Feature normalization and projection
        x_reshaped = x.view(-1, input_size)
        x_norm = self.feature_norm(x_reshaped)
        x_proj = self.feature_proj(x_norm)
        x_proj = x_proj.view(batch_size, seq_len, self.hidden_size)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x_proj)
        
        # Attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last timestep for prediction
        final_hidden = attn_out[:, -1, :]
        
        # Output projection
        output = self.dropout(final_hidden)
        output = self.output_proj(output)
        
        return output

class MAMLTrader:
    """
    Model-Agnostic Meta-Learning for Trading Strategy Adaptation.
    
    Implements MAML to quickly adapt trading strategies to new market conditions:
    1. Learn meta-parameters across multiple market regimes
    2. Fast adaptation to new market conditions with few gradient steps
    3. Continuous online learning from market feedback
    
    Mathematical Framework:
    - Meta-objective: min_θ Σ_i L(θ - α∇_θL(θ,D_i^tr), D_i^val)
    - Inner loop: θ' = θ - α∇_θL(θ,D^support)
    - Outer loop: θ ← θ - β∇_θΣ_tasks L(θ',D^query)
    """
    
    def __init__(self, config: MAMLConfig, input_size: int):
        """
        Initialize MAML trader.
        
        Args:
            config: MAML configuration
            input_size: Number of input features
        """
        self.config = config
        self.input_size = input_size
        
        # Initialize network
        self.network = TradingNetwork(input_size).to(config.device)
        self.meta_optimizer = optim.Adam(self.network.parameters(), lr=config.meta_lr)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.meta_losses = []
        self.adaptation_losses = []
        self.validation_accuracies = []
        
        # Task generation
        self.task_generator = None
        
        # Performance tracking
        self.best_meta_loss = float('inf')
        self.patience_counter = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized MAML trader with {sum(p.numel() for p in self.network.parameters())} parameters")
    
    def create_task_from_data(self, data: pd.DataFrame, 
                            target_col: str = 'target',
                            sequence_length: int = 20,
                            support_ratio: float = 0.7) -> Task:
        """
        Create a meta-learning task from market data.
        
        Args:
            data: Market data with features and targets
            target_col: Name of target column
            sequence_length: Length of input sequences
            support_ratio: Ratio of data for support set
            
        Returns:
            Meta-learning task
        """
        # Convert to sequences
        X, y = self._create_sequences(data.drop(columns=[target_col]), 
                                    data[target_col], sequence_length)
        
        # Split into support and query sets
        n_support = int(len(X) * support_ratio)
        
        support_X = torch.FloatTensor(X[:n_support]).to(self.config.device)
        support_y = torch.LongTensor(y[:n_support]).to(self.config.device)
        query_X = torch.FloatTensor(X[n_support:]).to(self.config.device)
        query_y = torch.LongTensor(y[n_support:]).to(self.config.device)
        
        return Task(support_X, support_y, query_X, query_y)
    
    def _create_sequences(self, features: pd.DataFrame, targets: pd.Series, 
                         sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences from time series data."""
        X, y = [], []
        
        for i in range(sequence_length, len(features)):
            X.append(features.iloc[i-sequence_length:i].values)
            y.append(targets.iloc[i])
        
        return np.array(X), np.array(y)
    
    def inner_loop_update(self, task: Task) -> nn.Module:
        """
        Perform inner loop update for task adaptation.
        
        Args:
            task: Meta-learning task
            
        Returns:
            Adapted model parameters
        """
        # Create a copy of the network for adaptation
        adapted_network = deepcopy(self.network)
        inner_optimizer = optim.SGD(adapted_network.parameters(), lr=self.config.inner_lr)
        
        # Adaptation steps on support set
        for step in range(self.config.inner_steps):
            inner_optimizer.zero_grad()
            
            # Forward pass on support set
            support_pred = adapted_network(task.support_X)
            support_loss = self.criterion(support_pred, task.support_y)
            
            # Backward pass
            support_loss.backward()
            inner_optimizer.step()
        
        return adapted_network
    
    def meta_update(self, tasks: List[Task]) -> float:
        """
        Perform meta-update across multiple tasks.
        
        Args:
            tasks: List of meta-learning tasks
            
        Returns:
            Meta-loss across all tasks
        """
        self.meta_optimizer.zero_grad()
        
        meta_loss = 0.0
        
        for task in tasks:
            # Inner loop adaptation
            adapted_network = self.inner_loop_update(task)
            
            # Evaluate on query set
            query_pred = adapted_network(task.query_X)
            query_loss = self.criterion(query_pred, task.query_y)
            
            meta_loss += query_loss
        
        # Average meta-loss
        meta_loss = meta_loss / len(tasks)
        
        # Meta-gradient update
        meta_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def train_meta_learning(self, task_generator: Callable, validation_tasks: List[Task] = None):
        """
        Train the meta-learning model.
        
        Args:
            task_generator: Function that generates training tasks
            validation_tasks: Tasks for validation
        """
        self.logger.info("Starting meta-learning training...")
        
        for epoch in range(self.config.max_epochs):
            # Generate batch of tasks
            tasks = [task_generator() for _ in range(self.config.meta_batch_size)]
            
            # Meta-update
            meta_loss = self.meta_update(tasks)
            self.meta_losses.append(meta_loss)
            
            # Validation
            if validation_tasks and epoch % 10 == 0:
                val_accuracy = self.evaluate_adaptation(validation_tasks)
                self.validation_accuracies.append(val_accuracy)
                
                self.logger.info(f"Epoch {epoch}: Meta-loss={meta_loss:.4f}, Val-accuracy={val_accuracy:.4f}")
                
                # Early stopping
                if meta_loss < self.best_meta_loss:
                    self.best_meta_loss = meta_loss
                    self.patience_counter = 0
                    self.save_model(f"best_maml_model_epoch_{epoch}.pt")
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                if epoch % 100 == 0:
                    self.logger.info(f"Epoch {epoch}: Meta-loss={meta_loss:.4f}")
        
        self.logger.info("Meta-learning training completed!")
    
    def adapt_to_task(self, task: Task) -> nn.Module:
        """
        Quickly adapt model to a new task.
        
        Args:
            task: New market condition task
            
        Returns:
            Adapted model
        """
        return self.inner_loop_update(task)
    
    def predict_with_adaptation(self, support_data: pd.DataFrame, 
                              query_data: pd.DataFrame,
                              target_col: str = 'target',
                              sequence_length: int = 20) -> np.ndarray:
        """
        Make predictions after adapting to support data.
        
        Args:
            support_data: Data for adaptation
            query_data: Data for prediction
            target_col: Target column name
            sequence_length: Input sequence length
            
        Returns:
            Predictions for query data
        """
        # Create task
        task = self.create_task_from_data(
            pd.concat([support_data, query_data]), 
            target_col, 
            sequence_length
        )
        
        # Adapt model
        adapted_model = self.adapt_to_task(task)
        
        # Make predictions
        adapted_model.eval()
        with torch.no_grad():
            query_X, _ = self._create_sequences(
                query_data.drop(columns=[target_col]), 
                query_data[target_col], 
                sequence_length
            )
            query_X = torch.FloatTensor(query_X).to(self.config.device)
            predictions = adapted_model(query_X)
            
        return torch.softmax(predictions, dim=1).cpu().numpy()
    
    def evaluate_adaptation(self, tasks: List[Task]) -> float:
        """
        Evaluate adaptation performance on tasks.
        
        Args:
            tasks: Tasks for evaluation
            
        Returns:
            Average accuracy after adaptation
        """
        accuracies = []
        
        for task in tasks:
            # Adapt to task
            adapted_model = self.adapt_to_task(task)
            
            # Evaluate on query set
            adapted_model.eval()
            with torch.no_grad():
                query_pred = adapted_model(task.query_X)
                predicted_labels = torch.argmax(query_pred, dim=1)
                accuracy = (predicted_labels == task.query_y).float().mean().item()
                accuracies.append(accuracy)
        
        return np.mean(accuracies)
    
    def online_meta_update(self, new_task: Task, meta_lr_decay: float = 0.99):
        """
        Perform online meta-learning update with a single new task.
        
        Args:
            new_task: Newly observed market data task
            meta_lr_decay: Decay factor for meta-learning rate
        """
        # Adapt current meta-learning rate
        for param_group in self.meta_optimizer.param_groups:
            param_group['lr'] *= meta_lr_decay
        
        # Single task meta-update
        meta_loss = self.meta_update([new_task])
        self.meta_losses.append(meta_loss)
        
        self.logger.debug(f"Online meta-update: loss={meta_loss:.4f}")
    
    def get_feature_importance(self, task: Task) -> Dict[int, float]:
        """
        Compute feature importance using gradient-based attribution.
        
        Args:
            task: Task for analysis
            
        Returns:
            Feature importance scores
        """
        adapted_model = self.adapt_to_task(task)
        adapted_model.eval()
        
        # Enable gradient computation for inputs
        task.query_X.requires_grad_(True)
        
        # Forward pass
        output = adapted_model(task.query_X)
        
        # Compute gradients w.r.t. inputs
        target_class = torch.argmax(output, dim=1)
        grad_outputs = torch.zeros_like(output)
        grad_outputs.scatter_(1, target_class.unsqueeze(1), 1.0)
        
        gradients = torch.autograd.grad(
            outputs=output,
            inputs=task.query_X,
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=False
        )[0]
        
        # Compute importance as absolute gradient magnitude
        importance = torch.abs(gradients).mean(dim=(0, 1)).cpu().numpy()
        
        return {i: float(importance[i]) for i in range(len(importance))}
    
    def save_model(self, path: str):
        """Save the meta-learned model."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
            'config': self.config,
            'meta_losses': self.meta_losses,
            'validation_accuracies': self.validation_accuracies
        }, path)
        
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a pre-trained meta-learning model."""
        checkpoint = torch.load(path, map_location=self.config.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
        self.meta_losses = checkpoint.get('meta_losses', [])
        self.validation_accuracies = checkpoint.get('validation_accuracies', [])
        
        self.logger.info(f"Model loaded from {path}")

# Example usage and testing
if __name__ == "__main__":
    # Create sample configuration
    config = MAMLConfig(
        inner_lr=0.01,
        meta_lr=0.001,
        inner_steps=5,
        meta_batch_size=8,
        max_epochs=100,
        device='cpu'
    )
    
    # Initialize MAML trader
    input_size = 50  # Number of features
    maml_trader = MAMLTrader(config, input_size)
    
    # Create sample task generator (placeholder)
    def sample_task_generator():
        # Generate random task for demonstration
        support_X = torch.randn(32, 20, input_size)
        support_y = torch.randint(0, 3, (32,))
        query_X = torch.randn(16, 20, input_size)
        query_y = torch.randint(0, 3, (16,))
        
        return Task(support_X, support_y, query_X, query_y)
    
    # Train meta-learning model
    validation_tasks = [sample_task_generator() for _ in range(10)]
    maml_trader.train_meta_learning(sample_task_generator, validation_tasks)
    
    print("MAML training completed!")
    print(f"Final meta-loss: {maml_trader.meta_losses[-1]:.4f}")
    print(f"Final validation accuracy: {maml_trader.validation_accuracies[-1]:.4f}")