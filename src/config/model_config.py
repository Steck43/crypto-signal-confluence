"""
Model Configuration Module

Defines configuration parameters for machine learning models used in the
institutional AI crypto trading system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ModelType(Enum):
    """Machine learning model types."""
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    RANDOM_FOREST = "random_forest"
    NEURAL_NETWORK = "neural_network"
    GRADIENT_BOOSTING = "gradient_boosting"
    SUPPORT_VECTOR = "support_vector"
    LINEAR_REGRESSION = "linear_regression"
    RIDGE_REGRESSION = "ridge_regression"
    LASSO_REGRESSION = "lasso_regression"
    ELASTIC_NET = "elastic_net"


@dataclass
class ModelConfig:
    """Machine learning model configuration."""
    
    # Model selection
    primary_model: ModelType = ModelType.XGBOOST
    ensemble_models: List[ModelType] = field(default_factory=lambda: [
        ModelType.XGBOOST,
        ModelType.LIGHTGBM,
        ModelType.RANDOM_FOREST
    ])
    
    # Training parameters
    train_test_split: float = 0.8
    validation_split: float = 0.2
    random_state: int = 42
    n_jobs: int = -1  # Use all CPU cores
    
    # Feature engineering
    feature_selection_method: str = "mutual_info"
    max_features: int = 50
    feature_importance_threshold: float = 0.01
    use_pca: bool = False
    pca_components: int = 20
    
    # Hyperparameter tuning
    enable_hyperparameter_tuning: bool = True
    tuning_method: str = "optuna"  # optuna, grid_search, random_search
    n_trials: int = 100
    cv_folds: int = 5
    
    # Model persistence
    save_models: bool = True
    model_directory: str = "models"
    model_versioning: bool = True
    
    # Performance monitoring
    enable_model_monitoring: bool = True
    performance_threshold: float = 0.6
    retrain_threshold: float = 0.5
    
    # XGBoost specific
    xgboost_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    })
    
    # LightGBM specific
    lightgbm_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    })
    
    # Random Forest specific
    random_forest_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42
    })
    
    # Neural Network specific
    neural_network_params: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_layer_sizes": (100, 50, 25),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0001,
        "learning_rate": "adaptive",
        "max_iter": 1000,
        "random_state": 42
    })
    
    def get_model_params(self, model_type: ModelType) -> Dict[str, Any]:
        """Get parameters for a specific model type."""
        param_maps = {
            ModelType.XGBOOST: self.xgboost_params,
            ModelType.LIGHTGBM: self.lightgbm_params,
            ModelType.RANDOM_FOREST: self.random_forest_params,
            ModelType.NEURAL_NETWORK: self.neural_network_params
        }
        return param_maps.get(model_type, {})
    
    def update_model_params(self, model_type: ModelType, params: Dict[str, Any]):
        """Update parameters for a specific model type."""
        if model_type == ModelType.XGBOOST:
            self.xgboost_params.update(params)
        elif model_type == ModelType.LIGHTGBM:
            self.lightgbm_params.update(params)
        elif model_type == ModelType.RANDOM_FOREST:
            self.random_forest_params.update(params)
        elif model_type == ModelType.NEURAL_NETWORK:
            self.neural_network_params.update(params)


# Default model configuration
default_model_config = ModelConfig() 