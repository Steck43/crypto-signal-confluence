# Enhancement Implementation Notes (Exploratory)

EXPLORATORY: design notes for integration experiments. Not validated. Not on the proven ablation path.

This document records directions explored alongside the validated harness (`run_sentiment_ablation.py`). It does not claim production readiness or measured trading outcomes.

## Directions explored

### Dynamic threshold adjustment
- **Module**: `src/trading/integrated_trading_system.py`
- **Intent**: Adaptive buy/sell thresholds as an alternative to fixed ensemble cutoffs.
- **Status**: Incomplete scaffold; data fetchers return empty structures.

### GPU-accelerated ML ensemble
- **Module**: `src/machine_learning/enhanced_ml_ensemble.py`
- **Intent**: Adaptive weights, uncertainty estimates, continuous-learning hooks.
- **Status**: Core feature-prep methods raise `NotImplementedError`.

### Enhanced sentiment
- **Module**: `src/sentiment_analysis/enhanced_sentiment_analyzer.py`
- **Intent**: Transformer + keyword fallback beyond simulated RSS in the primary generator.
- **Status**: Prototype; not wired to the validated path.

### Integrated trading system
- **Module**: `src/trading/integrated_trading_system.py`
- **Intent**: Unify ML, sentiment, volume, and dynamic thresholds in one loop.
- **Status**: Incomplete; volume analysis hook raises `NotImplementedError`.

## Running exploratory modules

Exploratory code may require optional dependencies (PyTorch, Freqtrade, etc.) not listed in `requirements-ablation.txt`. The validated reproduction path uses only `requirements-ablation.txt`.
