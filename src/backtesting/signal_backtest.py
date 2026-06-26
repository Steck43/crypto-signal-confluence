"""
Walk-forward signal evaluation with purged cross-validation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from backtesting.purged_cv import PurgedKFold
from trading.signal_generator import SimplifiedInstitutionalSignalGenerator


def _sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    excess = returns - risk_free_rate / 252
    return float(excess.mean() / excess.std() * np.sqrt(252))


@dataclass
class BacktestMetrics:
    directional_accuracy: float
    precision: float
    recall: float
    f1: float
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    n_predictions: int
    buy_ratio: float
    sell_ratio: float
    hold_ratio: float


def _forward_return_label(
    prices: pd.Series,
    index: int,
    horizon: int,
    buy_threshold: float,
    sell_threshold: float,
) -> int:
    if index + horizon >= len(prices):
        return 1
    future_return = prices.iloc[index + horizon] / prices.iloc[index] - 1
    if future_return >= buy_threshold:
        return 2
    if future_return <= sell_threshold:
        return 0
    return 1


def _signal_to_class(signal: str) -> int:
    return {"sell": 0, "hold": 1, "buy": 2}.get(signal, 1)


def _frictionless_returns(
    prices: pd.Series,
    indices: List[int],
    signals: List[str],
    horizon: int,
) -> pd.Series:
    returns = []
    for idx, signal in zip(indices, signals):
        if idx + horizon >= len(prices):
            continue
        fwd = prices.iloc[idx + horizon] / prices.iloc[idx] - 1
        if signal == "buy":
            returns.append(fwd)
        elif signal == "sell":
            returns.append(-fwd)
        else:
            returns.append(0.0)
    return pd.Series(returns)


class PurgedSignalBacktester:
    """Evaluate institutional signals on purged walk-forward folds."""

    def __init__(
        self,
        label_horizon: int = 5,
        embargo: int = 5,
        n_splits: int = 5,
        buy_threshold: float = 0.015,
        sell_threshold: float = -0.015,
        min_history: int = 120,
    ) -> None:
        self.label_horizon = label_horizon
        self.embargo = embargo
        self.n_splits = n_splits
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_history = min_history

    async def run(
        self,
        market_data: pd.DataFrame,
        include_sentiment: bool,
        symbol: str = "SOL",
    ) -> BacktestMetrics:
        generator = SimplifiedInstitutionalSignalGenerator(
            include_sentiment=include_sentiment
        )
        init_window = market_data.iloc[: self.min_history].copy()
        await generator.initialize_system(init_window, symbol=symbol)

        prices = market_data["close"].reset_index(drop=True)
        splitter = PurgedKFold(
            n_splits=self.n_splits,
            label_horizon=self.label_horizon,
            embargo=self.embargo,
        )

        y_true: List[int] = []
        y_pred: List[int] = []
        eval_indices: List[int] = []
        signal_labels: List[str] = []
        bars_evaluated = 0

        for _train_idx, test_idx in splitter.split(market_data.values):
            for idx in test_idx:
                if idx < self.min_history:
                    continue
                bars_evaluated += 1
                if bars_evaluated % 500 == 0:
                    print(
                        f"Ensemble walk-forward: {bars_evaluated} bars evaluated "
                        f"(sentiment={'on' if include_sentiment else 'off'})..."
                    )
                window = market_data.iloc[: idx + 1].copy()
                result = await generator.generate_trading_signals(window, symbol=symbol)
                predicted = _signal_to_class(result["signal"])
                actual = _forward_return_label(
                    prices,
                    idx,
                    self.label_horizon,
                    self.buy_threshold,
                    self.sell_threshold,
                )
                y_true.append(actual)
                y_pred.append(predicted)
                eval_indices.append(int(idx))
                signal_labels.append(result["signal"])

        if not y_true:
            return BacktestMetrics(
                directional_accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                sharpe_ratio=0.0,
                total_return=0.0,
                max_drawdown=0.0,
                n_predictions=0,
                buy_ratio=0.0,
                sell_ratio=0.0,
                hold_ratio=0.0,
            )

        strat_returns = _frictionless_returns(
            prices, eval_indices, signal_labels, self.label_horizon
        )
        cumulative = (1 + strat_returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak

        n = len(signal_labels)
        return BacktestMetrics(
            directional_accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(
                precision_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            recall=float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            sharpe_ratio=_sharpe_ratio(strat_returns),
            total_return=float(cumulative.iloc[-1] - 1) if len(cumulative) else 0.0,
            max_drawdown=float(drawdown.min()) if len(drawdown) else 0.0,
            n_predictions=n,
            buy_ratio=signal_labels.count("buy") / n,
            sell_ratio=signal_labels.count("sell") / n,
            hold_ratio=signal_labels.count("hold") / n,
        )


def metrics_to_dict(metrics: BacktestMetrics) -> Dict[str, float]:
    return {
        "directional_accuracy": metrics.directional_accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "sharpe_ratio": metrics.sharpe_ratio,
        "total_return": metrics.total_return,
        "max_drawdown": metrics.max_drawdown,
        "n_predictions": float(metrics.n_predictions),
        "buy_ratio": metrics.buy_ratio,
        "sell_ratio": metrics.sell_ratio,
        "hold_ratio": metrics.hold_ratio,
    }
