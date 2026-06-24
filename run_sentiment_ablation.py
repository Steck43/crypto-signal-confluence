#!/usr/bin/env python3
"""
Validation hardening check and sentiment ablation on purged walk-forward folds.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from analysis.volume_anomaly_detection import InstitutionalVolumeAnomalyDetector
from backtesting.signal_backtest import PurgedSignalBacktester, metrics_to_dict

logging.basicConfig(level=logging.WARNING)


def generate_market_data(n_samples: int = 2400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_price = 100.0
    trend = 0.00005
    volatility = 0.035

    prices = [base_price]
    volumes = [1_000_000.0]
    for _ in range(1, n_samples):
        shock = rng.normal(0, volatility / np.sqrt(288))
        prices.append(prices[-1] * (1 + trend + shock))
        vol_shock = rng.normal(0, 0.25)
        if abs(shock) > volatility:
            vol_shock += rng.normal(0.4, 0.15)
        volumes.append(max(volumes[-1] * (1 + vol_shock), 50_000))

    rows = []
    start = datetime(2023, 1, 1)
    for i, (price, volume) in enumerate(zip(prices, volumes)):
        spread = price * 0.001
        rows.append(
            {
                "timestamp": start + timedelta(minutes=5 * i),
                "open": price + rng.uniform(-spread / 2, spread / 2),
                "high": price + rng.uniform(0, spread),
                "low": price - rng.uniform(0, spread),
                "close": price,
                "volume": volume,
            }
        )
    return pd.DataFrame(rows)


def _build_xgboost_features(market_data: pd.DataFrame) -> pd.DataFrame:
    from machine_learning.xgboost_predictor import XGBoostSignalPredictor

    detector = InstitutionalVolumeAnomalyDetector(
        algorithms=["isolation_forest"],
        contamination=0.05,
        lookback_period=100,
        random_state=42,
    )
    prepared = market_data.copy()
    prepared["price"] = prepared["close"]
    if "timestamp" not in prepared.columns:
        prepared["timestamp"] = pd.RangeIndex(len(prepared))
    if "high" not in prepared.columns:
        prepared["high"] = prepared["close"] * 1.01
    if "low" not in prepared.columns:
        prepared["low"] = prepared["close"] * 0.99
    detector.fit(prepared)
    volume_features = detector.engineer_institutional_features(prepared).select_dtypes(
        include=[np.number]
    )
    volume_features = volume_features.loc[:, ~volume_features.columns.duplicated()]
    volume_features = volume_features.apply(pd.to_numeric, errors="coerce").fillna(0)

    predictor = XGBoostSignalPredictor()
    sentiment = {"rss_score": 0.1, "av_score": 0.05, "confidence": 0.6, "article_count": 2}
    technical = {
        "rsi": 50.0,
        "momentum": 0.0,
        "volume_ratio": 1.0,
        "volatility": 0.02,
        "technical_score": 0.0,
        "confidence": 0.5,
    }
    return predictor.prepare_institutional_features(
        volume_features, sentiment, technical, market_data
    ).apply(pd.to_numeric, errors="coerce").fillna(0)


def run_validation_comparison(market_data: pd.DataFrame) -> dict:
    from machine_learning.xgboost_predictor import XGBoostSignalPredictor

    features = _build_xgboost_features(market_data)
    prices = market_data["close"]

    standard = XGBoostSignalPredictor()
    standard.fit(
        features,
        prices,
        validation_mode="standard",
        validation_splits=5,
        future_periods=5,
        embargo=5,
        buy_threshold=0.006,
        sell_threshold=-0.006,
        cv_only=True,
    )

    purged = XGBoostSignalPredictor()
    purged.fit(
        features,
        prices,
        validation_mode="purged",
        validation_splits=5,
        future_periods=5,
        embargo=5,
        buy_threshold=0.006,
        sell_threshold=-0.006,
        cv_only=True,
    )

    def summarize(scores: dict) -> dict:
        return {
            "accuracy_mean": float(np.mean(scores["accuracy"])),
            "accuracy_std": float(np.std(scores["accuracy"])),
            "precision_mean": float(np.mean(scores["precision"])),
            "recall_mean": float(np.mean(scores["recall"])),
            "f1_mean": float(np.mean(scores["f1"])),
        }

    standard_summary = summarize(standard.cv_scores)
    purged_summary = summarize(purged.cv_scores)

    delta = {
        key: purged_summary[key] - standard_summary[key]
        for key in standard_summary
        if key.endswith("_mean")
    }

    return {
        "standard": standard_summary,
        "purged_embargo": purged_summary,
        "delta_purged_minus_standard": delta,
        "metrics_moved": any(abs(v) > 1e-6 for v in delta.values()),
    }


async def run_sentiment_ablation(market_data: pd.DataFrame) -> dict:
    backtester = PurgedSignalBacktester(
        label_horizon=5,
        embargo=5,
        n_splits=5,
        min_history=120,
    )
    with_sentiment = await backtester.run(market_data, include_sentiment=True)
    without_sentiment = await backtester.run(market_data, include_sentiment=False)

    with_dict = metrics_to_dict(with_sentiment)
    without_dict = metrics_to_dict(without_sentiment)
    delta = {k: without_dict[k] - with_dict[k] for k in with_dict}

    return {
        "sentiment_in": with_dict,
        "sentiment_out": without_dict,
        "delta_out_minus_in": delta,
    }


def verdict(ablation: dict) -> str:
    delta = ablation["delta_out_minus_in"]
    acc_drop = -delta["directional_accuracy"]
    f1_drop = -delta["f1"]
    sharpe_drop = -delta["sharpe_ratio"]

    if ablation["sentiment_in"]["n_predictions"] < 50:
        return "inconclusive — too few purged test predictions"

    if acc_drop > 0.01 or f1_drop > 0.01 or sharpe_drop > 0.05:
        return "yes — removing sentiment reduces measured performance"
    if acc_drop < -0.01 or f1_drop < -0.01 or sharpe_drop < -0.05:
        return "no — removing sentiment improves measured performance"
    return "inconclusive — performance change within noise band on this sample"


def main() -> None:
    market_data = generate_market_data(n_samples=5000, seed=42)

    validation = run_validation_comparison(market_data)
    ablation = asyncio.run(run_sentiment_ablation(market_data))

    report = {
        "validation": validation,
        "ablation": ablation,
        "verdict": verdict(ablation),
        "caveats": [
            "Synthetic OHLCV data (seed=42); no live market or news feed.",
            "Sentiment uses hardcoded simulated RSS headlines, not real feeds.",
            "Frictionless fills; no fees, slippage, or latency.",
            "Volume detector initialized once on first 120 bars, not refit per fold.",
            "Ablation toggles rss_sentiment weight only in SimplifiedInstitutionalSignalGenerator.",
        ],
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
