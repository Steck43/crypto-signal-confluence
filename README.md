# crypto-signal-confluence

A research framework for studying whether multi-signal fusion produces predictive edge in cryptocurrency markets under realistic cost and validation constraints.

## The question

This repository investigates a single question. When volume-anomaly detection, news sentiment, and technical features are fused into one ensemble decision, does the result carry predictive edge that survives honest validation and realistic trading costs.

The name is the thesis. In market terms, confluence is the condition where independent signals agree before action is taken. This system implements that idea directly, a weighted ensemble that only acts when its components align, and then asks whether that alignment actually predicts anything. The framing matters. This is not a profitable trading bot, and it does not claim to be. It is an instrument for measuring whether a common retail premise, that combining signals creates edge, holds up when you remove the things that usually inflate it.

## Architecture

The system fuses three independent signal sources into a single decision, with an optional machine-learning overlay.

The primary path is a weighted ensemble. A volume-anomaly detector carries the largest weight and runs five unsupervised methods, Isolation Forest, Local Outlier Factor, Mahalanobis distance, statistical process control, and a one-class SVM, combined into a single anomaly score. A sentiment component derives a score from news headlines. A technical component contributes standard indicators. The three are fused in a weighted decision that emits buy, sell, or hold only when the combined score crosses a threshold, the confluence condition.

An optional layer adds an XGBoost classifier and a GRU forecaster under a separate fixed weighting, a second fusion stack stacked on the first. The core research question is studied on the primary ensemble path. The ML overlay is present but is not the center of the investigation.

Execution stops at signal generation and paper trading. No live order placement is wired end to end. Credentials are read from environment variables and the encryption layer derives its key with a per-deployment salt, so the repository ships with no secrets and no live keys.

## Methodology

Validation is the part of this work that matters most, because in financial machine learning the validation method, not the model, is usually what separates a real result from an inflated one.

The backtest uses purged cross-validation with an embargo period, following the approach in López de Prado's *Advances in Financial Machine Learning*. Purging removes training samples whose label windows overlap the test window, and the embargo adds a gap after each test fold before training resumes. Both steps exist to stop information from leaking across the train-test boundary, which ordinary time-series cross-validation allows through overlapping labels and serial correlation. The implementation lives in `src/backtesting/purged_cv.py`.

The fix is observable, not assumed. Moving from standard time-series splitting to purged and embargoed splitting lowered the cross-validated metrics on the XGBoost path on a fixed synthetic reference run (seed 42, 5,000 bars): accuracy and recall each fell by roughly 0.34 percentage points, F1 by roughly 0.30, and precision by roughly 5.1. The expected direction when leakage is removed. A validation change that did not move the metrics would mean the change did nothing. This one moved them.

## Findings

Results are reported as they came out, including the inconclusive one.

Validation hardening worked. The purged and embargoed splitter produced the expected decrease in inflated metrics, confirming that the standard split had been leaking.

The sentiment ablation was inconclusive on the current measured path, and the reason is itself the finding. With sentiment included and with sentiment removed, the system produced identical results, because on the synthetic series with simulated headlines the ensemble emitted hold on all 4,165 purged test bars. A component cannot be measured in a system that never trades. So the honest verdict is not that sentiment fails. It is that the test could not isolate sentiment's contribution under these inputs, and a real verdict requires real market data and real news, which is scoped as the next step. The ablation harness that would produce that verdict is built and lives in `run_sentiment_ablation.py`.

The transaction-cost analysis is the one component with a clean result. Round-trip cost varies by an order of magnitude across venues, and for a thin-edge strategy venue selection is a primary lever on viability, larger than any single model choice. Full dated tables, the uniform execution model, methodology, limitations, and sources are in `TRANSACTION_COST_ANALYSIS.md`.

## Limitations

These are stated as scope, not apology. Knowing precisely what a system has and has not demonstrated is the discipline the field runs on.

The system is complete through paper and synthetic backtesting. Live execution is out of current scope, not because it is hard to add, but because the prior questions, does the signal carry edge, do the costs allow it, are not yet answered, and live execution before those answers is how capital gets lost.

The sentiment path currently uses simulated headlines rather than live feeds, which is why the ablation could not reach a verdict.

The walk-forward backtest assumes frictionless fills, no slippage and no latency. This inflates any performance figure the system produces and is the next validation gap to close after the data question.

The ML-layer sentiment features were not fully isolated in the ablation, so even a real-data ablation on the primary path would be partial until the toggle is extended through every layer where sentiment appears.

Edge on real market data is unproven. That is the central honest status of this repository. The architecture is complete and the validation is rigorous. Whether the architecture produces edge is the open question this framework exists to answer.

## Roadmap

The next steps follow directly from the limitations, in priority order.

Run the sentiment ablation on real historical price data and real timestamped news, and extend the component toggle through the ML ensemble and the aggregator so sentiment is fully isolated. This is the test that turns the central open question into an answer.

Replace the frictionless-fill assumption in the backtest with modeled slippage and fees, using the venue cost analysis already in hand.

If, and only if, the sentiment component proves it earns its weight, upgrade the sentiment layer from lagging aggregated news toward earlier, source-level ingestion, since the value of sentiment as a signal depends on being early rather than on being published.

## Context

One component of a broader research program on building, governing, and defending coordinated multi-agent systems.

## Repository layout

```
src/backtesting/purged_cv.py        Purged cross-validation with embargo
src/backtesting/signal_backtest.py  Walk-forward signal evaluator
src/trading/signal_generator.py     Primary multi-signal ensemble
src/machine_learning/               XGBoost and GRU overlay
src/api/secure_manager.py           Environment-based credential handling
run_sentiment_ablation.py           Validation comparison and sentiment ablation
```

## Running it

The system reads all credentials from environment variables. Copy `.env.example` to `.env` and supply your own keys to run the data-dependent paths. The repository ships with no credentials.

```bash
cp .env.example .env
# add your keys to .env
PYTHONPATH=src python run_sentiment_ablation.py
```
