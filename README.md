# crypto-signal-confluence

A research framework for studying whether multi-signal fusion produces predictive edge in cryptocurrency markets under realistic cost and validation constraints.

## Background

This system started as a signal generator. It watched the market and notified me when its signals lined up, and I placed the trades by hand. I ran it that way for roughly fifteen months and traded its signals manually at a net profit.

Trading it by hand also showed me the ceiling. A manual operator adds two costs the signal does not, error and emotion. You misread, you hesitate, you size wrong, you hold too long. And you cannot watch a market that never closes. Crypto liquidity rotates through the US, European, and Asian sessions on no fixed schedule, and the move you want often comes at three in the morning. I ran the signal read across sessions by hand until I hit the limit every solo operator hits. One person cannot cover all of it, and discipline degrades when fatigue and emotion enter.

The path from there was straightforward. A signal generator, traded by hand. Then a machine-learning layer to sharpen the signals. Then automation, to take the human out of the execution loop and let it run every session without error, hesitation, or sleep. The next step after that was completing and validating the automation, but I had to put the build on hold before I got there.

I returned to it in 2026. Rather than pick up exactly where I stopped, the first thing I did was ask whether the architecture still held up after a year, given how fast this field moves. So I surveyed the current frontier, foundation models built specifically for financial time series, and concluded the existing approach was still sound and current, with the newer models worth evaluating rather than adopting wholesale. With the architecture validated as still appropriate, the work turned to the harder question, whether it actually has edge. That question is what this repository tests.

## The question

Before a machine trades these signals unattended, one thing has to hold. The edge has to be real. So this repository does not assume the manual results carry over. It tests the premise underneath them. When volume-anomaly detection, news sentiment, and technical features are fused into one ensemble decision, does the result carry predictive edge that survives honest validation and realistic trading costs.

The name is the thesis. In market terms, confluence is the condition where independent signals agree before action is taken. This system implements that idea directly, a weighted ensemble that only acts when its components align, and then asks whether that alignment actually predicts anything. This repository is not the bot. It is the instrument that asks whether the automated version would have an edge worth trading.

## What is validated and what is exploratory

This repository is organized by tier of evidence.

Validated: the multi-signal confluence harness, purged cross-validation, the sentiment ablation, the volume detector, and the cost analysis. These run under purged CV and reproduce the numbers reported below.

Exploratory: GRU and meta-learning model research, the adaptive ensemble, the macro regime detector, the paper-trading loop, and the dashboard. These are explorations of the design space. They are not validated, are not on the proven path, and some are incomplete. They are included to show the range of approaches considered.

Not included: live exchange execution beyond paper trading.

## Design under constraint

The system was built as an independent researcher on a student budget, and the constraints shaped the design in ways worth stating plainly, because each one was a deliberate tradeoff rather than an oversight.

The binding constraint was data. High-quality real-time sentiment and on-chain data carry subscription costs that run into the thousands per month, beyond a student budget. So the system was built on the data I could access, aggregated news sentiment from RSS rather than a premium real-time feed, and volume derived from standard OHLCV candles rather than paid order-flow data. The accepted cost is that these sources are coarser and more lagging than institutional-grade ones. I pursued research access to premium providers where I could. The roadmap below moves toward earlier, source-level data as access allows, which is the principled fix to this constraint rather than a workaround.

The second constraint was coverage, the 24/7 market one person cannot watch, which is why automation was the next step.

These are not apologies. They are the conditions the system was built under, and naming them is the difference between a system whose author understands its limits and one who does not.

## Why sentiment is in the system

Sentiment entered this system the hard way. I was trading a market that was behaving predictably, and a single high-impact actor moved it against the pattern with one post. Technical signals describe the market when it is left alone. They do not see a headline coming. That is the gap the sentiment component exists to close, and it is why the component is weighted to react to market-moving information rather than to background noise. Whether it succeeds at that on real data is one of the open questions below.

## Architecture

The system fuses three independent signal sources into a single decision, with an optional machine-learning overlay.

The primary path is a weighted ensemble. A volume-anomaly detector carries the largest weight and runs five unsupervised methods, Isolation Forest, Local Outlier Factor, Mahalanobis distance, statistical process control, and a one-class SVM, combined into a single anomaly score. A sentiment component derives a score from news headlines. A technical component contributes standard indicators. The three are fused in a weighted decision that emits buy, sell, or hold only when the combined score crosses a threshold, the confluence condition.

An optional layer adds an XGBoost classifier and a GRU forecaster under a separate fixed weighting, a second fusion stack stacked on the first. The core research question is studied on the primary ensemble path. The ML overlay is present but is not the center of the investigation.

The modules carry natural service boundaries, data ingestion, signal generation, execution, and monitoring, which would map to independent services in a production deployment. The system is kept as one repository here because it is a research artifact run as a unit, not a deployed system.

Execution stops at signal generation and paper trading. No live order placement is wired end to end. Credentials are read from environment variables and the encryption layer derives its key with a per-deployment salt, so the repository ships with no secrets and no live keys.

## Methodology

Validation is the part of this work that matters most, because in financial machine learning the validation method, not the model, is usually what separates a real result from an inflated one.

The backtest uses purged cross-validation with an embargo period, following the approach in López de Prado's *Advances in Financial Machine Learning*, chapter 7. Purging removes training samples whose label windows overlap the test window, and the embargo adds a gap after each test fold before training resumes. Both steps exist to stop information from leaking across the train-test boundary, which ordinary time-series cross-validation allows through overlapping labels and serial correlation. The implementation lives in `src/backtesting/purged_cv.py` as the `PurgedKFold` class.

The effect is measured rather than assumed. On the XGBoost path with a synthetic series and a fixed seed, moving from standard time-series splitting to purged and embargoed splitting changed the cross-validated metrics as follows. Accuracy and recall went from 80.32 percent to 80.51 percent. Precision went from 66.57 percent to 69.90 percent. F1 went from 72.29 percent to 72.75 percent. The deltas are small and mixed in direction on this synthetic set. The honest reading is that the standard split was not badly leaking on this particular synthetic data. The purpose of running purged validation is to check for that leakage and report what it finds.

Canonical XGBoost cross-validation results were produced on Windows with seed 42, `OMP_NUM_THREADS=1` (set inside `run_sentiment_ablation.py`), and single-threaded XGBoost (`n_jobs=1`). Cross-platform floating-point summation can shift metrics by roughly one to two percentage points on precision; the repository pins threads and seed so two clean clones on the same platform reproduce the reported table.

These numbers are produced by the committed code and reproduce from the run command below.

## Findings

Results are reported as they came out, including the inconclusive one.

The validation method behaves as designed. On this synthetic set, purged cross-validation produced XGBoost metrics close to standard time-series splitting: accuracy and recall 80.32 percent versus 80.51 percent, precision 66.57 percent versus 69.90 percent, F1 72.29 percent versus 72.75 percent. The gaps are small and are not read as evidence that purging materially changed the estimate on this data.

A prior defect in the volume scoring path was corrected. The institutional detector had been evaluated on single-bar windows that could not produce features; it now scores every bar from a rolling history window with the full five-algorithm ensemble, including Mahalanobis distance relative to the fit distribution. Anomaly scores and confidence values are nonzero on the ablation path.

The sentiment ablation is inconclusive on synthetic data because the ensemble holds on all 4,165 purged test bars in both arms. With sentiment included and with sentiment removed, the action distribution is identical: zero buy, zero sell, 4,165 hold. That is not a bug and not a claim that sentiment lacks edge. The weighted confluence signal does not cross the ensemble action threshold on this synthetic series, so the ablation toggle cannot separate sentiment's contribution. A run on real historical prices and timestamped news is the necessary next step to make the test conclusive. The harness for that run is in `run_sentiment_ablation.py`.

The transaction-cost analysis is the one component with a clean result. Across the venues studied, round-trip cost varies by more than an order of magnitude, and for a thin-edge strategy that gap is decisive. The cheapest venues clear a far lower break-even edge per trade than the standard-fee venues, which means venue selection is a larger lever on viability than any model choice. The full dated table, methodology, and sources are in [`TRANSACTION_COST_ANALYSIS.md`](TRANSACTION_COST_ANALYSIS.md).

## Limitations

These are stated as scope, not apology. Knowing precisely what a system has and has not demonstrated is the discipline the field runs on.

The system is complete through paper and synthetic backtesting. Live execution is out of current scope, not because it is hard to add, but because the prior questions, does the signal carry edge, do the costs allow it, are not yet answered, and live execution before those answers is how capital gets lost.

The validation results above are on synthetic data with a fixed seed, not real market data. They demonstrate that the method is implemented correctly, not that the system has edge.

The sentiment path currently uses simulated headlines rather than live feeds. Combined with the all-hold ensemble outcome on synthetic data, the ablation cannot reach a verdict until real news is wired in.

The walk-forward backtest assumes frictionless fills, no slippage and no latency. This inflates any performance figure the system produces and is the next validation gap to close after the data question.

During the ensemble ablation, the volume anomaly detector is fitted once on the first 120 bars and is not refit per purged fold. That is a stated methodology limitation, also noted in `run_sentiment_ablation.py`.

The ML-layer sentiment features were not fully isolated in the ablation, so even a real-data ablation on the primary path would be partial until the toggle is extended through every layer where sentiment appears.

Exploratory modules outside the validated closure may require optional dependencies beyond `requirements-ablation.txt` (for example torch, aiohttp, requests, cryptography, ta, tweepy). A missing-import error on those paths is expected unless those extras are installed.

Edge on real market data is unproven. That is the central honest status of this repository. The architecture is complete and the validation is rigorous. Whether the architecture produces edge is the open question this framework exists to answer.

## Roadmap

The next steps follow directly from the limitations and the constraints, in priority order.

Run the sentiment ablation on real historical price data and real timestamped news, and extend the component toggle through the ML ensemble and the aggregator so sentiment is fully isolated. This is the test that turns the central open question into an answer.

Replace the frictionless-fill assumption in the backtest with modeled slippage and fees, using the venue cost analysis already in hand.

Move the sentiment layer from lagging aggregated news toward earlier, source-level ingestion, which is the direct fix to the data constraint the system was built under. The value of sentiment as a signal depends on being early rather than on being published. This step is gated on the ablation first proving the component earns its weight.

## Context

One component of a broader research program on building, governing, and defending coordinated multi-agent systems.

## Repository layout

```
src/backtesting/purged_cv.py        Purged cross-validation with embargo (PurgedKFold)
src/backtesting/signal_backtest.py  Walk-forward signal evaluator
src/trading/signal_generator.py     Primary multi-signal ensemble
src/trading/risk_manager.py         Kelly-based position sizing (exploratory)
src/machine_learning/xgboost_predictor.py  Validated XGBoost + purged CV path
src/api/secure_manager.py           Credential vault exhibit (exploratory)
run_sentiment_ablation.py           Validation comparison and sentiment ablation
TRANSACTION_COST_ANALYSIS.md        Venue cost study and break-even analysis
```

## Running it

The validation and ablation run uses a synthetic series with a fixed seed and requires no credentials. Thread pinning and seeding are set inside `run_sentiment_ablation.py`; no extra environment variables are required on Windows. Expect about fifteen minutes end to end: XGBoost cross-validation prints first, then the ensemble walk-forward (progress every 500 bars).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-ablation.txt
PYTHONPATH=src python3 run_sentiment_ablation.py
```

On Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-ablation.txt
$env:PYTHONPATH="src"
python run_sentiment_ablation.py
```

To run exploratory paths that need credentials (paper trading, encrypted stores), copy `.env.example` to `.env` and supply your own keys. The repository ships with no credentials.

## References

### Implemented foundations

Methods and components actually implemented in this repository.

- López de Prado, M. (2018). *Advances in Financial Machine Learning*, ch. 7. Wiley. ISBN 978-1-119-48208-6. Purged and embargoed cross-validation, implemented as `PurgedKFold`.
- Kelly, J. L. (1956). A New Interpretation of Information Rate. *Bell System Technical Journal*, 35(4). Basis for the Kelly position sizing in `risk_manager.py`.
- Chen, T., Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*. Gradient-boosted classifier in the ML overlay.
- Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder. arXiv:1406.1078. GRU, used in the price forecaster.

### Current research informing the roadmap

Recent work surveyed when returning to this system in 2026, to assess whether the architecture remained current. These are under evaluation for possible future integration and are not implemented in this repository.

- Shi, Y. et al. (2025). Kronos: A Foundation Model for the Language of Financial Markets. arXiv:2508.02739. A foundation model trained natively on financial K-line data. Assessed against this system's task; its reported strengths are in cross-sectional ranking rather than single-pair directional signals, so it is a candidate to evaluate, not a drop-in replacement.
- Das, S. R., Goyal, T., Yadav, M. (2026). Multivariate Financial Forecasting using the Chronos Time Series Foundation Models. arXiv:2605.21504. Finds multivariate inputs improve forecasts but that mixing unrelated markets degrades accuracy, relevant to keeping signal inputs clean.
- Ansari, A. F. et al. (2025). Chronos-2: From Univariate to Universal Forecasting. arXiv:2510.15821. The foundation model applied in the work above.
- Parente, M., Rizzuti, L., Trerotola, M. (2024). A profitable trading algorithm for cryptocurrencies using a Neural Network model. *Expert Systems with Applications*, 238. DOI 10.1016/j.eswa.2023.121806. Read as related work on neural-network crypto signal generation; its wavelet-and-MLP approach is not implemented here.
