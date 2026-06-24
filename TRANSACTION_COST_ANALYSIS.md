# Transaction Cost Analysis

Venue cost study for thin-edge signal viability. Captured June 2026.

## Why this matters

For a high-frequency, thin-edge strategy, the dominant adversary is not direction. It is cost. Each round trip pays the spread and the fee, and at high trade counts those costs compound faster than most models account for. A strategy can be right on direction and still lose, bled out by cumulative transaction cost. So before any model refinement is worth pursuing, the question is whether the per-trade edge can clear the per-trade cost on the venue being traded. This document establishes that cost floor.

## Execution model

One model is applied uniformly to every venue, stated once here. A maker order to enter, a taker order to exit. This reflects how a thin-edge strategy typically executes: it can post a limit order to enter patiently, but usually must take liquidity to exit with fill certainty. Round-trip cost is therefore the maker fee plus the taker fee. Break-even edge is the round-trip cost expressed as the minimum price move a trade must capture just to cover that cost, before any profit.

This is a spot-to-spot comparison. Perpetuals carry additional funding costs not modeled here and are noted separately.

## Cost table, base published rates

All figures are venue base-tier spot rates, captured June 2026. Fees change; each is dated and should be re-verified before any sizing decision.

| Venue | Maker | Taker | Round-trip (maker+taker) | Break-even edge per trade |
|---|---|---|---|---|
| Binance.US | 0.00% | 0.02% | 0.02% | 0.02% |
| Hyperliquid (spot) | 0.04% | 0.07% | 0.11% | 0.11% |
| OKX | 0.08% | 0.10% | 0.18% | 0.18% |
| Binance global | 0.10% | 0.10% | 0.20% | 0.20% |

## Cost table, best-case with available discounts

Each venue's standing discount applied uniformly, captured June 2026.

| Venue | Maker | Taker | Round-trip | Break-even edge per trade |
|---|---|---|---|---|
| Binance.US | 0.00% | 0.02% | 0.02% | 0.02% |
| Hyperliquid (with maker rebate at higher tiers) | ~0.00% | 0.07% | ~0.07% | ~0.07% |
| Binance global (BNB discount) | 0.075% | 0.075% | 0.15% | 0.15% |
| OKX (volume tier) | ~0.06% | ~0.075% | ~0.135% | ~0.135% |

## Finding

Round-trip cost varies by an order of magnitude across venues, from 0.02% to 0.20% at base rates. For a thin-edge strategy this gap is decisive, larger than any single model improvement. Binance.US and Hyperliquid clear a far lower break-even edge than Binance global and OKX, which means venue selection is the primary lever on whether the strategy is viable at all. A signal that cannot reliably capture more than 0.20% per trade is dead on Binance global and alive on Binance.US, on the same logic, purely from cost structure.

The practical implication: a thin-edge strategy is viable only on the low-cost venues, and venue selection should be treated as a first-order design decision, not an afterthought.

## Limitations

Slippage is not included. These break-even figures assume fills at the quoted price. Real slippage at any meaningful size raises the required edge above the figures here, and slippage on a less-liquid pair raises it further. A complete cost model requires order-book depth data at the intended trade size, which was not retrievable from public sources at capture time and remains an open input.

Funding costs on perpetuals are not modeled. The comparison is spot-to-spot. Any perpetual position carries recurring funding, settled hourly on Hyperliquid, which is additional to the fee round-trip.

Fees are a moving target. Every figure here is a June 2026 snapshot. Promotional rates, the OKX volume tier and the Hyperliquid maker rebate in particular, are time-limited and should be re-confirmed against the venue before use.

## Sources

Fee figures captured June 2026 from venue fee documentation and current fee reporting. Binance global and Hyperliquid rates were verified against the venues' own published fee pages. Binance.US (0% maker, 0.02% taker, effective April 2026) and OKX base spot rates were captured from current fee reporting at the same time. Re-verify against the live venue fee pages before any decision, as all rates are subject to change.
