# Derive Options Flow Signal Analysis

**TL;DR:** We tested whether unusual OTM options flow from Derive predicts future spot returns. ETH unusual call flow shows a statistically significant contrarian signal (bearish 72h, p=0.016), but after correcting for signal clustering, the 95% confidence interval includes zero—meaning the signal is **not reliably tradeable**.

## Hypothesis

Unusual out-of-the-money (OTM) options flow may predict future spot price returns or volatility. Specifically, we hypothesize that large, unusual options trades—detected by comparing notional size and moneyness to trailing averages—contain information about future price direction or volatility expansion.

## Data

| Metric | Value |
|--------|-------|
| Source | Derive Telegram feed + Hyperliquid spot prices |
| Date range | Dec 4, 2025 → Jan 31, 2026 (59 days) |
| Total trades | 7,314 |
| OTM trades | 5,920 |
| Unusual signals | 87 |

**Unusual signals by asset:**
- ETH: 64 (35 calls, 29 puts)
- BTC: 12 (6 calls, 6 puts)
- HYPE: 11 (5 calls, 6 puts)

## Methodology

### Signal Detection

1. **OTM filter:** Only trades where `is_itm = False`
2. **Unusual score:** `(notional / trailing_7d_avg) × (1 + |moneyness_pct| / 100)`
3. **Threshold:** Per-asset, `mean + 2σ` of unusual scores

### Forward Returns

- **Windows:** 24h and 72h forward spot returns
- **Baseline:** 500 random samples from same asset's price history

### Statistical Tests

- **Directional:** One-sample t-test vs zero; two-sample t-test vs baseline
- **Volatility:** Absolute returns compared to baseline via t-test

### Robustness Checks

- **Leave-3-out:** Cross-validation stability
- **Threshold sensitivity:** 1.5σ to 3.0σ thresholds
- **Signal clustering:** Merge signals within 72h windows

### Tradeability Assessment

- **Independent windows:** Count non-overlapping 72h periods
- **Bootstrap Sharpe CI:** 10,000 resamples for 95% confidence interval
- **Transaction costs:** Sharpe impact at 5-50 bps

## Results

### Directional Returns

Statistically significant results (p < 0.05):

| Asset | Instrument | Horizon | Mean Return | p-value (vs zero) | p-value (vs baseline) | n |
|-------|------------|---------|-------------|-------------------|----------------------|---|
| ETH | call | 72h | -2.83% | 0.016 | 0.010 | 31 |
| ETH | put | 24h | -1.04% | 0.053 | 0.048 | 29 |

*ETH unusual call flow is contrarian: large OTM call buying precedes 72h bearish moves.*

### Volatility Prediction

Statistically significant results (p < 0.05):

| Asset | Horizon | Mean Abs Return | Baseline | p-value |
|-------|---------|-----------------|----------|---------|
| ETH | 72h | 4.95% | 3.64% | 0.003 |
| HYPE | 24h | 9.55% | 3.84% | 0.0003 |

*Unusual flow precedes elevated volatility, particularly for HYPE.*

### Tradeability Analysis (ETH Call 72h)

| Metric | Raw | Corrected |
|--------|-----|-----------|
| Signals | 31 | 11 windows |
| Overlap | - | 64.5% |
| Mean return | 2.83% | 1.60% |
| Sharpe (annualized) | 7.12 | 2.99 |
| 95% CI | - | [-2.62, 11.43] |

**Conclusion:** 95% CI includes zero → **not tradeable with confidence**

### Threshold Sensitivity (ETH 72h)

| σ Threshold | n Signals | Mean Return |
|-------------|-----------|-------------|
| 1.5σ | 70 | -1.80% |
| 2.0σ | 64 | -1.80% |
| 2.5σ | 61 | -1.33% |
| 3.0σ | 54 | -1.13% |

*Signal persists across thresholds but magnitude decreases.*

## Key Findings

- **ETH unusual call flow is contrarian:** 72h bearish signal with p=0.016
- **64.5% of signals cluster:** 31 raw signals → 11 independent windows
- **Robustness:** Signal survives leave-3-out and threshold sensitivity tests
- **Not tradeable:** 95% bootstrap CI includes zero; insufficient sample size for confidence
- **Volatility timing:** HYPE shows strongest volatility prediction (p=0.0003 at 24h)
- **BTC/HYPE directional:** Insufficient sample (n<10 per instrument) for reliable inference

## Limitations

1. **No aggressor side data:** Cannot distinguish buyer vs seller initiated trades
2. **Short sample period:** 2 months, single market regime (Dec-Jan 2025-26)
3. **Small sample after filtering:** 31 ETH call signals → 11 independent windows
4. **Asset coverage:** BTC and HYPE have insufficient signals for robust inference
5. **No order book context:** Strike selection may be liquidity-driven, not directional

## Future Work

1. **Extend data collection:** 6-12 months for statistical power
2. **Aggressor side:** If available, separate buyer vs seller initiated flow
3. **Multi-signal model:** Combine with funding rates, open interest, implied vol
4. **Cross-venue:** Compare with other DeFi options venues (Lyra, Premia)
5. **Regime conditioning:** Test signal performance in trending vs ranging markets

## How to Run

### Prerequisites

```bash
pip install -e ".[dev]"
```

### Run Analysis

```bash
python scripts/unusual_flow_analysis.py
```

### Output Files

Results are written to `outputs/`:

| File | Description |
|------|-------------|
| `directional_results.csv` | Forward return statistics by asset/instrument/horizon |
| `volatility_results.csv` | Absolute return vs baseline comparison |
| `tradeability_analysis.csv` | Sharpe ratios, bootstrap CI, transaction costs |
| `unusual_trade_summary.csv` | Signal breakdown by DTE bucket |
| `selectivity_analysis.csv` | Signal frequency statistics |
| `threshold_sensitivity.csv` | Results across different σ thresholds |
| `vol_timing_results.csv` | Per-signal volatility timing data |

## Repository Structure

```
derive-flows/
├── src/derive_flows/       # Trade ingestion pipeline
│   ├── parser.py           # Message parsing
│   ├── storage.py          # SQLite backend
│   └── enrichment.py       # Computed fields
├── scripts/
│   ├── fetch_hyperliquid_prices.py  # Price data fetcher
│   └── unusual_flow_analysis.py     # Signal analysis
├── data/                   # Raw data (gitignored)
│   ├── trades.db
│   └── prices/
├── outputs/                # Analysis results
└── CLAUDE.md              # Development instructions
```

## Data Sources

- **Trade messages:** Derive Telegram channel via Telethon API
- **Spot prices:** Hyperliquid perpetuals API (ETH, BTC) and spot index (HYPE)
