# Allium Integration Proposal: derive-flows

**Purpose**: This document outlines a proposed integration of Allium blockchain data into the derive-flows research platform. It is intended to support a request for Developer API access to Allium.

---

## 1. Project Overview: derive-flows

**derive-flows** is a research platform investigating whether unusual options flow from the Derive protocol can predict future spot price movements on crypto assets.

### Current Data Pipeline
| Data Source | Purpose | Method |
|-------------|---------|--------|
| Derive Telegram Channel | Options trade messages | Telethon ingestion + regex parsing |
| Hyperliquid API | Hourly spot/perp prices | REST API → Parquet files |

### Research Methodology
1. Parse structured trade data from Telegram messages (asset, strike, expiry, quantity, premium)
2. Compute "unusual score" based on notional value and moneyness
3. Flag trades >2σ above trailing average as unusual
4. Measure forward price returns (24h, 48h, 72h) after unusual signals
5. Statistical testing with bootstrap confidence intervals and temporal clustering correction

### Key Finding (Preliminary)
ETH call options flow shows a statistically significant **contrarian** relationship: Large OTM call buying preceded -2.83% average 72h returns (p=0.016).

### Acknowledged Limitations
- Single market regime (bear market, Dec 2025-Jan 2026)
- No multi-signal models (funding rates, open interest not integrated)
- No aggressor side data (buyer vs seller initiated)
- Small sample size after temporal clustering correction

---

## 2. Why Allium?

Allium's Hyperliquid data can directly address the current limitations of this research. The integration would enable **conditional signal analysis**—testing whether options flow combined with market microstructure context produces stronger predictive signals.

### Allium Data That Matters

| Allium Table | Data Fields | Research Value |
|--------------|-------------|----------------|
| `hyperliquid.raw.perpetual_market_asset_contexts` | `funding`, `premium`, `open_interest` | Multi-signal models; regime context |
| `hyperliquid.dex.trades` | `liquidated_user`, `liquidation_mark_price` | Liquidation prediction analysis |
| `hyperliquid.dex.trades` | `buyer_address`, `seller_address`, `side` | Cross-venue aggressor flow |
| `hyperliquid.metrics.overview` | Daily volume, OI, fees | Regime detection and classification |

### What Allium Enables That Direct APIs Cannot
1. **Enriched liquidation data** — Direct Hyperliquid API doesn't surface liquidation details in a queryable format
2. **Historical backfill** — Allium's backfilled data extends beyond API retention limits
3. **Aggregated metrics** — Pre-computed daily overviews reduce custom aggregation work
4. **Entity context** — Builder labels and address metadata for future analysis

---

## 3. Proposed Research Questions

With Allium data, derive-flows can investigate the following questions:

### 3.1 Conditional Signal Quality
> **Does unusual options flow combined with extreme funding rates produce a stronger signal?**

- **Hypothesis**: Unusual call buying during negative funding (market is net short) is more contrarian and thus more predictive
- **Analysis**: Stratify signals by funding rate quintiles, compare forward returns across strata
- **Allium Data**: `funding` from `perpetual_market_asset_contexts`

### 3.2 Open Interest Confirmation
> **Is unusual options flow more predictive when accompanied by rising open interest?**

- **Hypothesis**: Rising OI indicates conviction; signals with rising OI should be more informative
- **Analysis**: Compute OI change in 24h window around signal, correlate with forward returns
- **Allium Data**: `open_interest` from `perpetual_market_asset_contexts`

### 3.3 Liquidation Cascade Prediction
> **Do unusual options trades precede liquidation cascades?**

- **Hypothesis**: Sophisticated options traders may anticipate liquidation-driven volatility
- **Analysis**: For each unusual signal, compute forward 24h/72h liquidation volume, test correlation
- **Allium Data**: `liquidated_user`, `liquidation_mark_price` from `dex.trades`

### 3.4 Cross-Venue Aggressor Flow
> **Is options flow more predictive when concurrent Hyperliquid perp flow confirms or conflicts?**

- **Hypothesis**: Signals where options and perp flow conflict (e.g., call buying + net selling) may be noise
- **Analysis**: Compute net aggressor flow in ±1h window around options signal, condition analysis on agreement
- **Allium Data**: `side`, `usd_amount` from `dex.trades`

### 3.5 Regime Classification
> **Can we build a market regime classifier to understand when signals work?**

- **Current Problem**: Analysis covers only bear market; unclear if findings generalize
- **Analysis**: Use Allium metrics to classify periods as trending/ranging, test signal conditioned on regime
- **Allium Data**: `hyperliquid.metrics.overview` (daily volume, OI, liquidations)

---

## 4. Expected Deliverables

If granted Allium access, the following outputs would be produced:

### Research Report
- Title: *"Conditioning Options Flow Signals with Hyperliquid Market Context"*
- Contents:
  - Side-by-side comparison: unconditioned signal vs funding-conditioned signal
  - Liquidation cascade analysis results
  - Regime-specific signal performance
  - Bootstrap confidence intervals for all metrics

### Open-Source Integration
- Allium data fetching scripts added to derive-flows repository
- Documentation in CLAUDE.md for reproducing analysis
- Clear attribution: "Market microstructure data powered by Allium"

### Potential Publication
- Findings could be shared publicly (blog post, research note) if statistically robust
- Demonstrates Allium's value for quantitative crypto research

---

## 5. Technical Integration Approach

### Minimal Viable Integration
```
scripts/
├── fetch_allium_context.py       # Pull funding + OI data
└── fetch_allium_liquidations.py  # Pull liquidation events

data/allium/
├── market_context.parquet        # Hourly funding, OI, premium
└── liquidations.parquet          # Liquidation events

scripts/unusual_flow_analysis.py  # Extend with conditional analysis
```

### API Requirements
- **Endpoint**: REST API with SQL query support
- **Tables Needed**:
  - `hyperliquid.raw.perpetual_market_asset_contexts`
  - `hyperliquid.dex.trades`
  - `hyperliquid.metrics.overview`
- **Query Volume**: Low (historical backfill once, then periodic updates)
- **Time Range**: Dec 2025 – present (expandable as data grows)

---

## 6. Value to Allium

### Novel Use Case
Cross-venue research combining off-chain options flow (Derive/Telegram) with on-chain market microstructure (Hyperliquid via Allium) is not commonly done. This demonstrates Allium's utility beyond standard blockchain analytics.

### Methodological Rigor
derive-flows uses proper statistical methodology:
- Bootstrap confidence intervals (not just point estimates)
- Temporal clustering correction (accounting for overlapping signals)
- Out-of-sample considerations documented
- Limitations transparently stated

### Attribution and Visibility
- All Allium data usage will be clearly attributed
- Code and methodology will be open-source
- Findings can be shared publicly with Allium's permission

---

## 7. Specific Allium Data Request

| Table | Fields | Purpose |
|-------|--------|---------|
| `hyperliquid.raw.perpetual_market_asset_contexts` | timestamp, coin, funding, premium, open_interest, mark_price | Market context for conditional analysis |
| `hyperliquid.dex.trades` | timestamp, coin, side, usd_amount, liquidated_user, liquidation_mark_price | Liquidation and aggressor flow analysis |
| `hyperliquid.metrics.overview` | date, trading_volume, liquidations, open_interest | Regime classification |

**Assets**: ETH, BTC (primary), HYPE (secondary)
**Time Range**: December 2025 – present
**Refresh Frequency**: Weekly historical updates sufficient

---

## 8. Summary

derive-flows has established a research foundation for analyzing unusual options flow as a predictive signal. The current limitations—single regime, no multi-signal models, small sample size—can be directly addressed by Allium's Hyperliquid data.

The proposed integration would:
1. Enable conditional signal analysis (funding, OI, liquidations)
2. Produce publishable research with proper attribution
3. Demonstrate a novel cross-venue research use case for Allium

**Request**: Developer API access to Allium's Hyperliquid tables for research purposes.

---

## Contact

Repository: [derive-flows](https://github.com/your-username/derive-flows)
