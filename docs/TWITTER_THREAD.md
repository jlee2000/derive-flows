# Twitter Thread: Unusual Options Flow Research

A 3-tweet thread for Twitter Premium (extended tweets) summarizing the unusual options flow research findings.

---

## Tweet 1: Hook + Setup + Finding

Unusual OTM call flow on @DeriveXYZ is a contrarian signal - large call buying precedes bearish moves. Tested it rigorously. It's statistically significant and completely untradeable.

Scraped 7,314 trades over two months, flagged anything more than 2 standard deviations above the trailing 7-day average notional.

ETH calls showed -2.83% average return over 72 hours after unusual call flow (p=0.016). Beat the random baseline by 2.3% and held up across different threshold levels.

**Attachments:** `unusual_flow_eth.png`, `eth_signal_consistency.png`

---

## Tweet 2: The Kill Shot + Volatility Angle

Then I did the statistics properly and it falls apart.

64.5% of signals cluster within 72 hours of each other. What looks like 31 signals is really 11 independent trades. Once you account for that overlap, the bootstrap confidence interval is [-2.6%, +11.4%] - it includes zero. Can't trade that with any real confidence.

The actually interesting finding was volatility prediction. HYPE unusual flow predicted 9.55% forward 24h volatility vs 3.84% baseline (p=0.0003). ETH showed similar patterns. Doesn't tell you direction, but it clearly predicts that something is about to move.

Might be useful for volatility-targeting strategies.

**Attachments:** `clustering_breakdown.png`, `vol_prediction_bars.png`

---

## Tweet 3: Limitations + CTA

Two big caveats: tested only during a bear market when ETH dropped 25%, and only 11 independent trade windows which is borderline for sample size. Needs out-of-sample validation.

Full code and methodology: github.com/jlee2000/derive-flows

If anyone has aggressor data from Derive, access to other options protocols (Lyra, Premia), or ideas for extending this - DM me.

---

## Charts Summary

| Tweet | Charts |
|-------|--------|
| 1 | `outputs/unusual_flow_eth.png`, `outputs/eth_signal_consistency.png` |
| 2 | `outputs/thread/clustering_breakdown.png`, `outputs/thread/vol_prediction_bars.png` |
| 3 | None |

---

## Key Statistics Quick Reference

| Metric | Value |
|--------|-------|
| Total trades | 7,314 |
| Sample period | 59 days (Dec 4 - Jan 31) |
| Unusual signals (total) | 87 |
| ETH call signals | 31 |
| ETH call mean return (72h) | -2.83% |
| ETH call p-value | 0.016 |
| Signal clustering | 64.5% overlap |
| Independent windows | 11 |
| Bootstrap CI | [-2.62%, +11.43%] |
| HYPE vol (unusual, 24h) | 9.55% |
| HYPE vol (baseline, 24h) | 3.84% |
| HYPE vol p-value | 0.0003 |
