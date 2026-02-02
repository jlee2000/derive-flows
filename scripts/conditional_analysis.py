#!/usr/bin/env python3
"""Conditional Signal Analysis with Allium Data.

This script extends the unusual flow analysis with conditioning on:
- Funding rates (from Allium market context)
- Open interest changes
- Liquidation volume
- Aggressor flow

Requires Allium data to be fetched first:
  python scripts/fetch_allium_context.py
  python scripts/fetch_allium_liquidations.py
"""

import asyncio
from pathlib import Path
from datetime import timedelta

import pandas as pd
import numpy as np
from scipy import stats

from derive_flows.storage import TradeStorage
from derive_flows.enrichment import enrich_trades


# ============================================================================
# Data Loading
# ============================================================================


async def load_all_data():
    """Load trades, prices, and Allium data."""
    # Load trades
    storage = TradeStorage(Path("./data/trades.db"))
    trades = await storage.to_dataframe()
    trades = enrich_trades(trades)

    if trades["timestamp"].dt.tz is None:
        trades["timestamp"] = trades["timestamp"].dt.tz_localize("UTC")

    # Load price data
    prices = {}
    for asset in ["eth", "btc", "hype"]:
        price_path = Path(f"./data/prices/{asset}_hourly_prices.parquet")
        if price_path.exists():
            df = pd.read_parquet(price_path)
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
            prices[asset] = df

    # Load Allium data
    allium_dir = Path("./data/allium")
    allium_data = {}

    market_context_path = allium_dir / "market_context.parquet"
    if market_context_path.exists():
        allium_data["market_context"] = pd.read_parquet(market_context_path)
        print(f"  Loaded market context: {len(allium_data['market_context'])} rows")

    liquidations_path = allium_dir / "liquidations.parquet"
    if liquidations_path.exists():
        allium_data["liquidations"] = pd.read_parquet(liquidations_path)
        print(f"  Loaded liquidations: {len(allium_data['liquidations'])} rows")

    aggressor_path = allium_dir / "aggressor_flow.parquet"
    if aggressor_path.exists():
        allium_data["aggressor_flow"] = pd.read_parquet(aggressor_path)
        print(f"  Loaded aggressor flow: {len(allium_data['aggressor_flow'])} rows")

    return trades, prices, allium_data


# ============================================================================
# Conditioning Functions
# ============================================================================


def get_funding_at_time(market_context: pd.DataFrame, asset: str, timestamp: pd.Timestamp) -> float:
    """Get funding rate closest to timestamp for given asset."""
    asset_data = market_context[market_context["coin"] == asset].copy()
    if asset_data.empty:
        return np.nan

    asset_data = asset_data.sort_values("timestamp")
    idx = asset_data["timestamp"].searchsorted(timestamp)

    if idx >= len(asset_data):
        idx = len(asset_data) - 1
    elif idx > 0:
        # Get closest timestamp
        before = abs((asset_data.iloc[idx - 1]["timestamp"] - timestamp).total_seconds())
        after = abs((asset_data.iloc[idx]["timestamp"] - timestamp).total_seconds())
        if before < after:
            idx = idx - 1

    return asset_data.iloc[idx]["funding"]


def get_oi_change(market_context: pd.DataFrame, asset: str, timestamp: pd.Timestamp,
                  lookback_hours: int = 24) -> float:
    """Compute OI change in lookback window before timestamp."""
    asset_data = market_context[market_context["coin"] == asset].copy()
    if asset_data.empty:
        return np.nan

    asset_data = asset_data.sort_values("timestamp")

    # Find index closest to timestamp
    idx = asset_data["timestamp"].searchsorted(timestamp)
    if idx >= len(asset_data):
        idx = len(asset_data) - 1

    # Find index at lookback
    lookback_time = timestamp - timedelta(hours=lookback_hours)
    lookback_idx = asset_data["timestamp"].searchsorted(lookback_time)

    if lookback_idx >= idx or lookback_idx >= len(asset_data):
        return np.nan

    oi_now = asset_data.iloc[idx]["open_interest"]
    oi_before = asset_data.iloc[lookback_idx]["open_interest"]

    if oi_before == 0 or pd.isna(oi_before):
        return np.nan

    return (oi_now - oi_before) / oi_before


def get_liquidation_volume(liquidations: pd.DataFrame, asset: str, timestamp: pd.Timestamp,
                           forward_hours: int = 24) -> dict:
    """Compute liquidation volume in forward window after timestamp."""
    asset_data = liquidations[liquidations["coin"] == asset].copy()
    if asset_data.empty:
        return {"long_liq_usd": 0, "short_liq_usd": 0, "total_liq_usd": 0}

    end_time = timestamp + timedelta(hours=forward_hours)
    window_data = asset_data[
        (asset_data["timestamp"] >= timestamp) &
        (asset_data["timestamp"] < end_time)
    ]

    # Long liquidations = forced sells, Short liquidations = forced buys
    long_liq = window_data[window_data["side"] == "sell"]["usd_amount"].sum()
    short_liq = window_data[window_data["side"] == "buy"]["usd_amount"].sum()

    return {
        "long_liq_usd": long_liq,
        "short_liq_usd": short_liq,
        "total_liq_usd": long_liq + short_liq,
    }


def get_aggressor_flow(aggressor_flow: pd.DataFrame, asset: str, timestamp: pd.Timestamp,
                       window_hours: int = 2) -> float:
    """Get net aggressor flow in window around timestamp."""
    asset_data = aggressor_flow[aggressor_flow["coin"] == asset].copy()
    if asset_data.empty or "net_flow_usd" not in asset_data.columns:
        return np.nan

    start_time = timestamp - timedelta(hours=window_hours // 2)
    end_time = timestamp + timedelta(hours=window_hours // 2)

    window_data = asset_data[
        (asset_data["hour"] >= start_time) &
        (asset_data["hour"] < end_time)
    ]

    if window_data.empty:
        return np.nan

    return window_data["net_flow_usd"].sum()


# ============================================================================
# Enrichment
# ============================================================================


def enrich_with_allium(unusual_trades: pd.DataFrame, allium_data: dict) -> pd.DataFrame:
    """Add Allium-derived fields to unusual trades DataFrame."""
    df = unusual_trades.copy()

    # Initialize columns
    df["funding_rate"] = np.nan
    df["oi_change_24h"] = np.nan
    df["forward_liq_24h"] = np.nan
    df["forward_liq_72h"] = np.nan
    df["aggressor_net_flow"] = np.nan

    market_context = allium_data.get("market_context")
    liquidations = allium_data.get("liquidations")
    aggressor_flow = allium_data.get("aggressor_flow")

    for idx, row in df.iterrows():
        asset = row["asset"]
        timestamp = row["timestamp"]

        # Funding rate at signal time
        if market_context is not None and not market_context.empty:
            df.at[idx, "funding_rate"] = get_funding_at_time(market_context, asset, timestamp)
            df.at[idx, "oi_change_24h"] = get_oi_change(market_context, asset, timestamp, 24)

        # Forward liquidation volume
        if liquidations is not None and not liquidations.empty:
            liq_24h = get_liquidation_volume(liquidations, asset, timestamp, 24)
            liq_72h = get_liquidation_volume(liquidations, asset, timestamp, 72)
            df.at[idx, "forward_liq_24h"] = liq_24h["total_liq_usd"]
            df.at[idx, "forward_liq_72h"] = liq_72h["total_liq_usd"]

        # Aggressor flow
        if aggressor_flow is not None and not aggressor_flow.empty:
            df.at[idx, "aggressor_net_flow"] = get_aggressor_flow(aggressor_flow, asset, timestamp, 2)

    # Compute funding quintiles
    df["funding_quintile"] = pd.qcut(
        df["funding_rate"].rank(method="first"),
        q=5,
        labels=["Q1 (most negative)", "Q2", "Q3", "Q4", "Q5 (most positive)"],
    )

    # OI change direction
    df["oi_rising"] = df["oi_change_24h"] > 0

    # Aggressor flow direction
    df["aggressor_bullish"] = df["aggressor_net_flow"] > 0

    return df


# ============================================================================
# Conditional Analysis
# ============================================================================


def analyze_by_funding_quintile(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze returns stratified by funding rate quintile."""
    results = []

    for asset in df["asset"].unique():
        asset_df = df[df["asset"] == asset]

        for instrument in asset_df["instrument"].unique():
            inst_df = asset_df[asset_df["instrument"] == instrument]

            for quintile in inst_df["funding_quintile"].dropna().unique():
                q_df = inst_df[inst_df["funding_quintile"] == quintile]

                if len(q_df) < 2:
                    continue

                returns_72h = q_df["return_72h"].dropna()

                if len(returns_72h) < 2:
                    continue

                mean_ret = returns_72h.mean()
                std_ret = returns_72h.std()
                t_stat, p_val = stats.ttest_1samp(returns_72h, 0)

                results.append({
                    "asset": asset,
                    "instrument": instrument,
                    "funding_quintile": quintile,
                    "n_signals": len(returns_72h),
                    "mean_return_72h": mean_ret,
                    "std_return_72h": std_ret,
                    "t_stat": t_stat,
                    "p_value": p_val,
                })

    return pd.DataFrame(results)


def analyze_by_oi_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze returns stratified by OI change direction."""
    results = []

    for asset in df["asset"].unique():
        asset_df = df[df["asset"] == asset]

        for instrument in asset_df["instrument"].unique():
            inst_df = asset_df[asset_df["instrument"] == instrument]

            for oi_rising in [True, False]:
                oi_df = inst_df[inst_df["oi_rising"] == oi_rising]

                if len(oi_df) < 2:
                    continue

                returns_72h = oi_df["return_72h"].dropna()

                if len(returns_72h) < 2:
                    continue

                mean_ret = returns_72h.mean()
                std_ret = returns_72h.std()
                t_stat, p_val = stats.ttest_1samp(returns_72h, 0)

                results.append({
                    "asset": asset,
                    "instrument": instrument,
                    "oi_direction": "Rising" if oi_rising else "Falling",
                    "n_signals": len(returns_72h),
                    "mean_return_72h": mean_ret,
                    "std_return_72h": std_ret,
                    "t_stat": t_stat,
                    "p_value": p_val,
                })

    return pd.DataFrame(results)


def analyze_liquidation_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze correlation between unusual signals and forward liquidations."""
    results = []

    for asset in df["asset"].unique():
        asset_df = df[df["asset"] == asset]

        for instrument in asset_df["instrument"].unique():
            inst_df = asset_df[asset_df["instrument"] == instrument]

            # Correlation between unusual score and forward liquidation volume
            valid = inst_df.dropna(subset=["unusual_score", "forward_liq_72h"])

            if len(valid) < 3:
                continue

            corr_72h, p_corr_72h = stats.spearmanr(
                valid["unusual_score"],
                valid["forward_liq_72h"]
            )

            # Mean liquidation volume after unusual signals
            mean_liq_24h = inst_df["forward_liq_24h"].mean()
            mean_liq_72h = inst_df["forward_liq_72h"].mean()

            results.append({
                "asset": asset,
                "instrument": instrument,
                "n_signals": len(valid),
                "mean_forward_liq_24h": mean_liq_24h,
                "mean_forward_liq_72h": mean_liq_72h,
                "corr_score_vs_liq_72h": corr_72h,
                "p_value_corr": p_corr_72h,
            })

    return pd.DataFrame(results)


def analyze_aggressor_confirmation(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze returns when aggressor flow confirms or conflicts with signal."""
    results = []

    for asset in df["asset"].unique():
        asset_df = df[df["asset"] == asset]

        for instrument in ["call", "put"]:
            inst_df = asset_df[asset_df["instrument"] == instrument]

            if inst_df.empty:
                continue

            # For calls: bullish aggressor = confirmation, bearish = conflict
            # For puts: bearish aggressor = confirmation, bullish = conflict
            if instrument == "call":
                inst_df = inst_df.copy()
                inst_df["confirms"] = inst_df["aggressor_bullish"]
            else:
                inst_df = inst_df.copy()
                inst_df["confirms"] = ~inst_df["aggressor_bullish"]

            for confirms in [True, False]:
                subset = inst_df[inst_df["confirms"] == confirms]

                if len(subset) < 2:
                    continue

                returns_72h = subset["return_72h"].dropna()

                if len(returns_72h) < 2:
                    continue

                mean_ret = returns_72h.mean()
                std_ret = returns_72h.std()
                t_stat, p_val = stats.ttest_1samp(returns_72h, 0)

                results.append({
                    "asset": asset,
                    "instrument": instrument,
                    "aggressor_confirms": confirms,
                    "n_signals": len(returns_72h),
                    "mean_return_72h": mean_ret,
                    "std_return_72h": std_ret,
                    "t_stat": t_stat,
                    "p_value": p_val,
                })

    return pd.DataFrame(results)


# ============================================================================
# Output
# ============================================================================


def print_conditional_findings(
    funding_results: pd.DataFrame,
    oi_results: pd.DataFrame,
    liq_results: pd.DataFrame,
    aggressor_results: pd.DataFrame,
):
    """Print key findings from conditional analysis."""
    print("\n" + "=" * 70)
    print("CONDITIONAL ANALYSIS FINDINGS")
    print("=" * 70)

    # Funding rate conditioning
    print("\n--- Funding Rate Conditioning ---")
    if not funding_results.empty:
        sig = funding_results[funding_results["p_value"] < 0.1]
        if not sig.empty:
            print("Marginally significant results (p < 0.1):")
            for _, row in sig.iterrows():
                direction = "+" if row["mean_return_72h"] > 0 else ""
                print(f"  {row['asset']} {row['instrument']} @ {row['funding_quintile']}: "
                      f"{direction}{row['mean_return_72h']*100:.2f}% (p={row['p_value']:.3f}, n={row['n_signals']})")
        else:
            print("No significant results by funding quintile.")
    else:
        print("No funding rate data available.")

    # OI conditioning
    print("\n--- OI Change Conditioning ---")
    if not oi_results.empty:
        sig = oi_results[oi_results["p_value"] < 0.1]
        if not sig.empty:
            print("Marginally significant results (p < 0.1):")
            for _, row in sig.iterrows():
                direction = "+" if row["mean_return_72h"] > 0 else ""
                print(f"  {row['asset']} {row['instrument']} @ OI {row['oi_direction']}: "
                      f"{direction}{row['mean_return_72h']*100:.2f}% (p={row['p_value']:.3f}, n={row['n_signals']})")
        else:
            print("No significant results by OI direction.")
    else:
        print("No OI data available.")

    # Liquidation prediction
    print("\n--- Liquidation Prediction ---")
    if not liq_results.empty:
        for _, row in liq_results.iterrows():
            sig_marker = "*" if row["p_value_corr"] < 0.1 else ""
            print(f"  {row['asset']} {row['instrument']}: "
                  f"corr(score, 72h liq) = {row['corr_score_vs_liq_72h']:.3f}{sig_marker} "
                  f"(mean 72h liq: ${row['mean_forward_liq_72h']:,.0f})")
    else:
        print("No liquidation data available.")

    # Aggressor confirmation
    print("\n--- Aggressor Flow Confirmation ---")
    if not aggressor_results.empty:
        sig = aggressor_results[aggressor_results["p_value"] < 0.1]
        if not sig.empty:
            print("Marginally significant results (p < 0.1):")
            for _, row in sig.iterrows():
                conf_str = "confirms" if row["aggressor_confirms"] else "conflicts"
                direction = "+" if row["mean_return_72h"] > 0 else ""
                print(f"  {row['asset']} {row['instrument']} (aggressor {conf_str}): "
                      f"{direction}{row['mean_return_72h']*100:.2f}% (p={row['p_value']:.3f}, n={row['n_signals']})")
        else:
            print("No significant results by aggressor confirmation.")
    else:
        print("No aggressor flow data available.")


# ============================================================================
# Main
# ============================================================================


async def main():
    print("=" * 70)
    print("CONDITIONAL SIGNAL ANALYSIS WITH ALLIUM DATA")
    print("=" * 70)

    output_dir = Path("./outputs")
    output_dir.mkdir(exist_ok=True)

    # Load data
    print("\n[1/6] Loading data...")
    trades, prices, allium_data = await load_all_data()
    print(f"  Trades: {len(trades)}")

    if not allium_data:
        print("\nERROR: No Allium data found. Run fetch scripts first:")
        print("  python scripts/fetch_allium_context.py")
        print("  python scripts/fetch_allium_liquidations.py")
        return

    # Load unusual trades with returns (from previous analysis)
    unusual_path = output_dir / "unusual_trades_with_returns.parquet"
    if not unusual_path.exists():
        print("\nWARNING: unusual_trades_with_returns.parquet not found.")
        print("Running basic unusual flow pipeline first...")

        # Import and run the base analysis to generate the file
        from unusual_flow_analysis import (
            filter_otm, compute_unusual_score, flag_unusual, join_forward_prices
        )

        trades_otm = filter_otm(trades)
        trades_otm = compute_unusual_score(trades, trades_otm)
        trades_otm = flag_unusual(trades_otm)
        unusual_trades = trades_otm[trades_otm["is_unusual"]].copy()
        unusual_with_returns = join_forward_prices(unusual_trades, prices)

        if not unusual_with_returns.empty:
            unusual_with_returns.to_parquet(unusual_path, index=False)
            print(f"  Generated: {unusual_path}")
    else:
        unusual_with_returns = pd.read_parquet(unusual_path)
        print(f"  Loaded unusual trades: {len(unusual_with_returns)}")

    if unusual_with_returns.empty:
        print("\nERROR: No unusual trades with returns. Cannot proceed.")
        return

    # Ensure timestamp is tz-aware
    if unusual_with_returns["timestamp"].dt.tz is None:
        unusual_with_returns["timestamp"] = unusual_with_returns["timestamp"].dt.tz_localize("UTC")

    # Enrich with Allium data
    print("\n[2/6] Enriching with Allium data...")
    enriched = enrich_with_allium(unusual_with_returns, allium_data)
    print(f"  Funding data coverage: {enriched['funding_rate'].notna().sum()}/{len(enriched)}")
    print(f"  OI change coverage: {enriched['oi_change_24h'].notna().sum()}/{len(enriched)}")
    print(f"  Liquidation coverage: {enriched['forward_liq_72h'].notna().sum()}/{len(enriched)}")
    print(f"  Aggressor coverage: {enriched['aggressor_net_flow'].notna().sum()}/{len(enriched)}")

    # Save enriched data
    enriched.to_parquet(output_dir / "unusual_trades_enriched.parquet", index=False)
    print(f"  Saved: unusual_trades_enriched.parquet")

    # Conditional analyses
    print("\n[3/6] Analyzing by funding quintile...")
    funding_results = analyze_by_funding_quintile(enriched)
    if not funding_results.empty:
        funding_results.to_csv(output_dir / "funding_quintile_analysis.csv", index=False)
        print(f"  Saved: funding_quintile_analysis.csv ({len(funding_results)} rows)")

    print("\n[4/6] Analyzing by OI direction...")
    oi_results = analyze_by_oi_direction(enriched)
    if not oi_results.empty:
        oi_results.to_csv(output_dir / "oi_direction_analysis.csv", index=False)
        print(f"  Saved: oi_direction_analysis.csv ({len(oi_results)} rows)")

    print("\n[5/6] Analyzing liquidation prediction...")
    liq_results = analyze_liquidation_prediction(enriched)
    if not liq_results.empty:
        liq_results.to_csv(output_dir / "liquidation_prediction.csv", index=False)
        print(f"  Saved: liquidation_prediction.csv ({len(liq_results)} rows)")

    print("\n[6/6] Analyzing aggressor confirmation...")
    aggressor_results = analyze_aggressor_confirmation(enriched)
    if not aggressor_results.empty:
        aggressor_results.to_csv(output_dir / "aggressor_confirmation.csv", index=False)
        print(f"  Saved: aggressor_confirmation.csv ({len(aggressor_results)} rows)")

    # Print findings
    print_conditional_findings(funding_results, oi_results, liq_results, aggressor_results)

    print("\n" + "=" * 70)
    print("CONDITIONAL ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    asyncio.run(main())
