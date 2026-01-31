#!/usr/bin/env python3
"""Unusual Options Flow Analysis Pipeline.

Identifies unusual options flow and measures predictive power for subsequent
spot price movements.
"""

import asyncio
from pathlib import Path
from datetime import timedelta

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from derive_flows.storage import TradeStorage
from derive_flows.enrichment import enrich_trades


# ============================================================================
# Data Loading
# ============================================================================


async def load_data():
    """Load trades from SQLite and price data from parquet files."""
    # Load trades from SQLite
    storage = TradeStorage(Path("./data/trades.db"))
    trades = await storage.to_dataframe()
    trades = enrich_trades(trades)

    # Ensure timestamp is timezone-aware (UTC)
    if trades["timestamp"].dt.tz is None:
        trades["timestamp"] = trades["timestamp"].dt.tz_localize("UTC")

    # Load price data (dict keyed by lowercase asset)
    prices = {}
    for asset in ["eth", "btc", "hype"]:
        price_path = Path(f"./data/prices/{asset}_hourly_prices.parquet")
        if price_path.exists():
            df = pd.read_parquet(price_path)
            # Ensure timestamp is timezone-aware (UTC)
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
            prices[asset] = df

    return trades, prices


# ============================================================================
# Filtering
# ============================================================================


def filter_otm(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to out-of-the-money trades only."""
    return df[~df["is_itm"]].copy()


# ============================================================================
# Unusual Score Computation
# ============================================================================


def compute_unusual_score(trades_all: pd.DataFrame, trades_otm: pd.DataFrame) -> pd.DataFrame:
    """Compute unusual_score for OTM trades.

    Formula: (notional / trailing_7d_avg_notional) * (1 + abs(moneyness_pct) / 100)
    """
    trades_all = trades_all.sort_values("timestamp").copy()
    trades_otm = trades_otm.sort_values("timestamp").copy()

    # Compute rolling 7-day average notional per asset (on ALL trades)
    rolling_avgs = []
    for asset in trades_all["asset"].unique():
        asset_trades = trades_all[trades_all["asset"] == asset].copy()
        asset_trades = asset_trades.set_index("timestamp").sort_index()

        # Rolling 7-day mean
        asset_trades["trailing_7d_avg_notional"] = (
            asset_trades["notional_value"]
            .rolling("7D", min_periods=1)
            .mean()
        )
        asset_trades = asset_trades.reset_index()
        rolling_avgs.append(asset_trades[["message_id", "trailing_7d_avg_notional"]])

    rolling_df = pd.concat(rolling_avgs, ignore_index=True)

    # Merge rolling averages to OTM trades
    trades_otm = trades_otm.merge(rolling_df, on="message_id", how="left")

    # Calculate unusual_score
    trades_otm["unusual_score"] = (
        trades_otm["notional_value"] / trades_otm["trailing_7d_avg_notional"]
    ) * (1 + abs(trades_otm["moneyness_pct"]) / 100)

    return trades_otm


def flag_unusual(df: pd.DataFrame) -> pd.DataFrame:
    """Flag trades as unusual if score > mean + 2*std (per asset)."""
    df = df.copy()

    # Calculate thresholds per asset
    thresholds = df.groupby("asset")["unusual_score"].agg(["mean", "std"])
    thresholds["threshold"] = thresholds["mean"] + 2 * thresholds["std"]

    # Map thresholds back to dataframe
    df["unusual_threshold"] = df["asset"].map(thresholds["threshold"])
    df["is_unusual"] = df["unusual_score"] > df["unusual_threshold"]

    return df


# ============================================================================
# Forward Price Joins
# ============================================================================


def join_forward_prices(unusual_trades: pd.DataFrame, prices: dict) -> pd.DataFrame:
    """Join unusual trades with forward prices at t+24h and t+72h."""
    results = []

    for asset in unusual_trades["asset"].unique():
        asset_lower = asset.lower()
        if asset_lower not in prices:
            print(f"  Warning: No price data for {asset}")
            continue

        asset_trades = unusual_trades[unusual_trades["asset"] == asset]
        price_df = prices[asset_lower].sort_values("timestamp").reset_index(drop=True)

        for _, trade in asset_trades.iterrows():
            t = trade["timestamp"]

            # Find closest price timestamp (searchsorted finds insertion point)
            idx = price_df["timestamp"].searchsorted(t)

            # Clamp to valid range
            if idx >= len(price_df):
                idx = len(price_df) - 1

            # Get t+24h and t+72h prices
            idx_24h = idx + 24  # hourly data
            idx_72h = idx + 72

            if idx_72h < len(price_df):
                price_t0 = price_df.iloc[idx]["close"]
                price_24h = price_df.iloc[idx_24h]["close"]
                price_72h = price_df.iloc[idx_72h]["close"]

                return_24h = (price_24h - trade["ref_price"]) / trade["ref_price"]
                return_72h = (price_72h - trade["ref_price"]) / trade["ref_price"]

                trade_dict = trade.to_dict()
                trade_dict.update({
                    "price_t0": price_t0,
                    "price_24h": price_24h,
                    "price_72h": price_72h,
                    "return_24h": return_24h,
                    "return_72h": return_72h,
                    "abs_return_24h": abs(return_24h),
                    "abs_return_72h": abs(return_72h),
                })
                results.append(trade_dict)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


# ============================================================================
# Baseline Returns
# ============================================================================


def compute_baseline_returns(prices: dict, n_samples: int = 500) -> pd.DataFrame:
    """Compute baseline returns by sampling random timestamps from price data."""
    np.random.seed(42)  # Reproducibility
    baselines = []

    for asset, price_df in prices.items():
        price_df = price_df.sort_values("timestamp").reset_index(drop=True)

        # Sample indices that allow 72h forward lookup
        valid_indices = list(range(0, len(price_df) - 72))
        if not valid_indices:
            continue

        sample_size = min(n_samples, len(valid_indices))
        sample_indices = np.random.choice(valid_indices, size=sample_size, replace=False)

        for idx in sample_indices:
            ref_price = price_df.iloc[idx]["close"]
            price_24h = price_df.iloc[idx + 24]["close"]
            price_72h = price_df.iloc[idx + 72]["close"]

            return_24h = (price_24h - ref_price) / ref_price
            return_72h = (price_72h - ref_price) / ref_price

            baselines.append({
                "asset": asset.upper(),
                "ref_price": ref_price,
                "return_24h": return_24h,
                "return_72h": return_72h,
                "abs_return_24h": abs(return_24h),
                "abs_return_72h": abs(return_72h),
            })

    return pd.DataFrame(baselines)


# ============================================================================
# Statistical Analysis
# ============================================================================


def analyze_directional(unusual_with_returns: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Analyze directional returns for unusual trades vs baseline."""
    results = []

    for asset in unusual_with_returns["asset"].unique():
        asset_trades = unusual_with_returns[unusual_with_returns["asset"] == asset]
        asset_baseline = baseline[baseline["asset"] == asset]

        for instrument in asset_trades["instrument"].unique():
            inst_trades = asset_trades[asset_trades["instrument"] == instrument]

            for horizon in ["24h", "72h"]:
                col = f"return_{horizon}"
                returns = inst_trades[col].dropna()
                baseline_returns = asset_baseline[col].dropna()

                if len(returns) < 2:
                    continue

                mean_ret = returns.mean()
                std_ret = returns.std()

                # t-test vs zero
                t_stat_zero, p_val_zero = stats.ttest_1samp(returns, 0)

                # t-test vs baseline
                if len(baseline_returns) >= 2:
                    t_stat_base, p_val_base = stats.ttest_ind(returns, baseline_returns)
                else:
                    t_stat_base, p_val_base = np.nan, np.nan

                results.append({
                    "asset": asset,
                    "instrument": instrument,
                    "horizon": horizon,
                    "n_trades": len(returns),
                    "mean_return": mean_ret,
                    "std_return": std_ret,
                    "t_stat_vs_zero": t_stat_zero,
                    "p_value_vs_zero": p_val_zero,
                    "t_stat_vs_baseline": t_stat_base,
                    "p_value_vs_baseline": p_val_base,
                    "baseline_mean": baseline_returns.mean() if len(baseline_returns) > 0 else np.nan,
                    "baseline_n": len(baseline_returns),
                })

    return pd.DataFrame(results)


def analyze_volatility(unusual_with_returns: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Analyze absolute returns (volatility proxy) for unusual trades vs baseline."""
    results = []

    for asset in unusual_with_returns["asset"].unique():
        asset_trades = unusual_with_returns[unusual_with_returns["asset"] == asset]
        asset_baseline = baseline[baseline["asset"] == asset]

        for horizon in ["24h", "72h"]:
            col = f"abs_return_{horizon}"
            abs_returns = asset_trades[col].dropna()
            baseline_abs = asset_baseline[col].dropna()

            if len(abs_returns) < 2:
                continue

            mean_abs = abs_returns.mean()
            std_abs = abs_returns.std()

            # t-test vs baseline
            if len(baseline_abs) >= 2:
                t_stat, p_val = stats.ttest_ind(abs_returns, baseline_abs)
            else:
                t_stat, p_val = np.nan, np.nan

            results.append({
                "asset": asset,
                "horizon": horizon,
                "n_trades": len(abs_returns),
                "mean_abs_return": mean_abs,
                "std_abs_return": std_abs,
                "baseline_mean_abs": baseline_abs.mean() if len(baseline_abs) > 0 else np.nan,
                "t_stat_vs_baseline": t_stat,
                "p_value_vs_baseline": p_val,
            })

    return pd.DataFrame(results)


# ============================================================================
# Signal Consistency Analysis (Question 1)
# ============================================================================


def analyze_signal_consistency(unusual_with_returns: pd.DataFrame) -> dict:
    """Analyze if ETH call signal is consistent or driven by outliers.

    Returns dict with key metrics:
    - hit_rate: % of ETH calls with negative 72h return
    - mean_return: average 72h return
    - mean_excluding_top3: mean return after removing 3 largest magnitude trades
    - largest_trade_contribution: how much the largest trade affects the mean
    """
    eth_calls = unusual_with_returns[
        (unusual_with_returns["asset"] == "ETH") &
        (unusual_with_returns["instrument"] == "call")
    ].copy()

    if len(eth_calls) < 3:
        return {
            "hit_rate": np.nan,
            "n_trades": len(eth_calls),
            "mean_return": np.nan,
            "mean_excluding_top3": np.nan,
            "largest_trade_contribution": np.nan,
        }

    returns = eth_calls["return_72h"].dropna()
    n = len(returns)

    # Hit rate: % with negative return
    hit_rate = (returns < 0).sum() / n

    # Mean return
    mean_return = returns.mean()

    # Leave-one-out: remove largest magnitude trades
    sorted_by_magnitude = returns.abs().sort_values(ascending=False)
    top3_indices = sorted_by_magnitude.head(3).index
    returns_ex_top3 = returns.drop(top3_indices)
    mean_excluding_top3 = returns_ex_top3.mean() if len(returns_ex_top3) > 0 else np.nan

    # Largest trade contribution
    largest_idx = sorted_by_magnitude.head(1).index[0]
    mean_ex_largest = returns.drop(largest_idx).mean()
    largest_contribution = mean_return - mean_ex_largest

    return {
        "hit_rate": hit_rate,
        "n_trades": n,
        "mean_return": mean_return,
        "mean_excluding_top3": mean_excluding_top3,
        "largest_trade_contribution": largest_contribution,
    }


def plot_signal_consistency(unusual_with_returns: pd.DataFrame, prices: dict, output_path: Path):
    """Plot ETH call trade returns over time with price context."""
    eth_calls = unusual_with_returns[
        (unusual_with_returns["asset"] == "ETH") &
        (unusual_with_returns["instrument"] == "call")
    ].copy()

    if eth_calls.empty or "eth" not in prices:
        print("  Skipping signal consistency plot - insufficient ETH call data")
        return

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot ETH price on primary axis
    price_df = prices["eth"].sort_values("timestamp")
    ax1.plot(price_df["timestamp"], price_df["close"], color="steelblue",
             linewidth=1, alpha=0.7, label="ETH Price")
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("ETH Price (USD)", fontsize=12, color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    # Secondary axis for returns
    ax2 = ax1.twinx()

    # Scatter plot of 72h returns, sized by notional
    notional_scaled = eth_calls["notional_value"] / eth_calls["notional_value"].max() * 200 + 50
    colors = ["red" if r < 0 else "green" for r in eth_calls["return_72h"]]

    ax2.scatter(
        eth_calls["timestamp"],
        eth_calls["return_72h"] * 100,
        s=notional_scaled,
        c=colors,
        alpha=0.6,
        edgecolors="black",
        linewidths=0.5,
        zorder=5,
        label="72h Return"
    )

    ax2.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_ylabel("72h Forward Return (%)", fontsize=12, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    # Compute rolling 7-day hit rate
    eth_calls = eth_calls.sort_values("timestamp")
    eth_calls["is_negative"] = eth_calls["return_72h"] < 0
    eth_calls = eth_calls.set_index("timestamp")
    rolling_hit_rate = eth_calls["is_negative"].rolling("7D", min_periods=1).mean()
    eth_calls = eth_calls.reset_index()

    # Title with summary stats
    hit_rate = (eth_calls["return_72h"] < 0).mean()
    mean_ret = eth_calls["return_72h"].mean() * 100
    ax1.set_title(
        f"ETH Unusual Call Signal Consistency\n"
        f"Hit Rate (negative 72h return): {hit_rate:.1%} | Mean Return: {mean_ret:.2f}%",
        fontsize=14
    )

    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.xticks(rotation=45)

    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ============================================================================
# Volatility Timing Analysis (Question 2)
# ============================================================================


def compute_realized_vol(prices_df: pd.DataFrame, timestamp: pd.Timestamp,
                         window_hours: int, direction: str) -> float:
    """Compute realized volatility (std of hourly returns) before or after timestamp.

    Args:
        prices_df: DataFrame with timestamp and close columns (sorted)
        timestamp: Reference timestamp
        window_hours: Number of hours for the window
        direction: "trailing" (before) or "forward" (after)

    Returns:
        Annualized volatility (std of hourly returns * sqrt(24*365))
    """
    # Find the index closest to timestamp
    idx = prices_df["timestamp"].searchsorted(timestamp)
    if idx >= len(prices_df):
        idx = len(prices_df) - 1

    if direction == "trailing":
        start_idx = max(0, idx - window_hours)
        end_idx = idx
    else:  # forward
        start_idx = idx
        end_idx = min(len(prices_df) - 1, idx + window_hours)

    if end_idx - start_idx < 2:
        return np.nan

    window_prices = prices_df.iloc[start_idx:end_idx + 1]["close"].values
    hourly_returns = np.diff(window_prices) / window_prices[:-1]

    if len(hourly_returns) < 2:
        return np.nan

    # Return std of hourly returns (not annualized, for easier comparison)
    return np.std(hourly_returns)


def analyze_vol_timing(unusual_with_returns: pd.DataFrame, prices: dict) -> pd.DataFrame:
    """Compare trailing vs forward realized volatility for unusual trades.

    Returns DataFrame with:
    - trailing_vol_24h: realized vol in 24h before trade
    - forward_vol_24h: realized vol in 24h after trade
    - vol_diff: forward - trailing (positive = predictive)
    """
    results = []

    for _, trade in unusual_with_returns.iterrows():
        asset_lower = trade["asset"].lower()
        if asset_lower not in prices:
            continue

        price_df = prices[asset_lower].sort_values("timestamp").reset_index(drop=True)

        trailing_vol = compute_realized_vol(price_df, trade["timestamp"], 24, "trailing")
        forward_vol = compute_realized_vol(price_df, trade["timestamp"], 24, "forward")

        results.append({
            "message_id": trade["message_id"],
            "asset": trade["asset"],
            "instrument": trade["instrument"],
            "timestamp": trade["timestamp"],
            "trailing_vol_24h": trailing_vol,
            "forward_vol_24h": forward_vol,
            "vol_diff": forward_vol - trailing_vol if not (np.isnan(trailing_vol) or np.isnan(forward_vol)) else np.nan,
        })

    return pd.DataFrame(results)


def compute_baseline_vol_timing(prices: dict, n_samples: int = 500) -> pd.DataFrame:
    """Compute baseline vol timing by sampling random timestamps."""
    np.random.seed(42)
    results = []

    for asset, price_df in prices.items():
        price_df = price_df.sort_values("timestamp").reset_index(drop=True)

        # Sample indices that allow 24h lookback and forward
        valid_indices = list(range(24, len(price_df) - 24))
        if not valid_indices:
            continue

        sample_size = min(n_samples, len(valid_indices))
        sample_indices = np.random.choice(valid_indices, size=sample_size, replace=False)

        for idx in sample_indices:
            timestamp = price_df.iloc[idx]["timestamp"]
            trailing_vol = compute_realized_vol(price_df, timestamp, 24, "trailing")
            forward_vol = compute_realized_vol(price_df, timestamp, 24, "forward")

            results.append({
                "asset": asset.upper(),
                "trailing_vol_24h": trailing_vol,
                "forward_vol_24h": forward_vol,
                "vol_diff": forward_vol - trailing_vol if not (np.isnan(trailing_vol) or np.isnan(forward_vol)) else np.nan,
            })

    return pd.DataFrame(results)


def plot_vol_timing(vol_timing: pd.DataFrame, baseline_vol: pd.DataFrame, output_path: Path):
    """Plot trailing vs forward vol scatter with diagonal reference line."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Unusual trades
    ax1 = axes[0]
    vol_timing_clean = vol_timing.dropna(subset=["trailing_vol_24h", "forward_vol_24h"])

    if not vol_timing_clean.empty:
        colors = {"ETH": "blue", "BTC": "orange", "HYPE": "green"}
        for asset in vol_timing_clean["asset"].unique():
            asset_data = vol_timing_clean[vol_timing_clean["asset"] == asset]
            ax1.scatter(
                asset_data["trailing_vol_24h"] * 100,
                asset_data["forward_vol_24h"] * 100,
                c=colors.get(asset, "gray"),
                alpha=0.6,
                label=f"{asset} (n={len(asset_data)})",
                s=60
            )

        # Diagonal line
        max_val = max(vol_timing_clean["trailing_vol_24h"].max(),
                      vol_timing_clean["forward_vol_24h"].max()) * 100
        ax1.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="No change")

        # Stats
        predictive_frac = (vol_timing_clean["vol_diff"] > 0).mean()
        ax1.set_title(f"Unusual Trades Vol Timing\nPredictive: {predictive_frac:.1%} above diagonal")

    ax1.set_xlabel("Trailing 24h Volatility (%)", fontsize=11)
    ax1.set_ylabel("Forward 24h Volatility (%)", fontsize=11)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Right plot: Baseline comparison
    ax2 = axes[1]
    baseline_clean = baseline_vol.dropna(subset=["trailing_vol_24h", "forward_vol_24h"])

    if not baseline_clean.empty:
        ax2.scatter(
            baseline_clean["trailing_vol_24h"] * 100,
            baseline_clean["forward_vol_24h"] * 100,
            c="gray",
            alpha=0.2,
            s=30,
            label=f"Baseline (n={len(baseline_clean)})"
        )

        max_val = max(baseline_clean["trailing_vol_24h"].max(),
                      baseline_clean["forward_vol_24h"].max()) * 100
        ax2.plot([0, max_val], [0, max_val], "k--", alpha=0.5)

        baseline_predictive = (baseline_clean["vol_diff"] > 0).mean()
        ax2.set_title(f"Baseline Vol Timing\nPredictive: {baseline_predictive:.1%} above diagonal")

    ax2.set_xlabel("Trailing 24h Volatility (%)", fontsize=11)
    ax2.set_ylabel("Forward 24h Volatility (%)", fontsize=11)
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ============================================================================
# Selectivity Analysis (Question 3)
# ============================================================================


def analyze_selectivity(trades_otm: pd.DataFrame, unusual_trades: pd.DataFrame) -> pd.DataFrame:
    """Analyze signal frequency and threshold sensitivity.

    Returns DataFrame with:
    - Daily signal distribution stats
    - Threshold sensitivity analysis
    """
    # Add date column
    unusual_trades = unusual_trades.copy()
    unusual_trades["date"] = unusual_trades["timestamp"].dt.date

    # Daily distribution
    daily_counts = unusual_trades.groupby("date").size()

    # Get full date range from OTM trades
    all_dates = pd.date_range(
        start=trades_otm["timestamp"].min().date(),
        end=trades_otm["timestamp"].max().date(),
        freq="D"
    ).date
    daily_counts_full = daily_counts.reindex(all_dates, fill_value=0)

    results = [{
        "metric": "signals_per_day_mean",
        "value": daily_counts_full.mean(),
    }, {
        "metric": "signals_per_day_median",
        "value": daily_counts_full.median(),
    }, {
        "metric": "signals_per_day_max",
        "value": daily_counts_full.max(),
    }, {
        "metric": "days_with_zero_signals",
        "value": (daily_counts_full == 0).sum(),
    }, {
        "metric": "days_with_3plus_signals",
        "value": (daily_counts_full >= 3).sum(),
    }, {
        "metric": "total_days",
        "value": len(daily_counts_full),
    }, {
        "metric": "total_signals",
        "value": len(unusual_trades),
    }]

    return pd.DataFrame(results)


def analyze_threshold_sensitivity(trades_otm: pd.DataFrame, unusual_with_returns: pd.DataFrame) -> pd.DataFrame:
    """Analyze how different thresholds affect signal count and quality."""
    results = []

    # Calculate thresholds at different sigma levels
    thresholds_by_asset = trades_otm.groupby("asset")["unusual_score"].agg(["mean", "std"])

    for sigma in [1.5, 2.0, 2.5, 3.0]:
        thresholds_by_asset[f"threshold_{sigma}"] = (
            thresholds_by_asset["mean"] + sigma * thresholds_by_asset["std"]
        )

        # Count trades above threshold
        for asset in trades_otm["asset"].unique():
            threshold = thresholds_by_asset.loc[asset, f"threshold_{sigma}"]
            asset_trades = trades_otm[trades_otm["asset"] == asset]
            n_above = (asset_trades["unusual_score"] > threshold).sum()

            # Get mean return for these trades if we have return data
            if not unusual_with_returns.empty:
                asset_returns = unusual_with_returns[unusual_with_returns["asset"] == asset]
                above_threshold_ids = asset_trades[asset_trades["unusual_score"] > threshold]["message_id"]
                returns_above = asset_returns[asset_returns["message_id"].isin(above_threshold_ids)]
                mean_ret_72h = returns_above["return_72h"].mean() if not returns_above.empty else np.nan
            else:
                mean_ret_72h = np.nan

            results.append({
                "asset": asset,
                "sigma": sigma,
                "threshold": threshold,
                "n_signals": n_above,
                "mean_return_72h": mean_ret_72h,
            })

    return pd.DataFrame(results)


def plot_signal_frequency(unusual_trades: pd.DataFrame, trades_otm: pd.DataFrame, output_path: Path):
    """Plot distribution of signals per day."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Histogram of signals per day
    ax1 = axes[0]
    unusual_trades = unusual_trades.copy()
    unusual_trades["date"] = unusual_trades["timestamp"].dt.date
    daily_counts = unusual_trades.groupby("date").size()

    # Fill in zero days
    all_dates = pd.date_range(
        start=trades_otm["timestamp"].min().date(),
        end=trades_otm["timestamp"].max().date(),
        freq="D"
    ).date
    daily_counts_full = daily_counts.reindex(all_dates, fill_value=0)

    ax1.hist(daily_counts_full, bins=range(0, int(daily_counts_full.max()) + 2),
             edgecolor="black", alpha=0.7, color="steelblue")
    ax1.axvline(daily_counts_full.mean(), color="red", linestyle="--",
                label=f"Mean: {daily_counts_full.mean():.2f}")
    ax1.axvline(daily_counts_full.median(), color="orange", linestyle="--",
                label=f"Median: {daily_counts_full.median():.0f}")
    ax1.set_xlabel("Signals per Day", fontsize=11)
    ax1.set_ylabel("Number of Days", fontsize=11)
    ax1.set_title("Distribution of Unusual Signals per Day", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Time series of daily counts
    ax2 = axes[1]
    dates = pd.to_datetime(daily_counts_full.index)
    ax2.bar(dates, daily_counts_full.values, width=0.8, color="steelblue", alpha=0.7)
    ax2.axhline(daily_counts_full.mean(), color="red", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylabel("Number of Signals", fontsize=11)
    ax2.set_title("Unusual Signals Over Time", fontsize=12)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ============================================================================
# Tradeability Analysis (Sharpe Ratio with Clustering Correction)
# ============================================================================


def identify_trade_clusters(timestamps: pd.Series, hold_hours: int = 72) -> pd.Series:
    """Identify non-overlapping trade windows.

    Signals within hold_hours of each other belong to the same cluster.
    Returns a Series of cluster IDs aligned to the input timestamps.
    """
    timestamps = timestamps.sort_values().reset_index(drop=True)
    cluster_ids = []
    current_cluster = 0
    last_cluster_start = None

    for ts in timestamps:
        if last_cluster_start is None or (ts - last_cluster_start) >= timedelta(hours=hold_hours):
            current_cluster += 1
            last_cluster_start = ts
        cluster_ids.append(current_cluster)

    return pd.Series(cluster_ids, index=timestamps.index)


def bootstrap_sharpe_ci(returns: np.ndarray, trades_per_year: float,
                        n_bootstrap: int = 10000, ci: float = 0.95) -> tuple:
    """Bootstrap confidence interval for annualized Sharpe ratio.

    Args:
        returns: Array of per-trade returns
        trades_per_year: Estimated number of independent trades per year
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (default 95%)

    Returns:
        (lower_bound, upper_bound) for annualized Sharpe
    """
    np.random.seed(42)
    n = len(returns)
    if n < 2:
        return (np.nan, np.nan)

    sharpes = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(returns, size=n, replace=True)
        if sample.std() > 0:
            per_trade_sharpe = sample.mean() / sample.std()
            annual_sharpe = per_trade_sharpe * np.sqrt(trades_per_year)
            sharpes.append(annual_sharpe)

    if not sharpes:
        return (np.nan, np.nan)

    alpha = (1 - ci) / 2
    lower = np.percentile(sharpes, alpha * 100)
    upper = np.percentile(sharpes, (1 - alpha) * 100)
    return (lower, upper)


def analyze_tradeability(unusual_with_returns: pd.DataFrame, hold_hours: int = 72) -> dict:
    """Compute Sharpe ratio and tradeability metrics for ETH call contrarian strategy.

    Strategy: Short ETH when unusual call signal fires, hold for {hold_hours}h

    Returns dict with:
    - raw metrics (all trades)
    - corrected metrics (first signal per non-overlapping window only)
    - bootstrap confidence intervals
    - transaction cost sensitivity
    """
    # Filter to ETH calls only
    eth_calls = unusual_with_returns[
        (unusual_with_returns["asset"] == "ETH") &
        (unusual_with_returns["instrument"] == "call")
    ].copy()

    if len(eth_calls) < 3:
        return {
            "n_raw_signals": len(eth_calls),
            "n_windows": 0,
            "overlap_pct": np.nan,
            "raw_mean": np.nan,
            "raw_std": np.nan,
            "raw_sharpe_per_trade": np.nan,
            "raw_sharpe_annual": np.nan,
            "corrected_mean": np.nan,
            "corrected_std": np.nan,
            "corrected_sharpe_per_trade": np.nan,
            "corrected_sharpe_annual": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "trades_per_year_est": np.nan,
            "sharpe_5bps": np.nan,
            "sharpe_10bps": np.nan,
            "sharpe_20bps": np.nan,
            "sharpe_50bps": np.nan,
            "is_tradeable": False,
            "conclusion": "Insufficient data (< 3 signals)",
        }

    eth_calls = eth_calls.sort_values("timestamp").reset_index(drop=True)

    # Strategy returns: short ETH → return = -spot_return_72h
    eth_calls["strategy_return"] = -eth_calls["return_72h"]
    raw_returns = eth_calls["strategy_return"].dropna().values

    # --- Raw metrics (all signals, overcounts overlaps) ---
    raw_mean = raw_returns.mean()
    raw_std = raw_returns.std()
    raw_sharpe_per_trade = raw_mean / raw_std if raw_std > 0 else np.nan

    # --- Identify clusters and get first signal per cluster ---
    eth_calls["cluster_id"] = identify_trade_clusters(eth_calls["timestamp"], hold_hours)
    first_per_cluster = eth_calls.groupby("cluster_id").first().reset_index()
    corrected_returns = first_per_cluster["strategy_return"].dropna().values

    n_windows = len(corrected_returns)
    overlap_pct = (len(raw_returns) - n_windows) / len(raw_returns) * 100 if len(raw_returns) > 0 else 0

    # --- Corrected metrics ---
    corrected_mean = corrected_returns.mean() if n_windows > 0 else np.nan
    corrected_std = corrected_returns.std() if n_windows > 1 else np.nan
    corrected_sharpe_per_trade = corrected_mean / corrected_std if corrected_std and corrected_std > 0 else np.nan

    # --- Annualization ---
    # Estimate trades per year based on observation period
    date_range = (eth_calls["timestamp"].max() - eth_calls["timestamp"].min()).days
    if date_range > 0 and n_windows > 0:
        windows_per_day = n_windows / date_range
        trades_per_year = windows_per_day * 365
    else:
        trades_per_year = 0

    raw_sharpe_annual = raw_sharpe_per_trade * np.sqrt(len(raw_returns) / date_range * 365) if date_range > 0 and not np.isnan(raw_sharpe_per_trade) else np.nan
    corrected_sharpe_annual = corrected_sharpe_per_trade * np.sqrt(trades_per_year) if trades_per_year > 0 and not np.isnan(corrected_sharpe_per_trade) else np.nan

    # --- Bootstrap CI ---
    ci_low, ci_high = bootstrap_sharpe_ci(corrected_returns, trades_per_year) if n_windows >= 3 else (np.nan, np.nan)

    # --- Transaction cost sensitivity ---
    cost_levels = [0.0005, 0.001, 0.002, 0.005]  # 5, 10, 20, 50 bps round-trip
    cost_sharpes = {}
    for cost in cost_levels:
        net_returns = corrected_returns - cost
        if corrected_std and corrected_std > 0:
            net_sharpe = (net_returns.mean() / corrected_std) * np.sqrt(trades_per_year) if trades_per_year > 0 else np.nan
        else:
            net_sharpe = np.nan
        cost_sharpes[f"sharpe_{int(cost*10000)}bps"] = net_sharpe

    # --- Tradeability conclusion ---
    # Signal is tradeable if 95% CI lower bound > 0
    is_tradeable = ci_low > 0 if not np.isnan(ci_low) else False
    if np.isnan(ci_low):
        conclusion = "Insufficient data for confidence interval"
    elif is_tradeable:
        conclusion = f"Signal is tradeable (95% CI lower bound {ci_low:.2f} > 0)"
    else:
        conclusion = f"Signal not tradeable with confidence (95% CI includes zero: [{ci_low:.2f}, {ci_high:.2f}])"

    return {
        "n_raw_signals": len(raw_returns),
        "n_windows": n_windows,
        "overlap_pct": overlap_pct,
        "date_range_days": date_range,
        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "raw_sharpe_per_trade": raw_sharpe_per_trade,
        "raw_sharpe_annual": raw_sharpe_annual,
        "corrected_mean": corrected_mean,
        "corrected_std": corrected_std,
        "corrected_sharpe_per_trade": corrected_sharpe_per_trade,
        "corrected_sharpe_annual": corrected_sharpe_annual,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "trades_per_year_est": trades_per_year,
        **cost_sharpes,
        "is_tradeable": is_tradeable,
        "conclusion": conclusion,
    }


def print_tradeability_analysis(tradeability: dict):
    """Print formatted tradeability analysis to console."""
    print("\n" + "=" * 60)
    print("TRADEABILITY ANALYSIS: Short ETH on Unusual Call Signal")
    print("=" * 60)

    print(f"\nSignal clustering: {tradeability['overlap_pct']:.0f}% overlap within 72h")
    print(f"Independent windows: {tradeability['n_windows']} (of {tradeability['n_raw_signals']} raw signals)")
    print(f"Observation period: {tradeability['date_range_days']} days")
    print(f"Estimated windows/year: {tradeability['trades_per_year_est']:.0f}")

    print("\nRaw metrics (overcounts overlaps):")
    print(f"  Mean: {tradeability['raw_mean']*100:.2f}%, Std: {tradeability['raw_std']*100:.2f}%, Sharpe: {tradeability['raw_sharpe_per_trade']:.2f}")

    print("\nCorrected metrics (independent windows only):")
    print(f"  Mean: {tradeability['corrected_mean']*100:.2f}%, Std: {tradeability['corrected_std']*100:.2f}%, Sharpe: {tradeability['corrected_sharpe_per_trade']:.2f}")
    print(f"  Annualized Sharpe: {tradeability['corrected_sharpe_annual']:.2f}")
    print(f"  Bootstrap 95% CI: [{tradeability['ci_low']:.2f}, {tradeability['ci_high']:.2f}]")

    print("\nTransaction costs (round-trip):")
    print(f"   5 bps: Sharpe {tradeability.get('sharpe_5bps', np.nan):.2f}")
    print(f"  10 bps: Sharpe {tradeability.get('sharpe_10bps', np.nan):.2f}")
    print(f"  20 bps: Sharpe {tradeability.get('sharpe_20bps', np.nan):.2f}")
    print(f"  50 bps: Sharpe {tradeability.get('sharpe_50bps', np.nan):.2f}")

    print(f"\nCONCLUSION: {tradeability['conclusion']}")


# ============================================================================
# DTE Bucketing
# ============================================================================


def bucket_dte(df: pd.DataFrame) -> pd.DataFrame:
    """Add DTE bucket column."""
    df = df.copy()
    conditions = [
        df["dte"] <= 7,
        (df["dte"] > 7) & (df["dte"] <= 30),
        df["dte"] > 30
    ]
    labels = ["0-7 DTE", "7-30 DTE", "30+ DTE"]
    df["dte_bucket"] = np.select(conditions, labels, default="Unknown")
    return df


def analyze_by_dte_bucket(unusual_with_returns: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Analyze directional returns grouped by DTE bucket."""
    unusual_with_returns = bucket_dte(unusual_with_returns)
    results = []

    for asset in unusual_with_returns["asset"].unique():
        asset_trades = unusual_with_returns[unusual_with_returns["asset"] == asset]
        asset_baseline = baseline[baseline["asset"] == asset]

        for dte_bucket in ["0-7 DTE", "7-30 DTE", "30+ DTE"]:
            bucket_trades = asset_trades[asset_trades["dte_bucket"] == dte_bucket]

            for instrument in bucket_trades["instrument"].unique():
                inst_trades = bucket_trades[bucket_trades["instrument"] == instrument]

                for horizon in ["24h", "72h"]:
                    col = f"return_{horizon}"
                    returns = inst_trades[col].dropna()
                    baseline_returns = asset_baseline[col].dropna()

                    if len(returns) < 2:
                        continue

                    mean_ret = returns.mean()
                    std_ret = returns.std()

                    # t-test vs zero
                    t_stat_zero, p_val_zero = stats.ttest_1samp(returns, 0)

                    results.append({
                        "asset": asset,
                        "dte_bucket": dte_bucket,
                        "instrument": instrument,
                        "horizon": horizon,
                        "n_trades": len(returns),
                        "mean_return": mean_ret,
                        "std_return": std_ret,
                        "t_stat_vs_zero": t_stat_zero,
                        "p_value_vs_zero": p_val_zero,
                    })

    return pd.DataFrame(results)


# ============================================================================
# Visualization
# ============================================================================


def plot_unusual_flow_chart(asset: str, prices: dict, unusual_trades: pd.DataFrame, output_path: Path):
    """Plot price chart with unusual flow events marked."""
    asset_lower = asset.lower()
    if asset_lower not in prices:
        print(f"  Skipping chart for {asset} - no price data")
        return

    price_df = prices[asset_lower].sort_values("timestamp")
    asset_unusual = unusual_trades[unusual_trades["asset"] == asset]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot price line
    ax.plot(price_df["timestamp"], price_df["close"], color="steelblue", linewidth=1, label=f"{asset} Price")

    # Mark unusual calls (green triangles up)
    calls = asset_unusual[asset_unusual["instrument"] == "call"]
    if not calls.empty:
        ax.scatter(
            calls["timestamp"],
            calls["ref_price"],
            marker="^",
            color="green",
            s=80,
            alpha=0.7,
            label=f"Unusual Calls ({len(calls)})",
            zorder=5
        )

    # Mark unusual puts (red triangles down)
    puts = asset_unusual[asset_unusual["instrument"] == "put"]
    if not puts.empty:
        ax.scatter(
            puts["timestamp"],
            puts["ref_price"],
            marker="v",
            color="red",
            s=80,
            alpha=0.7,
            label=f"Unusual Puts ({len(puts)})",
            zorder=5
        )

    ax.set_title(f"{asset} Spot Price with Unusual Options Flow", fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (USD)", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_return_distributions(unusual_with_returns: pd.DataFrame, baseline: pd.DataFrame, output_path: Path):
    """Plot histogram of 24h returns: calls vs puts vs baseline."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    assets = unusual_with_returns["asset"].unique()

    for i, asset in enumerate(assets[:3]):  # Max 3 assets
        ax = axes[i] if len(assets) > 1 else axes

        asset_trades = unusual_with_returns[unusual_with_returns["asset"] == asset]
        asset_baseline = baseline[baseline["asset"] == asset]

        calls = asset_trades[asset_trades["instrument"] == "call"]["return_24h"].dropna()
        puts = asset_trades[asset_trades["instrument"] == "put"]["return_24h"].dropna()
        base = asset_baseline["return_24h"].dropna()

        bins = np.linspace(-0.15, 0.15, 31)

        if not calls.empty:
            ax.hist(calls, bins=bins, alpha=0.5, color="green", label=f"Calls (n={len(calls)})")
        if not puts.empty:
            ax.hist(puts, bins=bins, alpha=0.5, color="red", label=f"Puts (n={len(puts)})")
        if not base.empty:
            ax.hist(base, bins=bins, alpha=0.3, color="gray", label=f"Baseline (n={len(base)})")

        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{asset} 24h Return Distribution", fontsize=12)
        ax.set_xlabel("Return (%)", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Handle case with fewer than 3 assets
    for j in range(len(assets), 3):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ============================================================================
# Summary Output
# ============================================================================


def generate_summary(unusual_trades: pd.DataFrame, directional_results: pd.DataFrame, volatility_results: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics of unusual trades."""
    unusual_trades = bucket_dte(unusual_trades)

    summary = (
        unusual_trades.groupby(["asset", "instrument", "dte_bucket"])
        .agg(
            count=("message_id", "count"),
            total_notional=("notional_value", "sum"),
            avg_unusual_score=("unusual_score", "mean"),
            avg_moneyness_pct=("moneyness_pct", lambda x: abs(x).mean()),
        )
        .reset_index()
    )

    return summary


def print_key_findings(directional_results: pd.DataFrame, volatility_results: pd.DataFrame):
    """Print key statistical findings to console."""
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    # Significant directional results (p < 0.05)
    if not directional_results.empty:
        sig_directional = directional_results[directional_results["p_value_vs_zero"] < 0.05]
        if not sig_directional.empty:
            print("\nStatistically Significant Directional Results (p < 0.05):")
            for _, row in sig_directional.iterrows():
                direction = "+" if row["mean_return"] > 0 else ""
                print(f"  {row['asset']} {row['instrument']} {row['horizon']}: "
                      f"{direction}{row['mean_return']*100:.2f}% (p={row['p_value_vs_zero']:.4f}, n={row['n_trades']})")
        else:
            print("\nNo statistically significant directional results found.")

    # Volatility findings
    if not volatility_results.empty:
        sig_vol = volatility_results[volatility_results["p_value_vs_baseline"] < 0.05]
        if not sig_vol.empty:
            print("\nStatistically Significant Volatility Results (p < 0.05):")
            for _, row in sig_vol.iterrows():
                diff = row["mean_abs_return"] - row["baseline_mean_abs"]
                direction = "higher" if diff > 0 else "lower"
                print(f"  {row['asset']} {row['horizon']}: "
                      f"{row['mean_abs_return']*100:.2f}% abs return "
                      f"({direction} than baseline {row['baseline_mean_abs']*100:.2f}%, p={row['p_value_vs_baseline']:.4f})")


# ============================================================================
# Main Pipeline
# ============================================================================


async def main():
    print("=" * 60)
    print("UNUSUAL OPTIONS FLOW ANALYSIS PIPELINE")
    print("=" * 60)

    # Create output directory
    output_dir = Path("./outputs")
    output_dir.mkdir(exist_ok=True)

    # Step 1: Load data
    print("\n[1/13] Loading data...")
    trades, prices = await load_data()
    print(f"  Loaded {len(trades)} trades")
    print(f"  Price data: {list(prices.keys())}")
    for asset, pdf in prices.items():
        print(f"    {asset.upper()}: {len(pdf)} hourly candles ({pdf['timestamp'].min()} to {pdf['timestamp'].max()})")

    if trades.empty:
        print("ERROR: No trades loaded. Exiting.")
        return

    # Step 2: Filter to OTM trades
    print("\n[2/13] Filtering to OTM trades...")
    trades_otm = filter_otm(trades)
    print(f"  {len(trades_otm)} OTM trades (of {len(trades)} total)")

    if trades_otm.empty:
        print("ERROR: No OTM trades found. Exiting.")
        return

    # Step 3: Compute unusual score
    print("\n[3/13] Computing unusual scores...")
    trades_otm = compute_unusual_score(trades, trades_otm)
    print(f"  Unusual score range: {trades_otm['unusual_score'].min():.2f} - {trades_otm['unusual_score'].max():.2f}")

    # Step 4: Flag unusual trades
    print("\n[4/13] Flagging unusual trades...")
    trades_otm = flag_unusual(trades_otm)
    unusual_trades = trades_otm[trades_otm["is_unusual"]].copy()
    print(f"  {len(unusual_trades)} unusual trades identified")

    for asset in unusual_trades["asset"].unique():
        asset_unusual = unusual_trades[unusual_trades["asset"] == asset]
        calls = len(asset_unusual[asset_unusual["instrument"] == "call"])
        puts = len(asset_unusual[asset_unusual["instrument"] == "put"])
        print(f"    {asset}: {calls} calls, {puts} puts")

    if unusual_trades.empty:
        print("WARNING: No unusual trades identified. Continuing with analysis...")

    # Step 5: Join with forward prices
    print("\n[5/13] Joining with forward prices...")
    unusual_with_returns = join_forward_prices(unusual_trades, prices)
    print(f"  {len(unusual_with_returns)} trades with complete forward price data")

    # Step 6: Compute baseline returns
    print("\n[6/13] Computing baseline returns...")
    baseline = compute_baseline_returns(prices, n_samples=500)
    print(f"  {len(baseline)} baseline samples generated")

    # Step 7: Statistical analysis
    print("\n[7/13] Running statistical analysis...")

    if not unusual_with_returns.empty:
        directional_results = analyze_directional(unusual_with_returns, baseline)
        volatility_results = analyze_volatility(unusual_with_returns, baseline)
        dte_results = analyze_by_dte_bucket(unusual_with_returns, baseline)
        print(f"  Directional results: {len(directional_results)} rows")
        print(f"  Volatility results: {len(volatility_results)} rows")
        print(f"  DTE bucket results: {len(dte_results)} rows")
    else:
        directional_results = pd.DataFrame()
        volatility_results = pd.DataFrame()
        dte_results = pd.DataFrame()
        print("  Skipped - no unusual trades with returns")

    # Step 8: Signal consistency analysis
    print("\n[8/13] Analyzing signal consistency...")
    if not unusual_with_returns.empty:
        consistency_metrics = analyze_signal_consistency(unusual_with_returns)
        print(f"  ETH call hit rate (negative 72h return): {consistency_metrics['hit_rate']:.1%}")
        print(f"  Mean 72h return: {consistency_metrics['mean_return']*100:.2f}%")
        print(f"  Mean excluding top 3 trades: {consistency_metrics['mean_excluding_top3']*100:.2f}%")
        print(f"  Largest trade contribution to mean: {consistency_metrics['largest_trade_contribution']*100:.2f}%")

        plot_signal_consistency(unusual_with_returns, prices, output_dir / "eth_signal_consistency.png")
        print("  Saved: eth_signal_consistency.png")
    else:
        consistency_metrics = {}
        print("  Skipped - no unusual trades with returns")

    # Step 9: Volatility timing analysis
    print("\n[9/13] Analyzing volatility timing...")
    if not unusual_with_returns.empty:
        vol_timing = analyze_vol_timing(unusual_with_returns, prices)
        baseline_vol = compute_baseline_vol_timing(prices, n_samples=500)

        # Calculate summary stats
        vol_timing_clean = vol_timing.dropna(subset=["vol_diff"])
        if not vol_timing_clean.empty:
            mean_trailing = vol_timing_clean["trailing_vol_24h"].mean()
            mean_forward = vol_timing_clean["forward_vol_24h"].mean()
            predictive_frac = (vol_timing_clean["vol_diff"] > 0).mean()

            print(f"  Mean trailing vol (24h): {mean_trailing*100:.3f}%")
            print(f"  Mean forward vol (24h): {mean_forward*100:.3f}%")
            print(f"  Predictive fraction (forward > trailing): {predictive_frac:.1%}")

            # Paired t-test
            t_stat, p_val = stats.ttest_rel(
                vol_timing_clean["forward_vol_24h"],
                vol_timing_clean["trailing_vol_24h"]
            )
            print(f"  Paired t-test (forward vs trailing): t={t_stat:.2f}, p={p_val:.4f}")

            if p_val < 0.05:
                if mean_forward > mean_trailing:
                    print("  Conclusion: Signal is PREDICTIVE (forward vol > trailing vol, p<0.05)")
                else:
                    print("  Conclusion: Signal is REACTIVE (trailing vol > forward vol, p<0.05)")
            else:
                print("  Conclusion: No significant difference (p>=0.05)")

        vol_timing.to_csv(output_dir / "vol_timing_results.csv", index=False)
        print("  Saved: vol_timing_results.csv")

        plot_vol_timing(vol_timing, baseline_vol, output_dir / "vol_timing_scatter.png")
        print("  Saved: vol_timing_scatter.png")
    else:
        vol_timing = pd.DataFrame()
        print("  Skipped - no unusual trades with returns")

    # Step 10: Selectivity analysis
    print("\n[10/13] Analyzing selectivity...")
    if not unusual_trades.empty:
        selectivity = analyze_selectivity(trades_otm, unusual_trades)
        selectivity.to_csv(output_dir / "selectivity_analysis.csv", index=False)
        print("  Saved: selectivity_analysis.csv")

        # Print key metrics
        for _, row in selectivity.iterrows():
            print(f"  {row['metric']}: {row['value']:.2f}")

        # Threshold sensitivity
        threshold_sens = analyze_threshold_sensitivity(trades_otm, unusual_with_returns)
        threshold_sens.to_csv(output_dir / "threshold_sensitivity.csv", index=False)
        print("  Saved: threshold_sensitivity.csv")

        plot_signal_frequency(unusual_trades, trades_otm, output_dir / "signal_frequency.png")
        print("  Saved: signal_frequency.png")
    else:
        print("  Skipped - no unusual trades")

    # Step 11: Tradeability analysis
    print("\n[11/13] Analyzing tradeability...")
    if not unusual_with_returns.empty:
        tradeability = analyze_tradeability(unusual_with_returns)
        print_tradeability_analysis(tradeability)

        # Save detailed results
        tradeability_df = pd.DataFrame([tradeability])
        tradeability_df.to_csv(output_dir / "tradeability_analysis.csv", index=False)
        print(f"\n  Saved: tradeability_analysis.csv")
    else:
        print("  Skipped - no unusual trades with returns")

    # Step 12: Generate outputs
    print("\n[12/13] Generating output files...")

    # Summary CSV
    if not unusual_trades.empty:
        summary = generate_summary(unusual_trades, directional_results, volatility_results)
        summary.to_csv(output_dir / "unusual_trade_summary.csv", index=False)
        print(f"  Saved: unusual_trade_summary.csv ({len(summary)} rows)")
    else:
        pd.DataFrame().to_csv(output_dir / "unusual_trade_summary.csv", index=False)
        print("  Saved: unusual_trade_summary.csv (empty)")

    # Directional results CSV
    if not directional_results.empty:
        directional_results.to_csv(output_dir / "directional_results.csv", index=False)
        print(f"  Saved: directional_results.csv ({len(directional_results)} rows)")
    else:
        pd.DataFrame().to_csv(output_dir / "directional_results.csv", index=False)
        print("  Saved: directional_results.csv (empty)")

    # Volatility results CSV
    if not volatility_results.empty:
        volatility_results.to_csv(output_dir / "volatility_results.csv", index=False)
        print(f"  Saved: volatility_results.csv ({len(volatility_results)} rows)")
    else:
        pd.DataFrame().to_csv(output_dir / "volatility_results.csv", index=False)
        print("  Saved: volatility_results.csv (empty)")

    # Step 13: Generate charts
    print("\n[13/13] Generating charts...")

    for asset in ["ETH", "BTC", "HYPE"]:
        output_path = output_dir / f"unusual_flow_{asset.lower()}.png"
        plot_unusual_flow_chart(asset, prices, unusual_trades, output_path)
        print(f"  Saved: unusual_flow_{asset.lower()}.png")

    if not unusual_with_returns.empty and not baseline.empty:
        plot_return_distributions(unusual_with_returns, baseline, output_dir / "return_distributions.png")
        print("  Saved: return_distributions.png")
    else:
        print("  Skipped: return_distributions.png (insufficient data)")

    # Print key findings
    print_key_findings(directional_results, volatility_results)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    asyncio.run(main())
