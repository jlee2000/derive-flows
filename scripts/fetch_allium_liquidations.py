#!/usr/bin/env python3
"""Fetch liquidation events from Allium API.

This script fetches liquidation data from Allium's Hyperliquid dex.trades table
for use in liquidation cascade analysis.

Requires ALLIUM_API_KEY environment variable to be set.
"""

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ALLIUM_API_URL = "https://api.allium.so/api/v1/explorer/queries/run"
ALLIUM_API_KEY = os.getenv("ALLIUM_API_KEY")

ASSETS = ["ETH", "BTC"]

START_DATE = datetime(2025, 12, 4, 0, 0, tzinfo=timezone.utc)
END_DATE = datetime(2026, 2, 3, 0, 0, tzinfo=timezone.utc)

LIQUIDATIONS_QUERY = """
SELECT
    block_timestamp as timestamp,
    coin,
    side,
    size_usd as usd_amount,
    price as execution_price,
    liquidated_user,
    liquidation_mark_price
FROM hyperliquid.dex.trades
WHERE coin IN ({coins})
  AND block_timestamp >= '{start_date}'
  AND block_timestamp < '{end_date}'
  AND liquidated_user IS NOT NULL
ORDER BY block_timestamp, coin
"""

AGGRESSOR_FLOW_QUERY = """
SELECT
    DATE_TRUNC('hour', block_timestamp) as hour,
    coin,
    side,
    SUM(size_usd) as total_usd,
    COUNT(*) as trade_count
FROM hyperliquid.dex.trades
WHERE coin IN ({coins})
  AND block_timestamp >= '{start_date}'
  AND block_timestamp < '{end_date}'
  AND liquidated_user IS NULL
GROUP BY DATE_TRUNC('hour', block_timestamp), coin, side
ORDER BY hour, coin, side
"""


async def fetch_query(session: aiohttp.ClientSession, query: str) -> list[dict]:
    """Execute a query against Allium API and return results."""
    if not ALLIUM_API_KEY:
        raise ValueError("ALLIUM_API_KEY environment variable not set")

    headers = {
        "X-API-Key": ALLIUM_API_KEY,
        "Content-Type": "application/json",
    }

    payload = {
        "query": query,
        "parameters": {},
    }

    async with session.post(ALLIUM_API_URL, json=payload, headers=headers) as resp:
        if resp.status == 429:
            print("  Rate limited, waiting 30s...")
            await asyncio.sleep(30)
            return await fetch_query(session, query)
        if resp.status != 200:
            error_text = await resp.text()
            raise RuntimeError(f"Allium API error {resp.status}: {error_text}")

        result = await resp.json()
        return result.get("data", [])


async def fetch_liquidations() -> pd.DataFrame:
    """Fetch liquidation events for all assets."""
    coins_str = ", ".join(f"'{asset}'" for asset in ASSETS)
    query = LIQUIDATIONS_QUERY.format(
        coins=coins_str,
        start_date=START_DATE.strftime("%Y-%m-%d %H:%M:%S"),
        end_date=END_DATE.strftime("%Y-%m-%d %H:%M:%S"),
    )

    print(f"Fetching liquidations for {ASSETS}...")
    print(f"  Date range: {START_DATE} to {END_DATE}")

    async with aiohttp.ClientSession() as session:
        rows = await fetch_query(session, query)

    if not rows:
        print("  Warning: No liquidation data returned")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Convert types
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for col in ["usd_amount", "execution_price", "liquidation_mark_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


async def fetch_aggressor_flow() -> pd.DataFrame:
    """Fetch hourly aggressor flow data for all assets."""
    coins_str = ", ".join(f"'{asset}'" for asset in ASSETS)
    query = AGGRESSOR_FLOW_QUERY.format(
        coins=coins_str,
        start_date=START_DATE.strftime("%Y-%m-%d %H:%M:%S"),
        end_date=END_DATE.strftime("%Y-%m-%d %H:%M:%S"),
    )

    print(f"\nFetching aggressor flow for {ASSETS}...")

    async with aiohttp.ClientSession() as session:
        rows = await fetch_query(session, query)

    if not rows:
        print("  Warning: No aggressor flow data returned")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Convert types
    df["hour"] = pd.to_datetime(df["hour"], utc=True)
    df["total_usd"] = pd.to_numeric(df["total_usd"], errors="coerce")
    df["trade_count"] = pd.to_numeric(df["trade_count"], errors="coerce").astype(int)

    # Pivot to get buy/sell columns
    pivot = df.pivot_table(
        index=["hour", "coin"],
        columns="side",
        values=["total_usd", "trade_count"],
        fill_value=0,
    ).reset_index()

    # Flatten column names
    pivot.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in pivot.columns
    ]

    # Rename for clarity
    rename_map = {
        "total_usd_buy": "buy_volume_usd",
        "total_usd_sell": "sell_volume_usd",
        "trade_count_buy": "buy_count",
        "trade_count_sell": "sell_count",
    }
    pivot = pivot.rename(columns=rename_map)

    # Compute net flow
    if "buy_volume_usd" in pivot.columns and "sell_volume_usd" in pivot.columns:
        pivot["net_flow_usd"] = pivot["buy_volume_usd"] - pivot["sell_volume_usd"]

    pivot = pivot.sort_values(["hour", "coin"]).reset_index(drop=True)

    return pivot


def print_liquidation_summary(df: pd.DataFrame):
    """Print liquidation summary statistics."""
    print(f"\nLiquidation Summary:")
    print(f"  Total liquidations: {len(df)}")

    if df.empty:
        return

    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Total liquidation volume: ${df['usd_amount'].sum():,.0f}")

    for coin in df["coin"].unique():
        coin_df = df[df["coin"] == coin]
        longs = coin_df[coin_df["side"] == "sell"]  # Long liquidation = forced sell
        shorts = coin_df[coin_df["side"] == "buy"]  # Short liquidation = forced buy
        print(f"\n  {coin}:")
        print(f"    Long liquidations: {len(longs)} (${longs['usd_amount'].sum():,.0f})")
        print(f"    Short liquidations: {len(shorts)} (${shorts['usd_amount'].sum():,.0f})")


def print_aggressor_summary(df: pd.DataFrame):
    """Print aggressor flow summary statistics."""
    print(f"\nAggressor Flow Summary:")
    print(f"  Total hourly records: {len(df)}")

    if df.empty:
        return

    for coin in df["coin"].unique():
        coin_df = df[df["coin"] == coin]
        print(f"\n  {coin}:")
        print(f"    Hours: {len(coin_df)}")
        if "net_flow_usd" in coin_df.columns:
            print(f"    Net flow range: ${coin_df['net_flow_usd'].min():,.0f} to ${coin_df['net_flow_usd'].max():,.0f}")
            print(f"    Mean net flow: ${coin_df['net_flow_usd'].mean():,.0f}")


async def main():
    output_dir = Path("data/allium")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch liquidations
    liquidations = await fetch_liquidations()

    if not liquidations.empty:
        output_path = output_dir / "liquidations.parquet"
        liquidations.to_parquet(output_path, index=False)
        print(f"\nSaved liquidations to {output_path}")
        print_liquidation_summary(liquidations)
    else:
        print("No liquidation data fetched.")

    # Brief pause between queries
    await asyncio.sleep(1)

    # Fetch aggressor flow
    aggressor_flow = await fetch_aggressor_flow()

    if not aggressor_flow.empty:
        output_path = output_dir / "aggressor_flow.parquet"
        aggressor_flow.to_parquet(output_path, index=False)
        print(f"\nSaved aggressor flow to {output_path}")
        print_aggressor_summary(aggressor_flow)
    else:
        print("No aggressor flow data fetched.")


if __name__ == "__main__":
    asyncio.run(main())
