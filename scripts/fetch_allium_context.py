#!/usr/bin/env python3
"""Fetch market context data (funding, OI, premium) from Allium API.

This script fetches hourly market context data from Allium's Hyperliquid tables
for use in conditional signal analysis.

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

MARKET_CONTEXT_QUERY = """
SELECT
    block_timestamp as timestamp,
    coin,
    funding_rate as funding,
    premium,
    open_interest,
    mark_price
FROM hyperliquid.raw.perpetual_market_asset_contexts
WHERE coin IN ({coins})
  AND block_timestamp >= '{start_date}'
  AND block_timestamp < '{end_date}'
ORDER BY block_timestamp, coin
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


async def fetch_market_context() -> pd.DataFrame:
    """Fetch market context data for all assets."""
    coins_str = ", ".join(f"'{asset}'" for asset in ASSETS)
    query = MARKET_CONTEXT_QUERY.format(
        coins=coins_str,
        start_date=START_DATE.strftime("%Y-%m-%d %H:%M:%S"),
        end_date=END_DATE.strftime("%Y-%m-%d %H:%M:%S"),
    )

    print(f"Fetching market context for {ASSETS}...")
    print(f"  Date range: {START_DATE} to {END_DATE}")

    async with aiohttp.ClientSession() as session:
        rows = await fetch_query(session, query)

    if not rows:
        print("  Warning: No data returned")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Convert types
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for col in ["funding", "premium", "open_interest", "mark_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["timestamp", "coin"]).reset_index(drop=True)

    return df


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print(f"\nMarket Context Summary:")
    print(f"  Total rows: {len(df)}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    for coin in df["coin"].unique():
        coin_df = df[df["coin"] == coin]
        print(f"\n  {coin}:")
        print(f"    Rows: {len(coin_df)}")
        print(f"    Funding range: {coin_df['funding'].min():.6f} to {coin_df['funding'].max():.6f}")
        print(f"    OI range: ${coin_df['open_interest'].min():,.0f} to ${coin_df['open_interest'].max():,.0f}")


async def main():
    output_dir = Path("data/allium")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = await fetch_market_context()

    if df.empty:
        print("No data fetched. Check API key and query.")
        return

    output_path = output_dir / "market_context.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")

    print_summary(df)


if __name__ == "__main__":
    asyncio.run(main())
