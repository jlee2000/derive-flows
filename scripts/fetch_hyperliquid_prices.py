#!/usr/bin/env python3
"""Fetch historical hourly prices from Hyperliquid API."""

import asyncio
import aiohttp
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

API_URL = "https://api.hyperliquid.xyz/info"

ASSETS = {
    "eth": "ETH",      # Perpetual
    "btc": "BTC",      # Perpetual
    "hype": "@107",    # Spot (mainnet index)
}

START_DATE = datetime(2025, 12, 4, 0, 0, tzinfo=timezone.utc)
END_DATE = datetime(2026, 2, 3, 0, 0, tzinfo=timezone.utc)  # Jan 31 + 72h


async def fetch_candles(session: aiohttp.ClientSession, coin: str, start_ms: int, end_ms: int) -> list:
    """Fetch candlestick data from Hyperliquid."""
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": "1h",
            "startTime": start_ms,
            "endTime": end_ms
        }
    }
    async with session.post(API_URL, json=payload) as resp:
        if resp.status == 429:  # Rate limited
            print("  Rate limited, waiting 5s...")
            await asyncio.sleep(5)
            return await fetch_candles(session, coin, start_ms, end_ms)
        resp.raise_for_status()
        return await resp.json()


async def fetch_asset_prices(name: str, coin: str) -> pd.DataFrame:
    """Fetch all hourly candles for an asset."""
    start_ms = int(START_DATE.timestamp() * 1000)
    end_ms = int(END_DATE.timestamp() * 1000)

    async with aiohttp.ClientSession() as session:
        candles = await fetch_candles(session, coin, start_ms, end_ms)

    if not candles:
        print(f"  Warning: No data returned for {name}")
        return pd.DataFrame()

    df = pd.DataFrame(candles)
    df = df.rename(columns={
        "t": "timestamp_ms",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "n": "trades"
    })

    # Convert types
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Select and order columns
    df = df[["timestamp", "open", "high", "low", "close", "volume", "trades"]]
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def print_summary(name: str, df: pd.DataFrame):
    """Print summary statistics for a DataFrame."""
    print(f"\n{name.upper()}:")
    print(f"  Rows: {len(df)}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Check for gaps
    expected = pd.date_range(
        start=df["timestamp"].min(),
        end=df["timestamp"].max(),
        freq="1h",
        tz="UTC"
    )
    missing = set(expected) - set(df["timestamp"])
    if missing:
        print(f"  Gaps: {len(missing)} missing hours")
    else:
        print(f"  Gaps: None (complete)")


async def main():
    output_dir = Path("data/prices")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, coin in ASSETS.items():
        print(f"Fetching {name.upper()} ({coin})...")
        df = await fetch_asset_prices(name, coin)

        if df.empty:
            continue

        output_path = output_dir / f"{name}_hourly_prices.parquet"
        df.to_parquet(output_path, index=False)
        print(f"  Saved to {output_path}")
        print_summary(name, df)

        await asyncio.sleep(0.5)  # Brief pause between assets


if __name__ == "__main__":
    asyncio.run(main())
