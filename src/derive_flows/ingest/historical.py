"""Historical message backfill."""

from datetime import datetime
from typing import Callable

import pandas as pd

from derive_flows.client import create_client, get_channel_entity
from derive_flows.config import Config, load_config
from derive_flows.enrichment import enrich_trades
from derive_flows.models import ParsedTrade
from derive_flows.parser import parse_trade_message
from derive_flows.storage import TradeStorage


async def backfill(
    config: Config | None = None,
    since: datetime | str | None = None,
    until: datetime | str | None = None,
    limit: int | None = None,
    on_trade: Callable[[ParsedTrade], None] | None = None,
) -> int:
    """Backfill historical messages from Telegram channel.

    Args:
        config: Configuration object. If None, loads from environment.
        since: Only fetch messages after this date.
        until: Only fetch messages before this date.
        limit: Maximum number of messages to fetch.
        on_trade: Optional callback for each parsed trade.

    Returns:
        Number of trades stored.
    """
    if config is None:
        config = load_config()

    # Parse date strings
    if isinstance(since, str):
        since = datetime.fromisoformat(since)
    if isinstance(until, str):
        until = datetime.fromisoformat(until)

    storage = TradeStorage(config.db_path)
    await storage.initialize()

    trades: list[ParsedTrade] = []

    async with create_client(config) as client:
        channel = await get_channel_entity(client, config.channel)

        async for message in client.iter_messages(
            channel,
            offset_date=until,
            limit=limit,
        ):
            # Skip if before 'since' date
            if since and message.date.replace(tzinfo=None) < since:
                break

            if not message.text:
                continue

            trade = parse_trade_message(
                message.text,
                message.id,
                message.date.replace(tzinfo=None),
            )

            if trade:
                trades.append(trade)
                if on_trade:
                    on_trade(trade)

    # Batch insert
    count = await storage.insert_many(trades)
    return count


async def fetch_trades(
    config: Config | None = None,
    since: datetime | str | None = None,
    until: datetime | str | None = None,
    limit: int | None = None,
    enrich: bool = False,
) -> pd.DataFrame:
    """Fetch trades and return as DataFrame.

    Performs backfill if needed, then exports from storage.

    Args:
        config: Configuration object. If None, loads from environment.
        since: Only fetch messages after this date.
        until: Only fetch messages before this date.
        limit: Maximum number of messages to fetch.
        enrich: Whether to add computed fields.

    Returns:
        DataFrame with trade data.
    """
    if config is None:
        config = load_config()

    # Run backfill
    await backfill(config, since=since, until=until, limit=limit)

    # Export to DataFrame
    storage = TradeStorage(config.db_path)
    df = await storage.to_dataframe()

    # Filter by date range if specified
    if since:
        if isinstance(since, str):
            since = datetime.fromisoformat(since)
        df = df[df["timestamp"] >= since]

    if until:
        if isinstance(until, str):
            until = datetime.fromisoformat(until)
        df = df[df["timestamp"] <= until]

    if enrich:
        df = enrich_trades(df)

    return df
