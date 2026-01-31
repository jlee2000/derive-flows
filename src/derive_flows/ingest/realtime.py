"""Real-time message streaming."""

import asyncio
from typing import Callable

from telethon import events

from derive_flows.client import create_client, get_channel_entity
from derive_flows.config import Config, load_config
from derive_flows.models import ParsedTrade
from derive_flows.parser import parse_trade_message
from derive_flows.storage import TradeStorage


async def stream_trades(
    config: Config | None = None,
    on_trade: Callable[[ParsedTrade], None] | None = None,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Stream new trades from Telegram channel in real-time.

    Args:
        config: Configuration object. If None, loads from environment.
        on_trade: Optional callback for each parsed trade.
        stop_event: Optional event to signal stream should stop.
    """
    if config is None:
        config = load_config()

    storage = TradeStorage(config.db_path)
    await storage.initialize()

    async with create_client(config) as client:
        channel = await get_channel_entity(client, config.channel)
        channel_id = channel.id

        @client.on(events.NewMessage(chats=channel_id))
        async def handler(event):
            if not event.message.text:
                return

            trade = parse_trade_message(
                event.message.text,
                event.message.id,
                event.message.date.replace(tzinfo=None),
            )

            if trade:
                await storage.insert(trade)
                if on_trade:
                    on_trade(trade)

        print(f"Streaming trades from channel: {config.channel}")
        print("Press Ctrl+C to stop...")

        if stop_event:
            await stop_event.wait()
        else:
            await client.run_until_disconnected()


async def run_with_backfill(
    config: Config | None = None,
    on_trade: Callable[[ParsedTrade], None] | None = None,
) -> None:
    """Backfill historical messages then stream new ones.

    Recommended for production use - ensures no gaps in data.

    Args:
        config: Configuration object. If None, loads from environment.
        on_trade: Optional callback for each parsed trade.
    """
    if config is None:
        config = load_config()

    # Import here to avoid circular dependency
    from derive_flows.ingest.historical import backfill

    print("Starting backfill...")
    count = await backfill(config, on_trade=on_trade)
    print(f"Backfill complete: {count} trades stored")

    print("Starting real-time stream...")
    await stream_trades(config, on_trade=on_trade)
