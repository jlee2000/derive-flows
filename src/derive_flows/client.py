"""Telethon client setup and management."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from telethon import TelegramClient

from derive_flows.config import Config


@asynccontextmanager
async def create_client(config: Config) -> AsyncIterator[TelegramClient]:
    """Create and connect a Telegram client.

    Args:
        config: Application configuration with API credentials.

    Yields:
        Connected TelegramClient instance.

    Note:
        First run will prompt for phone verification code.
        If 2FA is enabled, it will also prompt for password.
    """
    client = TelegramClient(
        str(config.session_path),
        config.api_id,
        config.api_hash,
    )

    await client.start(phone=config.phone)

    try:
        yield client
    finally:
        await client.disconnect()


async def get_channel_entity(client: TelegramClient, channel: str):
    """Resolve a channel username or ID to an entity.

    Args:
        client: Connected Telegram client.
        channel: Channel username (without @) or numeric ID.

    Returns:
        Channel entity that can be used with iter_messages.
    """
    # Try to parse as numeric ID first
    try:
        channel_id = int(channel)
        return await client.get_entity(channel_id)
    except ValueError:
        pass

    # Otherwise treat as username
    return await client.get_entity(channel)
