"""Configuration loading from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Config:
    """Application configuration."""

    api_id: int
    api_hash: str
    phone: str
    channel: str
    data_dir: Path

    @property
    def session_path(self) -> Path:
        """Path to Telethon session file."""
        return self.data_dir / "telegram.session"

    @property
    def db_path(self) -> Path:
        """Path to SQLite database."""
        return self.data_dir / "trades.db"


def load_config(env_path: Path | None = None) -> Config:
    """Load configuration from environment variables.

    Args:
        env_path: Optional path to .env file. Defaults to .env in current directory.

    Returns:
        Config object with validated settings.

    Raises:
        ValueError: If required environment variables are missing.
    """
    load_dotenv(env_path)

    api_id = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")
    phone = os.getenv("TELEGRAM_PHONE")
    channel = os.getenv("TELEGRAM_CHANNEL")
    data_dir = os.getenv("DATA_DIR", "./data")

    missing = []
    if not api_id:
        missing.append("TELEGRAM_API_ID")
    if not api_hash:
        missing.append("TELEGRAM_API_HASH")
    if not phone:
        missing.append("TELEGRAM_PHONE")
    if not channel:
        missing.append("TELEGRAM_CHANNEL")

    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    return Config(
        api_id=int(api_id),
        api_hash=api_hash,
        phone=phone,
        channel=channel,
        data_dir=data_path,
    )
