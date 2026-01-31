"""SQLite storage backend for trade data."""

import aiosqlite
import pandas as pd
from pathlib import Path

from derive_flows.models import ParsedTrade


class TradeStorage:
    """Async SQLite storage for parsed trades."""

    def __init__(self, db_path: Path):
        """Initialize storage with database path.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path

    async def initialize(self) -> None:
        """Create database tables if they don't exist."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    message_id INTEGER PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    asset TEXT NOT NULL,
                    expiry TEXT NOT NULL,
                    strike REAL NOT NULL,
                    instrument TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    total_cost REAL NOT NULL,
                    option_price REAL NOT NULL,
                    ref_price REAL NOT NULL
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_timestamp
                ON trades(timestamp)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_asset
                ON trades(asset)
            """)
            await db.commit()

    async def insert(self, trade: ParsedTrade) -> None:
        """Insert or replace a trade record.

        Uses INSERT OR REPLACE for deduplication via message_id primary key.

        Args:
            trade: Parsed trade to store.
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO trades
                (message_id, timestamp, asset, expiry, strike, instrument,
                 quantity, total_cost, option_price, ref_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.message_id,
                    trade.timestamp.isoformat(),
                    trade.asset,
                    trade.expiry.isoformat(),
                    trade.strike,
                    trade.instrument,
                    trade.quantity,
                    trade.total_cost,
                    trade.option_price,
                    trade.ref_price,
                ),
            )
            await db.commit()

    async def insert_many(self, trades: list[ParsedTrade]) -> int:
        """Insert multiple trades in a single transaction.

        Args:
            trades: List of parsed trades to store.

        Returns:
            Number of trades inserted.
        """
        if not trades:
            return 0

        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(
                """
                INSERT OR REPLACE INTO trades
                (message_id, timestamp, asset, expiry, strike, instrument,
                 quantity, total_cost, option_price, ref_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        t.message_id,
                        t.timestamp.isoformat(),
                        t.asset,
                        t.expiry.isoformat(),
                        t.strike,
                        t.instrument,
                        t.quantity,
                        t.total_cost,
                        t.option_price,
                        t.ref_price,
                    )
                    for t in trades
                ],
            )
            await db.commit()

        return len(trades)

    async def get_latest_message_id(self) -> int | None:
        """Get the most recent message ID in the database.

        Returns:
            Latest message_id or None if database is empty.
        """
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT MAX(message_id) FROM trades"
            )
            row = await cursor.fetchone()
            return row[0] if row and row[0] else None

    async def to_dataframe(self) -> pd.DataFrame:
        """Export all trades to a pandas DataFrame.

        Returns:
            DataFrame with all trades, sorted by timestamp.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM trades ORDER BY timestamp"
            )
            rows = await cursor.fetchall()

        if not rows:
            return pd.DataFrame(columns=[
                "message_id", "timestamp", "asset", "expiry", "strike",
                "instrument", "quantity", "total_cost", "option_price", "ref_price",
            ])

        df = pd.DataFrame([dict(row) for row in rows])

        # Convert string columns to proper types
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["expiry"] = pd.to_datetime(df["expiry"])

        return df

    async def count(self) -> int:
        """Get total number of trades in database.

        Returns:
            Count of trade records.
        """
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM trades")
            row = await cursor.fetchone()
            return row[0] if row else 0
