"""Data models for parsed trade messages."""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class ParsedTrade:
    """Represents a parsed options trade from a Telegram message."""

    message_id: int
    timestamp: datetime
    asset: str
    expiry: datetime
    strike: float
    instrument: str  # "put" or "call"
    quantity: int
    total_cost: float
    option_price: float
    ref_price: float

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "asset": self.asset,
            "expiry": self.expiry,
            "strike": self.strike,
            "instrument": self.instrument,
            "quantity": self.quantity,
            "total_cost": self.total_cost,
            "option_price": self.option_price,
            "ref_price": self.ref_price,
        }
