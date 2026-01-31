"""Message parsing using regex patterns."""

import re
from datetime import datetime

from derive_flows.models import ParsedTrade

# Pattern breakdown:
# Trade: ETH 27 Feb 26 2,400 Put 15x ($38,130) @ $105.3000, Ref $2,542.01
#
# - Asset: ETH (word characters)
# - Expiry: 27 Feb 26 (day month year)
# - Strike: 2,400 (number with optional commas)
# - Instrument: Put or Call
# - Quantity: 15x (number followed by 'x')
# - Total cost: ($38,130) (in parentheses, with optional commas)
# - Option price: @ $105.3000 (after @, with decimals)
# - Ref price: Ref $2,542.01 (after Ref, with optional commas and decimals)

TRADE_PATTERN = re.compile(
    r"Trade:\s+"
    r"(?P<asset>\w+)\s+"
    r"(?P<day>\d{1,2})\s+(?P<month>\w{3})\s+(?P<year>\d{2})\s+"
    r"(?P<strike>[\d,]+(?:\.\d+)?)\s+"
    r"(?P<instrument>Put|Call)\s+"
    r"(?P<quantity>\d+)x\s+"
    r"\(\$(?P<total_cost>[\d,]+(?:\.\d+)?)\)\s+"
    r"@\s+\$(?P<option_price>[\d,]+(?:\.\d+)?),?\s+"
    r"Ref\s+\$(?P<ref_price>[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_number(value: str) -> float:
    """Parse a number string, removing commas."""
    return float(value.replace(",", ""))


def _parse_expiry(day: str, month: str, year: str) -> datetime:
    """Parse expiry date components into datetime."""
    month_num = MONTH_MAP.get(month.lower())
    if month_num is None:
        raise ValueError(f"Invalid month: {month}")

    # Assume 20xx for two-digit years
    full_year = 2000 + int(year)

    return datetime(full_year, month_num, int(day))


def parse_trade_message(
    text: str,
    message_id: int,
    timestamp: datetime,
) -> ParsedTrade | None:
    """Parse a trade message into a ParsedTrade object.

    Args:
        text: The message text to parse.
        message_id: The Telegram message ID.
        timestamp: The message timestamp.

    Returns:
        ParsedTrade if the message matches the expected format, None otherwise.
    """
    match = TRADE_PATTERN.search(text)
    if not match:
        return None

    try:
        expiry = _parse_expiry(
            match.group("day"),
            match.group("month"),
            match.group("year"),
        )

        return ParsedTrade(
            message_id=message_id,
            timestamp=timestamp,
            asset=match.group("asset").upper(),
            expiry=expiry,
            strike=_parse_number(match.group("strike")),
            instrument=match.group("instrument").lower(),
            quantity=int(match.group("quantity")),
            total_cost=_parse_number(match.group("total_cost")),
            option_price=_parse_number(match.group("option_price")),
            ref_price=_parse_number(match.group("ref_price")),
        )
    except (ValueError, KeyError):
        return None
