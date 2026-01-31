"""Tests for message parser."""

from datetime import datetime

import pytest

from derive_flows.parser import parse_trade_message, _parse_number, _parse_expiry


class TestParseNumber:
    def test_integer(self):
        assert _parse_number("100") == 100.0

    def test_with_commas(self):
        assert _parse_number("38,130") == 38130.0

    def test_with_decimals(self):
        assert _parse_number("105.3000") == 105.3

    def test_with_commas_and_decimals(self):
        assert _parse_number("2,542.01") == 2542.01


class TestParseExpiry:
    def test_valid_date(self):
        result = _parse_expiry("27", "Feb", "26")
        assert result == datetime(2026, 2, 27)

    def test_case_insensitive_month(self):
        result = _parse_expiry("15", "jan", "25")
        assert result == datetime(2025, 1, 15)

    def test_invalid_month(self):
        with pytest.raises(ValueError):
            _parse_expiry("1", "xyz", "25")


class TestParseTradeMessage:
    def test_eth_put(self):
        """Test parsing ETH put trade."""
        text = "Trade: ETH 27 Feb 26 2,400 Put 15x ($38,130) @ $105.3000, Ref $2,542.01"
        timestamp = datetime(2026, 1, 31, 14, 30)

        trade = parse_trade_message(text, message_id=12345, timestamp=timestamp)

        assert trade is not None
        assert trade.message_id == 12345
        assert trade.timestamp == timestamp
        assert trade.asset == "ETH"
        assert trade.expiry == datetime(2026, 2, 27)
        assert trade.strike == 2400.0
        assert trade.instrument == "put"
        assert trade.quantity == 15
        assert trade.total_cost == 38130.0
        assert trade.option_price == 105.30
        assert trade.ref_price == 2542.01

    def test_btc_call(self):
        """Test parsing BTC call trade."""
        text = "Trade: BTC 15 Mar 26 100,000 Call 5x ($12,500) @ $2,500.00, Ref $98,750.50"
        timestamp = datetime(2026, 2, 1, 10, 0)

        trade = parse_trade_message(text, message_id=99999, timestamp=timestamp)

        assert trade is not None
        assert trade.asset == "BTC"
        assert trade.expiry == datetime(2026, 3, 15)
        assert trade.strike == 100000.0
        assert trade.instrument == "call"
        assert trade.quantity == 5
        assert trade.total_cost == 12500.0
        assert trade.option_price == 2500.0
        assert trade.ref_price == 98750.50

    def test_lowercase_instrument(self):
        """Test that instrument is normalized to lowercase."""
        text = "Trade: ETH 27 Feb 26 2,400 PUT 15x ($38,130) @ $105.3000, Ref $2,542.01"
        trade = parse_trade_message(text, message_id=1, timestamp=datetime.now())

        assert trade is not None
        assert trade.instrument == "put"

    def test_non_trade_message(self):
        """Test that non-trade messages return None."""
        text = "Hello, this is not a trade message"
        trade = parse_trade_message(text, message_id=1, timestamp=datetime.now())

        assert trade is None

    def test_partial_match(self):
        """Test that incomplete trade messages return None."""
        text = "Trade: ETH 27 Feb 26"
        trade = parse_trade_message(text, message_id=1, timestamp=datetime.now())

        assert trade is None

    def test_message_with_prefix(self):
        """Test trade message with preceding text."""
        text = "Alert! Trade: ETH 27 Feb 26 2,400 Put 15x ($38,130) @ $105.3000, Ref $2,542.01"
        trade = parse_trade_message(text, message_id=1, timestamp=datetime.now())

        assert trade is not None
        assert trade.asset == "ETH"

    def test_to_dict(self):
        """Test ParsedTrade.to_dict() method."""
        text = "Trade: ETH 27 Feb 26 2,400 Put 15x ($38,130) @ $105.3000, Ref $2,542.01"
        timestamp = datetime(2026, 1, 31, 14, 30)

        trade = parse_trade_message(text, message_id=12345, timestamp=timestamp)
        data = trade.to_dict()

        assert data["message_id"] == 12345
        assert data["asset"] == "ETH"
        assert data["strike"] == 2400.0
        assert "timestamp" in data
        assert "expiry" in data

    def test_single_digit_day(self):
        """Test parsing with single-digit day."""
        text = "Trade: ETH 5 Jan 26 3,000 Call 10x ($50,000) @ $500.00, Ref $3,200.00"
        trade = parse_trade_message(text, message_id=1, timestamp=datetime.now())

        assert trade is not None
        assert trade.expiry == datetime(2026, 1, 5)

    def test_no_comma_in_option_price(self):
        """Test parsing when option price has no comma."""
        text = "Trade: ETH 27 Feb 26 2,400 Put 15x ($380) @ $25.30, Ref $2,542.01"
        trade = parse_trade_message(text, message_id=1, timestamp=datetime.now())

        assert trade is not None
        assert trade.total_cost == 380.0
        assert trade.option_price == 25.30
