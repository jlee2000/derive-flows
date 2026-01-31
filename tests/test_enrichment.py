"""Tests for enrichment module."""

from datetime import datetime

import pandas as pd
import pytest

from derive_flows.enrichment import enrich_trades


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        "message_id": [1, 2],
        "timestamp": [
            datetime(2026, 1, 31, 14, 30),
            datetime(2026, 1, 31, 16, 45),
        ],
        "asset": ["ETH", "BTC"],
        "expiry": [
            datetime(2026, 2, 27),
            datetime(2026, 3, 15),
        ],
        "strike": [2400.0, 100000.0],
        "instrument": ["put", "call"],
        "quantity": [15, 5],
        "total_cost": [38130.0, 12500.0],
        "option_price": [105.30, 2500.0],
        "ref_price": [2542.01, 98750.50],
    })


class TestEnrichTrades:
    def test_dte_calculation(self, sample_df):
        """Test days to expiry calculation."""
        df = enrich_trades(sample_df)

        # First trade: Jan 31 14:30 to Feb 27 00:00 = 26 days (fractional day truncated)
        assert df.loc[0, "dte"] == 26

        # Second trade: Jan 31 16:45 to Mar 15 00:00 = 42 days (fractional day truncated)
        assert df.loc[1, "dte"] == 42

    def test_moneyness(self, sample_df):
        """Test moneyness calculation."""
        df = enrich_trades(sample_df)

        # ETH: 2400 / 2542.01 ≈ 0.944
        assert abs(df.loc[0, "moneyness"] - 0.944) < 0.01

        # BTC: 100000 / 98750.50 ≈ 1.013
        assert abs(df.loc[1, "moneyness"] - 1.013) < 0.01

    def test_moneyness_pct(self, sample_df):
        """Test moneyness percentage calculation."""
        df = enrich_trades(sample_df)

        # ETH: (2400/2542.01 - 1) * 100 ≈ -5.6%
        assert abs(df.loc[0, "moneyness_pct"] - (-5.6)) < 0.1

        # BTC: (100000/98750.50 - 1) * 100 ≈ 1.3%
        assert abs(df.loc[1, "moneyness_pct"] - 1.3) < 0.1

    def test_is_itm_put(self, sample_df):
        """Test ITM flag for put option."""
        df = enrich_trades(sample_df)

        # Put with strike 2400 < ref 2542.01 is OTM
        assert df.loc[0, "is_itm"] == False

    def test_is_itm_call(self, sample_df):
        """Test ITM flag for call option."""
        df = enrich_trades(sample_df)

        # Call with strike 100000 > ref 98750.50 is OTM
        assert df.loc[1, "is_itm"] == False

    def test_itm_put_above_ref(self):
        """Test ITM put when strike > ref."""
        df = pd.DataFrame({
            "message_id": [1],
            "timestamp": [datetime(2026, 1, 31)],
            "asset": ["ETH"],
            "expiry": [datetime(2026, 2, 27)],
            "strike": [3000.0],
            "instrument": ["put"],
            "quantity": [10],
            "total_cost": [10000.0],
            "option_price": [100.0],
            "ref_price": [2500.0],  # strike > ref = ITM for put
        })

        result = enrich_trades(df)
        assert result.loc[0, "is_itm"] == True

    def test_itm_call_below_ref(self):
        """Test ITM call when strike < ref."""
        df = pd.DataFrame({
            "message_id": [1],
            "timestamp": [datetime(2026, 1, 31)],
            "asset": ["ETH"],
            "expiry": [datetime(2026, 2, 27)],
            "strike": [2000.0],
            "instrument": ["call"],
            "quantity": [10],
            "total_cost": [10000.0],
            "option_price": [100.0],
            "ref_price": [2500.0],  # strike < ref = ITM for call
        })

        result = enrich_trades(df)
        assert result.loc[0, "is_itm"] == True

    def test_implied_premium_pct(self, sample_df):
        """Test implied premium percentage calculation."""
        df = enrich_trades(sample_df)

        # ETH: 105.30 / 2542.01 * 100 ≈ 4.14%
        assert abs(df.loc[0, "implied_premium_pct"] - 4.14) < 0.1

    def test_notional_value(self, sample_df):
        """Test notional value calculation."""
        df = enrich_trades(sample_df)

        # ETH: 2400 * 15 = 36000
        assert df.loc[0, "notional_value"] == 36000.0

        # BTC: 100000 * 5 = 500000
        assert df.loc[1, "notional_value"] == 500000.0

    def test_trade_hour(self, sample_df):
        """Test trade hour extraction."""
        df = enrich_trades(sample_df)

        assert df.loc[0, "trade_hour"] == 14
        assert df.loc[1, "trade_hour"] == 16

    def test_trade_dow(self, sample_df):
        """Test day of week extraction."""
        df = enrich_trades(sample_df)

        # Jan 31, 2026 is a Saturday (dayofweek=5)
        assert df.loc[0, "trade_dow"] == 5
        assert df.loc[1, "trade_dow"] == 5

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        result = enrich_trades(df)
        assert result.empty

    def test_preserves_original_columns(self, sample_df):
        """Test that original columns are preserved."""
        df = enrich_trades(sample_df)

        for col in sample_df.columns:
            assert col in df.columns
