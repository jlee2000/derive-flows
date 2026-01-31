"""Computed fields for timeseries analysis."""

import pandas as pd


def enrich_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Add computed fields to trades DataFrame.

    Args:
        df: DataFrame with core trade fields.

    Returns:
        DataFrame with additional computed columns:
        - dte: Days to expiry
        - moneyness: strike / ref_price
        - moneyness_pct: Percentage OTM/ITM
        - is_itm: In-the-money flag
        - implied_premium_pct: option_price / ref_price * 100
        - notional_value: strike * quantity
        - trade_hour: Hour of trade (UTC)
        - trade_dow: Day of week (0=Monday)
    """
    if df.empty:
        return df

    df = df.copy()

    # Days to expiry (from trade timestamp)
    df["dte"] = (df["expiry"] - df["timestamp"]).dt.days

    # Moneyness metrics
    df["moneyness"] = df["strike"] / df["ref_price"]

    # Percentage OTM/ITM (negative = OTM for puts, positive = ITM for puts)
    # For calls: positive = OTM, negative = ITM
    df["moneyness_pct"] = (df["strike"] / df["ref_price"] - 1) * 100

    # In-the-money flag
    # Put is ITM when strike > ref_price
    # Call is ITM when strike < ref_price
    df["is_itm"] = (
        ((df["instrument"] == "put") & (df["strike"] > df["ref_price"]))
        | ((df["instrument"] == "call") & (df["strike"] < df["ref_price"]))
    )

    # Implied premium as percentage of underlying
    df["implied_premium_pct"] = df["option_price"] / df["ref_price"] * 100

    # Notional value
    df["notional_value"] = df["strike"] * df["quantity"]

    # Time-based features
    df["trade_hour"] = df["timestamp"].dt.hour
    df["trade_dow"] = df["timestamp"].dt.dayofweek

    return df
