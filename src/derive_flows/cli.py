"""Command-line interface for derive-flows."""

import argparse
import asyncio
import sys
from datetime import datetime

from derive_flows.config import load_config
from derive_flows.ingest.historical import backfill, fetch_trades
from derive_flows.ingest.realtime import stream_trades, run_with_backfill
from derive_flows.storage import TradeStorage
from derive_flows.enrichment import enrich_trades


def print_trade(trade):
    """Print a trade to stdout."""
    print(
        f"[{trade.timestamp}] {trade.asset} {trade.expiry.date()} "
        f"${trade.strike:,.0f} {trade.instrument.upper()} "
        f"{trade.quantity}x @ ${trade.option_price:,.2f}"
    )


async def cmd_fetch(args):
    """Execute fetch command."""
    config = load_config()

    since = datetime.fromisoformat(args.since) if args.since else None
    until = datetime.fromisoformat(args.until) if args.until else None

    print(f"Fetching trades from {config.channel}...")
    if since:
        print(f"  Since: {since}")
    if until:
        print(f"  Until: {until}")
    if args.limit:
        print(f"  Limit: {args.limit}")

    count = await backfill(
        config,
        since=since,
        until=until,
        limit=args.limit,
        on_trade=print_trade if args.verbose else None,
    )

    print(f"\nStored {count} trades")

    if args.output:
        storage = TradeStorage(config.db_path)
        df = await storage.to_dataframe()
        if args.enrich:
            df = enrich_trades(df)
        df.to_csv(args.output, index=False)
        print(f"Exported to {args.output}")


async def cmd_stream(args):
    """Execute stream command."""
    config = load_config()
    print(f"Starting real-time stream from {config.channel}...")
    await stream_trades(config, on_trade=print_trade if args.verbose else None)


async def cmd_run(args):
    """Execute run command (backfill + stream)."""
    config = load_config()
    await run_with_backfill(config, on_trade=print_trade if args.verbose else None)


async def cmd_export(args):
    """Execute export command."""
    config = load_config()
    storage = TradeStorage(config.db_path)

    df = await storage.to_dataframe()
    if df.empty:
        print("No trades in database")
        return

    if args.enrich:
        df = enrich_trades(df)

    output = args.output or "trades.csv"
    df.to_csv(output, index=False)
    print(f"Exported {len(df)} trades to {output}")


async def cmd_stats(args):
    """Execute stats command."""
    config = load_config()
    storage = TradeStorage(config.db_path)

    count = await storage.count()
    if count == 0:
        print("No trades in database")
        return

    df = await storage.to_dataframe()
    df = enrich_trades(df)

    print(f"Total trades: {count}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nBy asset:")
    print(df.groupby("asset").size().to_string())
    print(f"\nBy instrument:")
    print(df.groupby("instrument").size().to_string())
    print(f"\nTotal notional: ${df['notional_value'].sum():,.0f}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Telegram trade message ingestion"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # fetch command
    fetch_parser = subparsers.add_parser(
        "fetch", help="Fetch historical messages"
    )
    fetch_parser.add_argument(
        "--since", help="Start date (ISO format)"
    )
    fetch_parser.add_argument(
        "--until", help="End date (ISO format)"
    )
    fetch_parser.add_argument(
        "--limit", type=int, help="Maximum messages to fetch"
    )
    fetch_parser.add_argument(
        "-o", "--output", help="Export to CSV file"
    )
    fetch_parser.add_argument(
        "--enrich", action="store_true", help="Add computed fields"
    )
    fetch_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print each trade"
    )

    # stream command
    stream_parser = subparsers.add_parser(
        "stream", help="Stream new messages in real-time"
    )
    stream_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print each trade"
    )

    # run command
    run_parser = subparsers.add_parser(
        "run", help="Backfill then stream (recommended)"
    )
    run_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print each trade"
    )

    # export command
    export_parser = subparsers.add_parser(
        "export", help="Export database to CSV"
    )
    export_parser.add_argument(
        "-o", "--output", help="Output file (default: trades.csv)"
    )
    export_parser.add_argument(
        "--enrich", action="store_true", help="Add computed fields"
    )

    # stats command
    subparsers.add_parser("stats", help="Show database statistics")

    args = parser.parse_args()

    try:
        if args.command == "fetch":
            asyncio.run(cmd_fetch(args))
        elif args.command == "stream":
            asyncio.run(cmd_stream(args))
        elif args.command == "run":
            asyncio.run(cmd_run(args))
        elif args.command == "export":
            asyncio.run(cmd_export(args))
        elif args.command == "stats":
            asyncio.run(cmd_stats(args))
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(0)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
