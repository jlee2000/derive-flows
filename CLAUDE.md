# Claude Instructions: derive-flows

This document provides instructions for running and modifying the Telegram trade message ingestion pipeline.

## Quick Start: Historical Ingestion

### Prerequisites

1. **Telegram API credentials** from https://my.telegram.org/apps
2. **Python 3.10+** with pip

### Setup Steps

```bash
# 1. Install the package
pip install -e ".[dev]"

# 2. Create .env file from template
cp .env.example .env

# 3. Edit .env with your credentials:
#    TELEGRAM_API_ID=your_api_id
#    TELEGRAM_API_HASH=your_api_hash
#    TELEGRAM_PHONE=+1234567890  (your actual phone)
#    TELEGRAM_CHANNEL=channel_name  (without @)
#    DATA_DIR=./data

# 4. First run requires interactive authentication (run in terminal, not programmatically)
derive-flows fetch --limit 10 -v
# You'll be prompted for verification code sent to Telegram

# 5. After authentication, fetch full history
derive-flows fetch -v
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `derive-flows fetch` | Fetch historical messages |
| `derive-flows fetch --limit N` | Fetch last N messages |
| `derive-flows fetch --since 2024-01-01` | Fetch from date |
| `derive-flows stream` | Real-time streaming |
| `derive-flows run` | Backfill + stream (production) |
| `derive-flows export --enrich -o trades.csv` | Export to CSV |
| `derive-flows stats` | Show database statistics |

### Python API

```python
import asyncio
from derive_flows.storage import TradeStorage
from derive_flows.enrichment import enrich_trades
from pathlib import Path

async def load_trades():
    storage = TradeStorage(Path("./data/trades.db"))
    df = await storage.to_dataframe()
    return enrich_trades(df)

df = asyncio.run(load_trades())
```

---

## Architecture Overview

```
src/derive_flows/
├── __init__.py         # Public API exports
├── config.py           # Environment variable loading
├── client.py           # Telethon client setup
├── parser.py           # Regex message parsing
├── models.py           # ParsedTrade dataclass
├── storage.py          # SQLite backend
├── enrichment.py       # Computed fields
├── cli.py              # CLI entry points
└── ingest/
    ├── __init__.py
    ├── historical.py   # Backfill logic
    └── realtime.py     # Live streaming
```

### Data Flow

```
Telegram Channel
       │
       ▼
   client.py          TelegramClient connection
       │
       ▼
   parser.py          Regex extraction → ParsedTrade
       │
       ▼
   storage.py         SQLite INSERT OR REPLACE
       │
       ▼
   enrichment.py      Add computed columns
       │
       ▼
   pandas DataFrame
```

---

## Design Decisions

### 1. Message Parsing (parser.py)

**Format expected:**
```
Trade: ETH 27 Feb 26 2,400 Put 15x ($38,130) @ $105.3000, Ref $2,542.01
```

**Regex pattern breakdown:**
- `Trade:\s+` - literal prefix
- `(?P<asset>\w+)` - asset symbol (ETH, BTC, HYPE)
- `(?P<day>\d{1,2})\s+(?P<month>\w{3})\s+(?P<year>\d{2})` - expiry date
- `(?P<strike>[\d,]+(?:\.\d+)?)` - strike price with optional commas
- `(?P<instrument>Put|Call)` - option type (case insensitive)
- `(?P<quantity>\d+)x` - quantity with 'x' suffix
- `\(\$(?P<total_cost>[\d,]+(?:\.\d+)?)\)` - total cost in parentheses
- `@\s+\$(?P<option_price>[\d,]+(?:\.\d+)?)` - per-contract price
- `Ref\s+\$(?P<ref_price>[\d,]+(?:\.\d+)?)` - reference/spot price

**To support new message formats:** Modify `TRADE_PATTERN` in `parser.py`. The pattern uses named groups, so field extraction in `parse_trade_message()` will continue working if group names are preserved.

### 2. Storage (storage.py)

**SQLite schema:**
```sql
CREATE TABLE trades (
    message_id INTEGER PRIMARY KEY,  -- Telegram message ID, ensures deduplication
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
```

**Key decisions:**
- `message_id` as PRIMARY KEY enables idempotent ingestion via `INSERT OR REPLACE`
- Dates stored as ISO strings for SQLite compatibility
- Converted to datetime on DataFrame export

### 3. Enrichment (enrichment.py)

Computed fields added to DataFrame:

| Field | Formula | Purpose |
|-------|---------|---------|
| `dte` | expiry - timestamp | Days to expiration |
| `moneyness` | strike / ref_price | Strike relative to spot |
| `moneyness_pct` | (strike / ref_price - 1) * 100 | % OTM/ITM |
| `is_itm` | Put: strike > ref, Call: strike < ref | In-the-money flag |
| `implied_premium_pct` | option_price / ref_price * 100 | Premium as % of spot |
| `notional_value` | strike * quantity | Position size |
| `trade_hour` | timestamp.hour | For time-of-day analysis |
| `trade_dow` | timestamp.dayofweek | For day-of-week analysis |

### 4. Authentication (client.py)

- Telethon session saved to `{DATA_DIR}/telegram.session`
- First run requires interactive terminal for phone verification
- Subsequent runs reuse session automatically
- 2FA password prompted if account has it enabled

### 5. Async Design

All I/O operations are async:
- `aiosqlite` for non-blocking database access
- `telethon` for async Telegram API
- CLI uses `asyncio.run()` to bridge sync entry points

---

## Common Modifications

### Adding a new parsed field

1. Add field to `ParsedTrade` dataclass in `models.py`
2. Add named group to `TRADE_PATTERN` in `parser.py`
3. Extract in `parse_trade_message()` return statement
4. Add column to SQLite schema in `storage.py` (both CREATE TABLE and INSERT)
5. Update `to_dataframe()` column list for empty DataFrame case

### Adding a new computed field

1. Add calculation in `enrich_trades()` in `enrichment.py`
2. Add test in `tests/test_enrichment.py`

### Supporting a new message format

1. Create new regex pattern or modify `TRADE_PATTERN`
2. Add test cases in `tests/test_parser.py`
3. Pattern uses `re.search()` so it finds matches anywhere in message text

### Adding a new CLI command

1. Add `async def cmd_newcommand(args)` in `cli.py`
2. Add subparser in `main()` function
3. Add dispatch in the try block

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_parser.py -v

# Run with coverage
pytest tests/ --cov=derive_flows
```

Test files:
- `tests/test_parser.py` - Message parsing tests
- `tests/test_enrichment.py` - Computed field tests

---

## File Locations

| File | Purpose |
|------|---------|
| `.env` | Credentials (gitignored) |
| `.env.example` | Template for credentials |
| `data/telegram.session` | Telethon auth session (gitignored) |
| `data/trades.db` | SQLite database (gitignored) |

---

## Troubleshooting

### "PhoneNumberInvalidError"
Phone number format must include country code: `+14155551234`

### "EOF when reading a line"
First authentication must run in interactive terminal, not programmatically.

### Messages not parsing
Check message format against `TRADE_PATTERN`. Use verbose flag to see which messages are being parsed: `derive-flows fetch --limit 10 -v`

### Duplicate trades
Not an issue - `INSERT OR REPLACE` with `message_id` primary key handles deduplication automatically.
