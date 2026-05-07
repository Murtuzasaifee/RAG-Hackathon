"""
Seed API keys into Redis for local development and testing.

Usage:
    uv run python scripts/seed_keys.py

    # Custom Redis URL
    REDIS_URL=redis://localhost:6379/0 uv run python scripts/seed_keys.py

    # Generate a secure random key for production use
    python -c "import secrets; print(secrets.token_hex(32))"

Keys defined in SEED_KEYS below are inserted idempotently —
running this script multiple times is safe.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from datetime import UTC, datetime

import redis.asyncio as aioredis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Edit these keys before seeding. For production, use secrets.token_hex(32).
SEED_KEYS = [
    ("reader-test-key-abc123", "reader", "test-reader"),
    ("editor-test-key-def456", "editor", "test-editor"),
    ("admin-test-key-ghi789",  "admin",  "test-admin"),
]


async def seed(r: aioredis.Redis, raw_key: str, role: str, label: str) -> None:
    sha = hashlib.sha256(raw_key.encode()).hexdigest()
    redis_key = f"apikey:{sha}"

    existing = await r.exists(redis_key)
    if existing:
        print(f"  [skip]  {role:8s}  {label}  (already exists)")
        return

    await r.hset(
        redis_key,
        mapping={
            "key_id": label,
            "role": role,
            "created_at": datetime.now(UTC).isoformat(),
            "label": label,
        },
    )
    print(f"  [seed]  {role:8s}  {label}  →  X-API-Key: {raw_key}")


async def main() -> None:
    print(f"Connecting to Redis: {REDIS_URL}\n")
    r = aioredis.from_url(REDIS_URL)
    try:
        for raw_key, role, label in SEED_KEYS:
            await seed(r, raw_key, role, label)
    finally:
        await r.aclose()
    print("\nDone. Use the keys above in X-API-Key header.")


if __name__ == "__main__":
    asyncio.run(main())
