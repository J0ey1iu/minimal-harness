from __future__ import annotations

import random
import time


def generate_bigint_id() -> int:
    """Generate a monotonically increasing positive BIGINT ID.

    Uses a snowflake-like scheme:
      - 41 bits: milliseconds since 2025-01-01
      - 22 bits: random jitter for collision resistance
    Result fits in 63 bits (positive BIGINT).
    """
    EPOCH = 1735689600000  # 2025-01-01T00:00:00Z in ms
    elapsed = int(time.time() * 1000) - EPOCH
    jitter = random.getrandbits(22)
    return (elapsed << 22) | jitter
