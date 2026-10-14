"""CPU-testable guard for the hardware-specific archived integration."""

from datetime import datetime, timezone
from pathlib import Path

from ..errors import CompatibilityError


def accepted_prefix(target, draft):
    """Greedy speculative result through the first rejected draft, plus bonus."""
    if len(target) != len(draft) + 1:
        raise ValueError("Expected one target row per draft token plus a bonus row")
    output = []
    for token, proposal in zip(target, draft):
        output.append(token)
        if token != proposal:
            return output
    return output + [target[-1]]


def runtime_settings(environ, *, now=None, optimized=False):
    if optimized:
        raise CompatibilityError(
            "Prefix V8 cannot run with Python -O: invariant assertions are required"
        )
    value = environ.get("DRAFTSCOPE_DEADLINE_UTC")
    folder = environ.get("DRAFTSCOPE_CACHE_DIR")
    if not value or not folder:
        raise CompatibilityError(
            "Use the bounded CLI; prefix V8 requires a UTC deadline and per-run cache directory"
        )
    try:
        deadline = datetime.fromisoformat(value)
    except ValueError as exc:
        raise CompatibilityError("Invalid prefix deadline") from exc
    if deadline.tzinfo is None or deadline.utcoffset() is None:
        raise CompatibilityError("Prefix deadline must be timezone-aware")
    now = now or datetime.now(timezone.utc)
    if not 0 < (deadline - now).total_seconds() <= 86400:
        raise CompatibilityError("Prefix deadline expired or exceeds the 24-hour maximum")
    base = Path(folder)
    if not base.is_absolute():
        raise CompatibilityError("Prefix cache path must be absolute")
    return base, deadline
