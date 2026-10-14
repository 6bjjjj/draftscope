"""Arithmetic that distinguishes request throughput from measured decode throughput."""

from dataclasses import asdict, dataclass
import math
import statistics


@dataclass(frozen=True)
class Timing:
    generated_tokens: int
    request_seconds: float
    first_token_ts: float | None = None
    last_token_ts: float | None = None

    def __post_init__(self):
        if type(self.generated_tokens) is not int or self.generated_tokens < 0:
            raise ValueError("generated_tokens must be a nonnegative integer")
        if not math.isfinite(self.request_seconds) or self.request_seconds <= 0:
            raise ValueError("request_seconds must be finite and positive")
        if (self.first_token_ts is None) != (self.last_token_ts is None):
            raise ValueError("Supply both token timestamps or neither")
        if self.first_token_ts is not None:
            if not all(math.isfinite(x) for x in (self.first_token_ts, self.last_token_ts)):
                raise ValueError("Token timestamps must be finite")
            if self.first_token_ts <= 0 or self.last_token_ts < self.first_token_ts:
                raise ValueError("Invalid token timestamp order")
            if self.generated_tokens > 1 and self.last_token_ts == self.first_token_ts:
                raise ValueError("Multiple tokens need a positive decode duration")

    @property
    def request_tokens_per_second(self):
        return self.generated_tokens / self.request_seconds

    @property
    def decode_tokens_per_second(self):
        if self.generated_tokens < 2 or self.first_token_ts is None:
            return None
        return (self.generated_tokens - 1) / (self.last_token_ts - self.first_token_ts)

    def to_dict(self):
        return {
            **asdict(self),
            "request_tokens_per_second": self.request_tokens_per_second,
            "decode_tokens_per_second": self.decode_tokens_per_second,
        }


def summarize(records):
    if not records:
        raise ValueError("No timing records")
    times = [
        Timing(
            r["generated_tokens"],
            r["request_seconds"],
            r.get("first_token_ts"),
            r.get("last_token_ts"),
        )
        for r in records
    ]
    decodes = [t.decode_tokens_per_second for t in times if t.decode_tokens_per_second is not None]
    return {
        "requests": len(times),
        "pooled_request_tokens_per_second": sum(t.generated_tokens for t in times)
        / sum(t.request_seconds for t in times),
        "median_request_tokens_per_second": statistics.median(
            t.request_tokens_per_second for t in times
        ),
        "median_decode_tokens_per_second": statistics.median(decodes) if decodes else None,
        "decode_observations": len(decodes),
        "quality_accepted": None,
    }
