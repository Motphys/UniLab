"""Common parsing and utility functions."""
from typing import List

def parse_sizes(s: str) -> List[int]:
    vals = [int(p.strip()) for p in s.split(",") if p.strip()]
    if not vals:
        raise ValueError("sizes cannot be empty")
    return vals

def pow2_sizes(start: int, end: int) -> List[int]:
    if start > end:
        raise ValueError("start must be <= end")
    return [2**k for k in range(start, end + 1)]

def parse_dtypes(s: str) -> List[str]:
    vals = [p.strip() for p in s.split(",") if p.strip()]
    if not vals:
        raise ValueError("dtypes cannot be empty")
    return vals

def normalize_dtypes(dtypes: List[str]) -> List[str]:
    allowed = {"float16", "float32"}
    kept = []
    for dt in dtypes:
        if dt in allowed and dt not in kept:
            kept.append(dt)
        elif dt not in allowed:
            print(f"  - skip dtype={dt}")
    if not kept:
        raise ValueError("No valid dtypes")
    return kept
