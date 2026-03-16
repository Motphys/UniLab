"""Shared observation normalization statistics for multi-process training."""

import sys


class SharedObsNormStats:
    """Synchronize observation normalization statistics between learner and collector.

    Uses a queue to pass (mean, std) tuples from learner to collector.
    """

    def __init__(self, ctx):
        self.q = ctx.Queue(maxsize=2)
        self.last_stats = None

    def put(self, stats):
        """Put new stats, clearing old ones first."""
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except Exception as e:
                print(f"[SharedObsNormStats] queue error: {e}", file=sys.stderr)
        self.q.put(stats)

    def get(self):
        """Get latest stats, returns None if no new stats."""
        try:
            while not self.q.empty():
                self.last_stats = self.q.get_nowait()
        except Exception as e:
            print(f"[SharedObsNormStats] queue error: {e}", file=sys.stderr)
        return self.last_stats
