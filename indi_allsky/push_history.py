"""Push history cache utility."""
import time
from collections import deque


class PushHistoryCache:
    """Bounded, deque-backed time-windowed cache for push history.

    Stores (int_timestamp, data_dict) entries and supports pruning by
    a seconds window and resizing the backing deque.
    """
    def __init__(self, max_entries=None):
        self.max_entries = int(max_entries) if max_entries else None
        self._entries = deque(maxlen=self.max_entries)

    def append(self, ts, data):
        # store integer-second timestamps to avoid sub-second noise
        self._entries.append((int(ts), data.copy()))

    def get_recent(self, seconds_window):
        if seconds_window <= 0:
            return list(self._entries)
        cutoff = int(time.time()) - int(seconds_window)
        return [(ts, d) for ts, d in self._entries if ts >= cutoff]

    def prune(self, seconds_window):
        if seconds_window <= 0:
            return
        cutoff = int(time.time()) - int(seconds_window)
        while self._entries and self._entries[0][0] < cutoff:
            self._entries.popleft()

    def resize(self, max_entries):
        max_entries = int(max_entries) if max_entries else None
        if max_entries == self.max_entries:
            return
        entries = list(self._entries)
        self.max_entries = max_entries
        self._entries = deque(entries, maxlen=self.max_entries)
