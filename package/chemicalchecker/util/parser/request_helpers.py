"""Shared HTTP helpers: persistent cache and PubChem-aware throttling."""

import os
import re
import json
import time
import atexit
import threading
import requests


class _JsonCache:
    """Persistent JSON cache for external API lookups.

    Loaded lazily on first use. Flushed every 20 writes and on process exit.
    Only successful results (non-None) are stored, so failed lookups are
    retried in future runs once connectivity is restored.
    """

    def __init__(self):
        self._path = None
        self._data = None
        self._dirty = 0

    def _resolve_path(self):
        if self._path is not None:
            return
        try:
            from chemicalchecker.util import Config
            tmp_dir = Config().PATH.CC_TMP
        except Exception:
            tmp_dir = os.path.expanduser("~/.cache")
        os.makedirs(tmp_dir, exist_ok=True)
        self._path = os.path.join(tmp_dir, "converter_cache.json")

    def _load(self):
        if self._data is not None:
            return
        self._resolve_path()
        self._data = {}
        if os.path.isfile(self._path):
            try:
                with open(self._path) as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    def get(self, key):
        self._load()
        return self._data.get(key)

    def set(self, key, value):
        self._load()
        self._data[key] = value
        self._dirty += 1
        if self._dirty % 20 == 0:
            self.flush()

    def flush(self):
        if not self._path or not self._data:
            return
        tmp = self._path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(self._data, f)
            os.replace(tmp, self._path)
        except Exception:
            pass
        self._dirty = 0


_cache = _JsonCache()
atexit.register(_cache.flush)

# Seconds to wait before the next request for each PubChem throttle level.
_THROTTLE_DELAYS = {
    "Green":  0.2,   # < 50 % of limit
    "Yellow": 0.5,   # 50–75 % of limit
    "Red":    2.0,   # > 75 % of limit
    "Black":  10.0,  # limit exceeded — backing off
}

_throttle_lock = threading.Lock()
_throttle_last = 0.0
_throttle_delay = _THROTTLE_DELAYS["Green"]


def _parse_throttle_header(value):
    """Return the delay (s) for the worst status in an X-Throttling-Control header."""
    statuses = re.findall(r"status:\s*(Green|Yellow|Red|Black)", value)
    if not statuses:
        return _THROTTLE_DELAYS["Green"]
    for level in ("Black", "Red", "Yellow", "Green"):
        if level in statuses:
            return _THROTTLE_DELAYS[level]
    return _THROTTLE_DELAYS["Green"]


def _throttle():
    """Block until the current inter-request delay has elapsed."""
    global _throttle_last
    with _throttle_lock:
        wait = _throttle_delay - (time.time() - _throttle_last)
        if wait > 0:
            time.sleep(wait)
        _throttle_last = time.time()


def _urlopen_retry(url, timeout=30, retries=3, backoff=5):
    """GET *url* with throttling, dynamic delay from PubChem headers, and retries.

    The response's ``X-Throttling-Control`` header (sent by PubChem) is used
    to adjust the shared inter-request delay. When the header is absent (e.g.
    KEGG), the delay resets to the Green baseline (0.2 s).
    """
    global _throttle_delay

    from io import BytesIO

    for attempt in range(retries):
        _throttle()
        try:
            resp = requests.get(url, timeout=timeout)

            if resp.status_code == 503:
                with _throttle_lock:
                    _throttle_delay = 30.0
                time.sleep(30.0)
                continue

            throttle_hdr = resp.headers.get("X-Throttling-Control", "")
            with _throttle_lock:
                if throttle_hdr:
                    _throttle_delay = _parse_throttle_header(throttle_hdr)
                else:
                    _throttle_delay = _THROTTLE_DELAYS["Green"]

            resp.raise_for_status()
            return BytesIO(resp.content)

        except Exception:
            if attempt < retries - 1:
                time.sleep(backoff)
            else:
                raise