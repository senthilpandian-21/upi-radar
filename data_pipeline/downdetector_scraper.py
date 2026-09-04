"""Downdetector scraper — pulls per-bank complaint-volume signals.

Downdetector complaints are a leading indicator of payment trouble.
The scraper is defensive: network failures or a changed HTML layout
simply return the last cached complaint volume (or a zero baseline),
so downstream scoring never breaks.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

import requests
from loguru import logger

import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
for _ in range(4):
    if (ROOT / "config.py").exists():
        break
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from config import audit_dir

DOWNDETECTOR_URL = os.getenv(
    "DOWNDETECTOR_URL",
    "https://downdetector.in/status/{bank_slug}",
)
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# bank -> downdetector slug
BANK_SLUGS = {
    "SBI": "state-bank-of-india",
    "HDFC": "hdfc-bank",
    "ICICI": "icici-bank",
    "AXIS": "axis-bank",
    "PNB": "punjab-national-bank",
    "BOB": "bank-of-baroda",
    "KOTAK": "kotak-mahindra-bank",
    "YES": "yes-bank",
}

# crude, resilient complaint-count extraction from page text
COMPLAINT_PATTERNS = [
    re.compile(r"([\d,]{1,7})\s*(?:report(?:s)?|klachten|signalerings?)", re.I),
    re.compile(r"data-complaint-count=\"(\d+)\""),
]


class DowndetectorScraper:
    def __init__(self, timeout: int = 8):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._cache_file = audit_dir() / "downdetector_cache.json"

    # ── core ─────────────────────────────────────────────────────────
    def scrape_bank(self, bank: str) -> dict:
        """
        Returns {"bank": ..., "complaint_volume": int, "status": str, "scraped_at": iso}
        status: ok | degraded | cached | error
        """
        slug = BANK_SLUGS.get(bank.upper(), bank.lower())
        try:
            resp = self.session.get(DOWNDETECTOR_URL.format(bank_slug=slug),
                                    timeout=self.timeout)
            resp.raise_for_status()
            volume = self._extract_complaints(resp.text)
            result = {
                "bank": bank.upper(),
                "complaint_volume": volume,
                "status": "degraded" if volume > 50 else "ok",
                "scraped_at": datetime.now(timezone.utc).isoformat(),
            }
            self._update_cache(result)
            return result
        except Exception as exc:
            cached = self._read_cache().get(bank.upper(), {})
            logger.warning(f"Downdetector scrape failed for {bank} ({exc}) → cache={bool(cached)}")
            if cached:
                return {**cached, "status": "cached"}
            return {"bank": bank.upper(), "complaint_volume": 0,
                    "status": "error", "scraped_at": None}

    def scrape_all(self) -> dict:
        volumes = {}
        for bank in BANK_SLUGS:
            data = self.scrape_bank(bank)
            volumes[data["bank"]] = data
        return volumes

    # ── helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _extract_complaints(html: str) -> int:
        text = re.sub(r"<[^>]+>", " ", html)
        for pattern in COMPLAINT_PATTERNS:
            match = pattern.search(text)
            if match:
                try:
                    return int(match.group(1).replace(",", ""))
                except ValueError:
                    continue
        return 0

    def _cache_path(self):
        return self._cache_file

    def _read_cache(self) -> dict:
        try:
            if self._cache_file.exists():
                return json.loads(self._cache_file.read_text())
        except (OSError, json.JSONDecodeError):
            pass
        return {}

    def _update_cache(self, result: dict) -> None:
        cache = self._read_cache()
        cache[result["bank"]] = result
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            self._cache_file.write_text(json.dumps(cache, indent=2))
        except OSError:
            pass


if __name__ == "__main__":  # pragma: no cover
    scraper = DowndetectorScraper()
    print(json.dumps(scraper.scrape_all(), indent=2))
