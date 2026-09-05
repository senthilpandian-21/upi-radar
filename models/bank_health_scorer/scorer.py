"""Bank Health Scorer — weighted 0-100 health per bank, cached in Redis.

Weights (matching the system design):
    score = 0.50 × last-15-min health + 0.30 × last-1hr health
          + 0.20 × historical same-hour health
where health = max(0, 100 − TD% × 1000).

Redis is optional: if unavailable the scorer keeps an in-process cache
with the same 5-minute TTL semantics, so the demo never breaks.
"""
from __future__ import annotations

import json
from datetime import datetime

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


from config import redis_config

try:
    import redis as redis_lib

    REDIS_AVAILABLE = True
except Exception:  # pragma: no cover
    redis_lib = None
    REDIS_AVAILABLE = False

CACHE_TTL_SECONDS = 300  # 5 min
DEFAULT_SCORE = 100.0

# demo fallback scores — base values from the design doc + noise
DEMO_BASE_SCORES = {
    "SBI": 72.0, "HDFC": 96.0, "ICICI": 88.0, "AXIS": 91.0,
    "PNB": 61.0, "BOB": 54.0, "KOTAK": 93.0, "YES": 89.0,
}


class BankHealthScorer:
    def __init__(self, use_redis: bool = True):
        self._redis = None
        self._memory_cache: dict[str, tuple[float, str]] = {}
        if use_redis and REDIS_AVAILABLE:
            try:
                self._redis = redis_lib.Redis(**redis_config())
                self._redis.ping()
            except Exception as exc:
                logger.debug(f"Redis unavailable ({exc}) — in-process cache active")
                self._redis = None

    @property
    def connected(self) -> bool:
        return self._redis is not None

    # ── scoring math (pure, deterministic) ───────────────────────────
    @staticmethod
    def calculate_score(bank: str, td_15min: float, td_1hr: float,
                        td_historical: float) -> float:
        """Weighted bank health score, 0-100. Higher = healthier."""
        health_15min = max(0.0, 100.0 - td_15min * 1000)
        health_1hr = max(0.0, 100.0 - td_1hr * 1000)
        health_hist = max(0.0, 100.0 - td_historical * 1000)
        score = 0.50 * health_15min + 0.30 * health_1hr + 0.20 * health_hist
        return round(min(max(score, 0.0), 100.0), 2)

    @staticmethod
    def _score_to_status(score: float) -> str:
        if score >= 90:
            return "EXCELLENT"
        if score >= 70:
            return "GOOD"
        if score >= 50:
            return "DEGRADED"
        return "CRITICAL"

    # ── cache layer ──────────────────────────────────────────────────
    def update_bank_score(self, bank: str, score: float) -> None:
        data = {"score": score, "status": self._score_to_status(score),
                "updated_at": datetime.now().isoformat(),
                "source": "live"}
        if self._redis is not None:
            try:
                self._redis.setex(f"bank_health:{bank.upper()}", CACHE_TTL_SECONDS,
                                  json.dumps(data))
                return
            except Exception:
                pass
        self._memory_cache[bank.upper()] = (score, datetime.now().isoformat())

    def get_bank_score(self, bank: str) -> dict:
        cache = self._fetch(bank.upper())
        if cache is not None:
            return cache
        return {"score": DEFAULT_SCORE, "status": "EXCELLENT",
                "updated_at": datetime.now().isoformat(),
                "source": "default"}

    def _fetch(self, bank: str) -> dict | None:
        if self._redis is not None:
            try:
                raw = self._redis.get(f"bank_health:{bank}")
                if raw:
                    return json.loads(raw)
            except Exception:
                pass
        entry = self._memory_cache.get(bank)
        if entry:
            return {"score": entry[0], "status": self._score_to_status(entry[0]),
                    "updated_at": entry[1], "source": "live"}
        return None

    def get_all_bank_scores(self) -> dict:
        """Scores for every bank in bank_config (Redis/memory or defaults)."""
        from models.bank_health_scorer.bank_config import ALL_BANKS

        scores = {}
        for bank in ALL_BANKS:
            scores[bank] = self.get_bank_score(bank)
        return scores

    def get_best_available_banks(self, min_score: float = 50.0,
                                 exclude: list[str] | None = None) -> list:
        """Banks sorted by health desc, filtered by min_score / exclude."""
        exclude = exclude or []
        scores = self.get_all_bank_scores()
        healthy = sorted(
            ((bank, data["score"]) for bank, data in scores.items()
             if bank not in exclude and data["score"] >= min_score),
            key=lambda item: item[1], reverse=True,
        )
        return healthy

    # ── demo helpers ─────────────────────────────────────────────────
    def simulate_bank_scores(self, seed: int | None = None) -> dict:
        """Realistic demo scores for all banks (design doc values ± 5 noise)."""
        import numpy as np

        rng = np.random.default_rng(seed)
        scores = {}
        for bank, base in DEMO_BASE_SCORES.items():
            noise = float(rng.uniform(-5, 5))
            score = round(min(max(base + noise, 1.0), 100.0), 2)
            scores[bank] = {"score": score, "status": self._score_to_status(score),
                            "updated_at": datetime.now().isoformat(),
                            "source": "simulated"}
        return scores

    def refresh_from_simulation(self) -> dict:
        """Store demo scores into the cache (used when no live telemetry)."""
        scores = self.simulate_bank_scores()
        for bank, data in scores.items():
            self.update_bank_score(bank, data["score"])
        return scores


# ── shared instance ─────────────────────────────────────────────────────
# Every component (API routes, RouterAgent, MonitorAgent) must see the SAME
# cache — otherwise a freshly constructed scorer reports the default 100.0
# for every bank and the audit trail records scores nobody actually used.
_SHARED_SCORER: BankHealthScorer | None = None


def get_shared_scorer() -> BankHealthScorer:
    global _SHARED_SCORER
    if _SHARED_SCORER is None:
        _SHARED_SCORER = BankHealthScorer()
    return _SHARED_SCORER


def reset_shared_scorer() -> None:
    """Test hook: drop the shared instance (fresh cache per test)."""
    global _SHARED_SCORER
    _SHARED_SCORER = None


if __name__ == "__main__":  # pragma: no cover
    scorer = BankHealthScorer()
    print(json.dumps(scorer.simulate_bank_scores(seed=1), indent=2))
    print("Best banks:", [b for b, s in scorer.get_best_available_banks()])
