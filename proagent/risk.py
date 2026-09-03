"""Statistical risk floor.

Second of the two floors (the first is safety.py). This one is data-driven: it
won't let the agent act on its own in a context until the feedback there says
acting is unlikely to annoy the user.

Per context I keep one bit per feedback: was proceeding without asking actually
fine? Count the "not fine" ones as failures. For k failures in n tries, the
Clopper-Pearson upper bound U on the true failure rate holds w.p. >= 1 - delta.
Certify proceeding when U <= epsilon. A second, stricter bound gates silent vs
notify.

Caveat I'm aware of: this assumes the per-context feedback is exchangeable, so it
degrades if a user's preference drifts. See the future-work note in DESIGN.md for
how I'd handle that (adaptive conformal).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from scipy.stats import beta as _beta

from .config import RiskFloor
from .types import AutonomyLevel


def clopper_pearson_upper(k: int, n: int, delta: float) -> float:
    """(1 - delta) upper bound on a binomial failure rate. 1.0 if no evidence."""
    if n == 0 or k >= n:
        return 1.0
    return float(_beta.ppf(1.0 - delta, k + 1, n - k))


@dataclass
class _Counter:
    n: float = 0.0
    fails: float = 0.0
    forgetting: float = 1.0

    def update(self, failed: bool) -> None:
        if self.forgetting < 1.0:          # down-weight old feedback for drift
            self.n *= self.forgetting
            self.fails *= self.forgetting
        self.n += 1.0
        if failed:
            self.fails += 1.0


@dataclass
class RiskCertifier:
    cfg: RiskFloor = field(default_factory=RiskFloor)
    forgetting: float = 1.0
    _proceed: dict = field(default_factory=dict)
    _silent: dict = field(default_factory=dict)

    def _c(self, store, key) -> _Counter:
        if key not in store:
            store[key] = _Counter(forgetting=self.forgetting)
        return store[key]

    def _upper(self, store, key) -> float:
        c = self._c(store, key)
        return clopper_pearson_upper(round(c.fails), round(c.n), self.cfg.delta)

    def certified_level(self, key):
        u_proceed = self._upper(self._proceed, key)
        u_silent = self._upper(self._silent, key)
        enough_p = round(self._c(self._proceed, key).n) >= self.cfg.min_evidence
        enough_s = round(self._c(self._silent, key).n) >= self.cfg.min_evidence

        if not (enough_p and u_proceed <= self.cfg.epsilon):
            return AutonomyLevel.ASK_FIRST, u_proceed, u_silent
        if enough_s and u_silent <= self.cfg.epsilon:
            return AutonomyLevel.PROCEED_SILENTLY, u_proceed, u_silent
        return AutonomyLevel.PROCEED_AND_NOTIFY, u_proceed, u_silent

    def learn(self, key, comfort_ceiling) -> None:
        proceed_failed = comfort_ceiling >= AutonomyLevel.ASK_FIRST
        self._c(self._proceed, key).update(proceed_failed)
        if not proceed_failed:
            silent_failed = comfort_ceiling >= AutonomyLevel.PROCEED_AND_NOTIFY
            self._c(self._silent, key).update(silent_failed)

    def snapshot(self, key) -> dict:
        return {"proceed_upper": round(self._upper(self._proceed, key), 3),
                "silent_upper": round(self._upper(self._silent, key), 3)}
