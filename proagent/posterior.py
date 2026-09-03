"""The learner's belief about a user's preference in each context.

Two Beta-Bernoulli gates per context give a small categorical posterior over the
three soft comfort levels the policy reasons about:

    P(SILENT)  = P(proceed) * P(silent | proceed)
    P(NOTIFY)  = P(proceed) * (1 - P(silent | proceed))
    P(ASK)     = 1 - P(proceed)

The policy (policy.py) turns this distribution into an action by minimizing
expected cost. Uncertainty (small samples) is handled separately by the risk
floor, so here we only need the posterior means.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .types import AutonomyLevel


@dataclass
class BetaGate:
    alpha: float = 1.0   # uniform prior (1, 1)
    beta: float = 1.0
    forgetting: float = 1.0

    def update(self, success: bool) -> None:
        if self.forgetting < 1.0:
            self.alpha = 1.0 + (self.alpha - 1.0) * self.forgetting
            self.beta = 1.0 + (self.beta - 1.0) * self.forgetting
        if success:
            self.alpha += 1.0
        else:
            self.beta += 1.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def n(self) -> float:
        return (self.alpha - 1.0) + (self.beta - 1.0)


@dataclass
class PreferencePosterior:
    forgetting: float = 1.0
    _proceed: dict = field(default_factory=dict)
    _silent: dict = field(default_factory=dict)

    def _gate(self, store: dict, key) -> BetaGate:
        if key not in store:
            store[key] = BetaGate(forgetting=self.forgetting)
        return store[key]

    def category_probs(self, key) -> dict[AutonomyLevel, float]:
        """Distribution over {SILENT, NOTIFY, ASK} for a context."""
        p_proceed = self._gate(self._proceed, key).mean
        p_silent = self._gate(self._silent, key).mean
        return {
            AutonomyLevel.PROCEED_SILENTLY: p_proceed * p_silent,
            AutonomyLevel.PROCEED_AND_NOTIFY: p_proceed * (1.0 - p_silent),
            AutonomyLevel.ASK_FIRST: 1.0 - p_proceed,
        }

    def learn(self, key, comfort_ceiling: AutonomyLevel) -> None:
        proceed_ok = comfort_ceiling <= AutonomyLevel.PROCEED_AND_NOTIFY
        self._gate(self._proceed, key).update(proceed_ok)
        if proceed_ok:
            silent_ok = comfort_ceiling == AutonomyLevel.PROCEED_SILENTLY
            self._gate(self._silent, key).update(silent_ok)

    def snapshot(self, key) -> dict:
        p = self._gate(self._proceed, key)
        s = self._gate(self._silent, key)
        return {
            "proceed_mean": round(p.mean, 3), "proceed_n": round(p.n),
            "silent_mean": round(s.mean, 3), "silent_n": round(s.n),
        }
