"""The decision-theoretic policy.

Given the posterior over a user's comfort level and the loss matrix, pick the
soft action (SILENT, NOTIFY, or ASK) that minimizes expected cost. This is the
part that replaced the two hand-tuned thresholds: the choice now follows from an
explicit, asymmetric cost model, and the effective "how sure do I need to be"
threshold is implied by the c_over / c_under ratio rather than picked by hand.

ESCALATE isn't a soft action here; it comes only from the hard safety floor.
"""
from __future__ import annotations

from .config import Costs
from .posterior import PreferencePosterior
from .types import AutonomyLevel

_SOFT = (
    AutonomyLevel.PROCEED_SILENTLY,
    AutonomyLevel.PROCEED_AND_NOTIFY,
    AutonomyLevel.ASK_FIRST,
)


def expected_losses(probs: dict[AutonomyLevel, float], costs: Costs) -> dict[AutonomyLevel, float]:
    out = {}
    for action in _SOFT:
        out[action] = sum(
            p * costs.loss(int(action), int(ceiling))
            for ceiling, p in probs.items()
        )
    return out


def best_action(probs: dict[AutonomyLevel, float], costs: Costs) -> tuple[AutonomyLevel, dict]:
    losses = expected_losses(probs, costs)
    # ties break toward more human involvement (the safer side).
    best = min(_SOFT, key=lambda a: (losses[a], -int(a)))
    return best, losses


class DecisionPolicy:
    """Wraps the posterior and turns a context into a proposed soft level."""

    def __init__(self, costs: Costs, posterior: PreferencePosterior | None = None) -> None:
        self.costs = costs
        self.post = posterior or PreferencePosterior()

    def propose(self, key) -> tuple[AutonomyLevel, dict]:
        probs = self.post.category_probs(key)
        return best_action(probs, self.costs)

    def learn(self, key, comfort_ceiling: AutonomyLevel) -> None:
        self.post.learn(key, comfort_ceiling)
