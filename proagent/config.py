"""Configuration for the agent.

Everything the policy can be tuned on lives here, so nothing downstream carries a
magic number. Two groups: the loss matrix the decision-theoretic policy minimizes
against, and the parameters for the statistical risk floor.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Costs:
    """Asymmetric loss for the autonomy decision.

    The whole behaviour of the policy comes from one asymmetry: acting more
    autonomously than the user wanted is much worse than asking when you didn't
    need to. c_over is the per-level penalty for over-autonomy (the agent acted
    when the user wanted to be asked); c_under is the per-level penalty for
    over-caution (the agent asked when the user was fine with it handling things).

    Keeping c_over >> c_under is what makes the agent conservative, and the ratio
    is what the old hand-tuned thresholds were secretly encoding. Now it's
    explicit and you can point at it.
    """

    c_over: float = 10.0    # penalty per level of acting too autonomously
    c_under: float = 1.0    # penalty per level of asking when it wasn't needed

    def loss(self, action_level: int, true_ceiling_level: int) -> float:
        # levels are ordered by human involvement (SILENT=0 .. ESCALATE=3), so a
        # lower level is more autonomous. Acting below the user's ceiling means we
        # acted more freely than they wanted, which is the expensive mistake.
        gap = true_ceiling_level - action_level
        if gap > 0:
            return self.c_over * gap         # acted too autonomously
        return self.c_under * (-gap)         # asked/escalated more than needed


@dataclass(frozen=True)
class RiskFloor:
    """Parameters for the statistical risk floor.

    Before the agent is allowed to act autonomously in a context, the observed
    feedback for that context has to certify that acting is safe: a finite-sample
    upper bound on the chance the user would object has to sit at or below
    epsilon, at confidence 1 - delta.
    """

    epsilon: float = 0.10   # tolerated probability of an unwanted autonomous act
    delta: float = 0.05     # the certificate holds with probability >= 1 - delta
    min_evidence: int = 3   # never certify on fewer than this many observations


@dataclass(frozen=True)
class PolicyConfig:
    costs: Costs = Costs()
    risk: RiskFloor = RiskFloor()
    forgetting: float = 1.0   # <1 down-weights old feedback for drift
