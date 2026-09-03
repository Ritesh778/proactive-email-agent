"""The agent.

Pipeline per email:
    classify -> (hard floor, risk floor, policy) -> combine -> act -> learn

Three sources each propose an autonomy level, and the final decision is the most
cautious of them:

    level = max(hard_floor, risk_floor, policy_choice)

  - hard_floor  deterministic rules (money, injection, irreversible, external).
                Can't be learned around.
  - risk_floor  a finite-sample certificate that acting autonomously is safe in
                this context. Blocks autonomy until the evidence earns it.
  - policy      the expected-cost-minimizing soft action given the posterior.

Because AutonomyLevel is ordered by human involvement, max() means either floor
can raise caution but neither the policy nor any feedback can lower it below a
floor.
"""
from __future__ import annotations

from typing import Callable, Optional

from . import audit
from .config import PolicyConfig
from .classifier import classify as rule_classify
from .policy import DecisionPolicy
from .posterior import PreferencePosterior
from .risk import RiskCertifier
from .safety import floor_level, hard_never_silent
from .types import (
    AutonomyLevel,
    Decision,
    Email,
    Feedback,
    Observation,
)


class ProactiveAgent:
    def __init__(
        self,
        config: Optional[PolicyConfig] = None,
        classifier: Callable[[Email], tuple[Observation, list[str]]] = rule_classify,
    ) -> None:
        self.cfg = config or PolicyConfig()
        self.post = PreferencePosterior(forgetting=self.cfg.forgetting)
        self.policy = DecisionPolicy(self.cfg.costs, self.post)
        self.risk = RiskCertifier(cfg=self.cfg.risk, forgetting=self.cfg.forgetting)
        self.classify = classifier

    def decide(self, email: Email) -> Decision:
        obs, injection_signals = self.classify(email)
        key = obs.context_key

        hard, hard_reason = floor_level(obs)
        risk_level, u_proceed, u_silent = self.risk.certified_level(key)
        policy_choice, losses = self.policy.propose(key)
        probs = self.post.category_probs(key)
        p_proceed = probs[AutonomyLevel.PROCEED_SILENTLY] + probs[AutonomyLevel.PROCEED_AND_NOTIFY]

        level = AutonomyLevel(max(int(hard), int(risk_level), int(policy_choice)))

        # independent guard: never let a never-silent action slip out silently,
        # even if some upstream flag was wrong.
        if level == AutonomyLevel.PROCEED_SILENTLY and hard_never_silent(obs.action):
            level = AutonomyLevel.ASK_FIRST
            hard_reason = f"{hard_reason}; action is on the never-silent list"

        bound_by = self._bound_by(level, hard, risk_level, policy_choice)
        rationale = self._explain(bound_by, hard, hard_reason, risk_level,
                                  policy_choice, u_proceed, u_silent, self.cfg.risk.epsilon)

        d = Decision(
            level=level, hard_floor=hard, risk_floor=risk_level,
            policy_choice=policy_choice, rationale=rationale, obs=obs,
            bound_by=bound_by, expected_losses={k.label: round(v, 3) for k, v in losses.items()},
            p_proceed=p_proceed, proceed_upper=u_proceed, silent_upper=u_silent,
            injection_signals=injection_signals,
        )
        audit.record({
            "context": key, "action": obs.action.name, "category": obs.category.name,
            "hard_floor": hard.label, "risk_floor": risk_level.label,
            "policy": policy_choice.label, "decision": level.label,
            "bound_by": bound_by, "proceed_upper": round(u_proceed, 3),
            "injection_signals": injection_signals,
        })
        return d

    def learn(self, obs: Observation, feedback: Feedback) -> None:
        # Feedback only ever touches the posterior and the risk buffer. The hard
        # floor is stateless, so there's no path from feedback to it.
        self.policy.learn(obs.context_key, feedback.comfort_ceiling)
        self.risk.learn(obs.context_key, feedback.comfort_ceiling)

    @staticmethod
    def _bound_by(level, hard, risk_level, policy_choice) -> str:
        # what actually set the level? if the policy got its choice, it's the
        # policy; otherwise a floor raised caution, attribute to whichever reached
        # the final level (hard takes precedence on a tie).
        if level == policy_choice:
            return "policy"
        if level == hard and hard >= risk_level:
            return "hard"
        return "risk"

    @staticmethod
    def _explain(bound_by, hard, hard_reason, risk_level, policy_choice,
                 u_proceed, u_silent, epsilon) -> str:
        cert = f"[risk cert: P(unwanted proceed) <= {u_proceed:.2f} vs eps={epsilon:.2f}]"
        if bound_by == "hard":
            return f"hard floor {hard.label} ({hard_reason}) set the outcome. {cert}"
        if bound_by == "risk":
            return (f"risk floor held autonomy to {risk_level.label}: not yet "
                    f"certified safe to proceed. {cert}")
        return (f"policy chose {policy_choice.label} by expected cost, within both "
                f"floors. {cert}")
