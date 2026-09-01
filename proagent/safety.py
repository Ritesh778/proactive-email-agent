"""The safety floor: the minimum human involvement a situation requires.

No state, no learning. It maps risk flags to the lowest autonomy level the
agent is allowed to use. agent.py combines it with the learned proposal via
max(proposal, floor), so this can only ever raise caution, never lower it.

The test in tests/test_safety.py pins the invariant: for any observation and
any learner state, the emitted level is >= floor_level(obs). No feedback
sequence can break that.
"""
from __future__ import annotations

from .types import Action, AutonomyLevel, Observation


def floor_level(obs: Observation) -> tuple[AutonomyLevel, str]:
    """Minimum allowed autonomy level, plus a reason string for the audit trail.

    Rules are checked worst-first, so the first match wins.
    """
    r = obs.risk

    if r.credential_request:
        return AutonomyLevel.ESCALATE, "credential/secret action is never automated"

    if r.injection_suspected:
        return AutonomyLevel.ESCALATE, "possible prompt injection, handing to human"

    if r.money:
        return AutonomyLevel.ESCALATE, "money movement needs an explicit human decision"

    if r.irreversible:
        return AutonomyLevel.ASK_FIRST, "irreversible, confirm before acting"

    if r.external_send:
        return AutonomyLevel.ASK_FIRST, "leaves the org, confirm before acting"

    return AutonomyLevel.PROCEED_SILENTLY, "no hard-safety constraint"


# Actions that must never run silently, checked separately in the agent. This
# overlaps with floor_level on purpose: if assess_risk ever has a bug, this
# still stops a dangerous action from going out silently.
_NEVER_SILENT = {
    Action.SEND_REPLY,
    Action.FORWARD_EXTERNAL,
    Action.SCHEDULE_MEETING,
    Action.DELETE_PERMANENT,
    Action.PAY_INVOICE,
    Action.SHARE_CREDENTIAL,
}


def hard_never_silent(action: Action) -> bool:
    return action in _NEVER_SILENT
