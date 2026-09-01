"""Data types for the agent. Plain containers, no behaviour."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class AutonomyLevel(IntEnum):
    """Autonomy levels, ordered by how much the human is involved.

    The ordering is load-bearing: because ESCALATE > ASK_FIRST > ... we can
    combine several sources of caution (the hard floor, the risk floor, the
    policy) with max(), and the most cautious one always wins. That's the reason
    no floor can be learned away.
    """

    PROCEED_SILENTLY = 0      # just do it
    PROCEED_AND_NOTIFY = 1    # do it, then say what happened
    ASK_FIRST = 2             # propose and wait for a yes/no
    ESCALATE = 3              # don't act, hand it back to the human

    @property
    def label(self) -> str:
        return self.name


class Action(IntEnum):
    """What the agent is thinking about doing to an email.

    Risk properties for each action live in classifier.py. The learner never
    touches these; it only proposes a level for an (action, category) pair.
    """

    ARCHIVE = 0
    LABEL = 1
    MARK_READ = 2
    DRAFT_REPLY = 3          # writes a draft, doesn't send
    SEND_REPLY = 4           # goes to an external recipient
    FORWARD_INTERNAL = 5
    FORWARD_EXTERNAL = 6     # external + hard to take back
    SCHEDULE_MEETING = 7     # writes to the calendar, invites people
    UNSUBSCRIBE = 8
    DELETE_PERMANENT = 9     # irreversible
    PAY_INVOICE = 10         # money, irreversible, external
    SHARE_CREDENTIAL = 11    # never automated, ever


class Category(IntEnum):
    """Sender/topic bucket. This is the context the learner keys on."""

    NEWSLETTER = 0
    KNOWN_CONTACT = 1
    INTERNAL_COLLEAGUE = 2
    MANAGER = 3
    UNKNOWN_EXTERNAL = 4
    VENDOR_INVOICE = 5
    CALENDAR_INVITE = 6
    RECRUITER = 7
    SECURITY_ALERT = 8


@dataclass(frozen=True)
class Email:
    sender: str
    subject: str
    body: str
    category: Category
    proposed_action: Action   # what the upstream mail rules want to do
    external_domain: bool = False


@dataclass(frozen=True)
class RiskFeatures:
    """The boolean flags the safety floor looks at.

    Computed from the action and the email text. The floor reads nothing else,
    which keeps it short enough to check by eye.
    """

    irreversible: bool = False
    external_send: bool = False
    money: bool = False
    injection_suspected: bool = False
    credential_request: bool = False

    def any_hard(self) -> bool:
        return any((
            self.irreversible,
            self.external_send,
            self.money,
            self.injection_suspected,
            self.credential_request,
        ))


@dataclass(frozen=True)
class Observation:
    email: Email
    action: Action
    category: Category
    risk: RiskFeatures

    @property
    def context_key(self) -> tuple[int, int]:
        return (int(self.action), int(self.category))


@dataclass
class Decision:
    """The agent's output for one email, with everything needed to explain it."""

    level: AutonomyLevel
    hard_floor: AutonomyLevel
    risk_floor: AutonomyLevel
    policy_choice: AutonomyLevel
    rationale: str
    obs: Observation
    bound_by: str                      # "hard" | "risk" | "policy"
    expected_losses: dict = field(default_factory=dict)
    p_proceed: float = 0.0             # posterior mean P(proceeding is OK)
    proceed_upper: float = 1.0         # risk-floor CP bound on unwanted proceed
    silent_upper: float = 1.0          # risk-floor CP bound on unwanted silent
    injection_signals: list[str] = field(default_factory=list)


@dataclass
class Feedback:
    """What the user thought of a decision.

    comfort_ceiling is the most autonomous level they'd actually have been fine
    with here. In production you'd infer it from undo / thumbs / "you could've
    just handled that". In the eval it's the simulated user's hidden preference,
    sometimes with noise on top.
    """

    comfort_ceiling: AutonomyLevel
    corrected: bool = False
