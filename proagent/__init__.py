"""A proactive email agent that calibrates its autonomy under two floors.

A deterministic safety floor and a statistical risk floor bound a
decision-theoretic policy, so the agent asks less as it learns without ever being
able to learn past a hard safety limit.
"""
from .agent import ProactiveAgent
from .config import Costs, PolicyConfig, RiskFloor
from .policy import DecisionPolicy
from .posterior import PreferencePosterior
from .risk import RiskCertifier, clopper_pearson_upper
from .safety import floor_level, hard_never_silent
from .types import (
    Action,
    AutonomyLevel,
    Category,
    Decision,
    Email,
    Feedback,
    Observation,
    RiskFeatures,
)

__all__ = [
    "ProactiveAgent",
    "PolicyConfig", "Costs", "RiskFloor",
    "DecisionPolicy", "PreferencePosterior",
    "RiskCertifier", "clopper_pearson_upper",
    "floor_level", "hard_never_silent",
    "Action", "AutonomyLevel", "Category", "Decision", "Email", "Feedback",
    "Observation", "RiskFeatures",
]
