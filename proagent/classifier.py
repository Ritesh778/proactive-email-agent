"""Perception: turn an Email into an Observation.

Two deterministic questions:
  1. What risk does the proposed action carry (irreversible / external / money /
     credential)?
  2. Does the text look like a prompt-injection attempt?

Both feed the floor. Keeping it deterministic means the eval runs with no API
key and every flag traces back to a specific rule. There's an optional LLM
version in llm.py with the same signature; it can only add risk flags, never
remove one the rules found.
"""
from __future__ import annotations

import re

from .types import Action, Email, Observation, RiskFeatures

# Intrinsic properties of each action. These don't change with learning.
_IRREVERSIBLE = {Action.DELETE_PERMANENT, Action.PAY_INVOICE, Action.FORWARD_EXTERNAL}
_EXTERNAL_SEND = {
    Action.SEND_REPLY,
    Action.FORWARD_EXTERNAL,
    Action.PAY_INVOICE,
    Action.SCHEDULE_MEETING,  # fires invites at other people
}
_MONEY = {Action.PAY_INVOICE}
_CREDENTIAL = {Action.SHARE_CREDENTIAL}


# Injection patterns. Kept conservative. Detection is only the first line of
# defence anyway: even a missed injection can't cause autonomous harm, because
# whatever action it's pushing (send / money / delete) is floored on its own.
_INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("override_instructions", re.compile(r"ignore (all |previous |the )?(above |prior )?instructions", re.I)),
    ("role_spoof", re.compile(r"^\s*(system|assistant|developer)\s*:", re.I | re.M)),
    ("exfiltration", re.compile(r"(forward|send|email|cc)\b.{0,40}\b(all|every|entire).{0,20}(inbox|emails|messages)", re.I)),
    ("secret_request", re.compile(r"\b(password|api[ _-]?key|credential|mfa|2fa|otp|seed phrase|private key)\b", re.I)),
    ("urgency_coercion", re.compile(r"\b(do not (tell|ask|notify)|without (asking|confirming)|immediately wire)\b", re.I)),
    ("rule_rewrite", re.compile(r"\byou (are|must) now\b.{0,30}\b(agent|assistant|allowed|permitted)\b", re.I)),
    ("fake_authority", re.compile(r"\bas an authorized (admin|administrator|it|security)\b", re.I)),
]


def detect_injection(text: str) -> list[str]:
    """Names of any injection signals in text. Empty list if none."""
    signals = []
    for name, pat in _INJECTION_PATTERNS:
        if pat.search(text):
            signals.append(name)
    return signals


def assess_risk(action: Action, email: Email, injection_signals: list[str]) -> RiskFeatures:
    external = action in _EXTERNAL_SEND or (
        action == Action.SEND_REPLY and email.external_domain
    )
    return RiskFeatures(
        irreversible=action in _IRREVERSIBLE,
        external_send=external,
        money=action in _MONEY,
        injection_suspected=bool(injection_signals),
        credential_request=action in _CREDENTIAL,
    )


def classify(email: Email) -> tuple[Observation, list[str]]:
    # signals returned alongside so the transcript can show what tripped.
    signals = detect_injection(f"{email.subject}\n{email.body}")
    action = email.proposed_action
    risk = assess_risk(action, email, signals)
    obs = Observation(email=email, action=action, category=email.category, risk=risk)
    return obs, signals
