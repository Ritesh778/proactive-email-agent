"""End to end: it learns to stop asking on safe repeated contexts, keeps asking
where the user actually wants it, and never breaks the floor on the way."""
from __future__ import annotations

from proagent import (Action, AutonomyLevel, Category, Email, Feedback,
                      ProactiveAgent)

L = AutonomyLevel


def _email(action, category, external=False, body="hello"):
    return Email(sender="x@y.com", subject="s", body=body, category=category,
                 proposed_action=action, external_domain=external)


def _train(agent, email, ceiling, n):
    for _ in range(n):
        d = agent.decide(email)
        agent.learn(d.obs, Feedback(comfort_ceiling=ceiling))
    return agent.decide(email)


def test_newsletter_archive_becomes_silent():
    agent = ProactiveAgent()
    email = _email(Action.ARCHIVE, Category.NEWSLETTER)
    assert agent.decide(email).level == L.ASK_FIRST  # cautious to start
    assert _train(agent, email, L.PROCEED_SILENTLY, 40).level == L.PROCEED_SILENTLY


def test_manager_draft_keeps_asking():
    """A real ASK preference with no safety rule behind it. It shouldn't
    over-automate this even after a lot of interaction."""
    agent = ProactiveAgent()
    email = _email(Action.DRAFT_REPLY, Category.MANAGER)
    assert _train(agent, email, L.ASK_FIRST, 50).level == L.ASK_FIRST


def test_known_contact_draft_becomes_notify():
    agent = ProactiveAgent()
    email = _email(Action.DRAFT_REPLY, Category.KNOWN_CONTACT)
    assert _train(agent, email, L.PROCEED_AND_NOTIFY, 50).level == L.PROCEED_AND_NOTIFY


def test_learning_one_context_does_not_leak_to_another():
    agent = ProactiveAgent()
    _train(agent, _email(Action.ARCHIVE, Category.NEWSLETTER), L.PROCEED_SILENTLY, 40)
    other = _email(Action.DRAFT_REPLY, Category.KNOWN_CONTACT)
    assert agent.decide(other).level == L.ASK_FIRST  # different context, still cautious


def test_bound_by_is_reported():
    agent = ProactiveAgent()
    d = agent.decide(_email(Action.PAY_INVOICE, Category.VENDOR_INVOICE, external=True))
    assert d.level == L.ESCALATE and d.bound_by == "hard"
