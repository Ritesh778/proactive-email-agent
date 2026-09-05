"""The decision-theoretic policy: it should match the user's preference in the
clean case and stay conservative when it's unsure, because over-autonomy costs
more than over-caution."""
from __future__ import annotations

from proagent.config import Costs
from proagent.policy import best_action, expected_losses
from proagent.types import AutonomyLevel as L


def _p(silent, notify, ask):
    return {L.PROCEED_SILENTLY: silent, L.PROCEED_AND_NOTIFY: notify, L.ASK_FIRST: ask}


def test_matches_clean_preference():
    c = Costs()
    assert best_action(_p(1, 0, 0), c)[0] == L.PROCEED_SILENTLY
    assert best_action(_p(0, 1, 0), c)[0] == L.PROCEED_AND_NOTIFY
    assert best_action(_p(0, 0, 1), c)[0] == L.ASK_FIRST


def test_conservative_under_uncertainty():
    # a coin flip between silent and ask should ask, not act.
    c = Costs()
    assert best_action(_p(0.5, 0, 0.5), c)[0] == L.ASK_FIRST


def test_cost_ratio_sets_the_effective_threshold():
    # with a 10:1 ratio, even 80% confidence in silent isn't enough to act.
    strict = Costs(c_over=10, c_under=1)
    assert best_action(_p(0.8, 0, 0.2), strict)[0] == L.ASK_FIRST
    # flatten the ratio and the same belief now proceeds.
    loose = Costs(c_over=2, c_under=1)
    assert best_action(_p(0.8, 0, 0.2), loose)[0] == L.PROCEED_SILENTLY


def test_over_autonomy_costs_more_than_over_caution():
    c = Costs()
    losses = expected_losses(_p(0, 1, 0), c)  # true pref is NOTIFY
    # acting silently (too autonomous) should cost more than asking (too cautious)
    assert losses[L.PROCEED_SILENTLY] > losses[L.ASK_FIRST]
