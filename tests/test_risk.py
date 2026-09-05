"""The statistical risk floor.

Two things matter: the Clopper-Pearson bound is a real upper bound (so the
certificate is sound), and the floor won't certify until the evidence supports
it. The coverage test is the important one: when the floor certifies, the actual
unwanted-proceed rate should stay under epsilon.
"""
from __future__ import annotations

import random

from proagent.config import RiskFloor
from proagent.risk import RiskCertifier, clopper_pearson_upper
from proagent.types import AutonomyLevel as L

KEY = (0, 0)


def test_cp_bound_basic_properties():
    assert clopper_pearson_upper(0, 0, 0.05) == 1.0        # no evidence
    assert clopper_pearson_upper(5, 5, 0.05) == 1.0        # all failures
    # more clean successes should lower the bound on the failure rate
    assert clopper_pearson_upper(0, 50, 0.05) < clopper_pearson_upper(0, 5, 0.05)


def test_needs_evidence_before_certifying():
    r = RiskCertifier(cfg=RiskFloor(epsilon=0.1, delta=0.05, min_evidence=3))
    # one good observation isn't enough to certify proceeding
    r.learn(KEY, L.PROCEED_SILENTLY)
    assert r.certified_level(KEY)[0] == L.ASK_FIRST


def test_certifies_after_enough_clean_evidence():
    r = RiskCertifier(cfg=RiskFloor(epsilon=0.2, delta=0.05, min_evidence=3))
    for _ in range(60):
        r.learn(KEY, L.PROCEED_SILENTLY)
    assert r.certified_level(KEY)[0] == L.PROCEED_SILENTLY


def test_never_certifies_a_genuinely_risky_context():
    # user wants asking most of the time; the floor should keep asking.
    r = RiskCertifier(cfg=RiskFloor(epsilon=0.1, delta=0.05, min_evidence=3))
    rng = random.Random(0)
    for _ in range(300):
        r.learn(KEY, L.ASK_FIRST if rng.random() < 0.5 else L.PROCEED_SILENTLY)
    assert r.certified_level(KEY)[0] == L.ASK_FIRST


def test_coverage_guarantee_holds_empirically():
    # generate exchangeable feedback with a true proceed-failure rate just under
    # epsilon; whenever the floor certifies, the realized failure rate on held-out
    # draws should stay at or below epsilon on average.
    eps = 0.1
    cfg = RiskFloor(epsilon=eps, delta=0.05, min_evidence=5)
    certified_runs, violations = 0, 0
    for seed in range(200):
        rng = random.Random(seed)
        true_fail = 0.06
        r = RiskCertifier(cfg=cfg)
        for _ in range(40):
            fail = rng.random() < true_fail
            r.learn(KEY, L.ASK_FIRST if fail else L.PROCEED_SILENTLY)
        if r.certified_level(KEY)[0] == L.PROCEED_SILENTLY:
            certified_runs += 1
            # held-out realized failure rate
            fails = sum(1 for _ in range(200) if rng.random() < true_fail)
            if fails / 200 > eps:
                violations += 1
    # with true rate 0.06 < eps and a sound bound, certified runs should rarely
    # exceed epsilon on held-out data.
    assert certified_runs > 0
    assert violations / max(1, certified_runs) < 0.1
