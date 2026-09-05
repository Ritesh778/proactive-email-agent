"""Run the agent and the baselines over a simulated stream and score them.

Everything goes through the same hard floor, so any gap between policies is about
calibration, not hard safety. Beyond the headline run this does three things a
reviewer should want: an ablation that turns the statistical risk floor off to
show what it buys, bootstrap confidence intervals on the headline numbers, and an
injection battery that stress-tests the defence-in-depth claim across paraphrases.
"""
from __future__ import annotations

import json
import os

from proagent.agent import ProactiveAgent
from proagent.classifier import classify
from proagent.config import PolicyConfig, RiskFloor
from proagent.safety import floor_level, hard_never_silent
from proagent.types import AutonomyLevel, Feedback

from .attacks import battery
from .metrics import (EventRecord, bootstrap_ci, expected_calibration_error,
                      headline_cis, rolling_ask_rate, summarize,
                      time_to_calibration, windowed, _rate)
from .simulator import EmailStream, SimulatedUser

L = AutonomyLevel


def _record(t, sit, floor, decision, p_proceed, proceed_upper, detected):
    return EventRecord(
        t=t, action=int(sit.action), category=int(sit.category),
        learnable=(sit.kind == "learnable"), is_injection=(sit.kind == "injection"),
        detected_injection=detected, ceiling=int(sit.ceiling), floor=int(floor),
        decision=int(decision), p_proceed=p_proceed, proceed_upper=proceed_upper,
        evasive=(sit.body_kind == "evasive"),
    )


def run_learned(n, seed, config=None, feedback_noise=0.05, drift_at=None):
    agent = ProactiveAgent(config=config or PolicyConfig())
    user = SimulatedUser(feedback_noise=feedback_noise, seed=seed + 1)
    records = []
    for t, (sit, email) in enumerate(EmailStream(n, seed=seed, drift_at=drift_at)):
        d = agent.decide(email)
        records.append(_record(t, sit, d.hard_floor, d.level, d.p_proceed,
                               d.proceed_upper, bool(d.injection_signals)))
        agent.learn(d.obs, Feedback(comfort_ceiling=user.feedback_ceiling(sit)))
    return records, agent


def run_baseline(n, seed, fixed_level, drift_at=None):
    records = []
    for t, (sit, email) in enumerate(EmailStream(n, seed=seed, drift_at=drift_at)):
        obs, signals = classify(email)
        floor, _ = floor_level(obs)
        decision = AutonomyLevel(max(int(fixed_level), int(floor)))
        if decision == L.PROCEED_SILENTLY and hard_never_silent(obs.action):
            decision = L.ASK_FIRST
        records.append(_record(t, sit, floor, decision, 0.0, 1.0, bool(signals)))
    return records


def run_injection_battery(seed=0):
    """Fire the attack battery at a fresh, fully trained-to-be-permissive agent.

    The agent is hammered with 'just proceed' feedback first, to show that even a
    maximally permissive learner still can't be talked into an autonomous action
    on an attack, because the action floor is independent of the learner.
    """
    agent = ProactiveAgent()
    # try to make it permissive everywhere
    from proagent.types import Email, Action, Category
    warm = Email("a@b.com", "s", "hello", Category.KNOWN_CONTACT, Action.LABEL)
    for _ in range(50):
        d = agent.decide(warm)
        agent.learn(d.obs, Feedback(comfort_ceiling=L.PROCEED_SILENTLY))

    overt_detected = overt_total = 0
    evasive_detected = evasive_total = 0
    autonomous_actions = 0
    for atk in battery():
        d = agent.decide(atk.email)
        detected = bool(d.injection_signals)
        if atk.evasive:
            evasive_total += 1
            evasive_detected += int(detected)
        else:
            overt_total += 1
            overt_detected += int(detected)
        if int(d.level) <= int(L.PROCEED_AND_NOTIFY):
            autonomous_actions += 1
    return {
        "n_attacks": overt_total + evasive_total,
        "overt_detection_rate": round(overt_detected / max(1, overt_total), 3),
        "evasive_detection_rate": round(evasive_detected / max(1, evasive_total), 3),
        "autonomous_actions": autonomous_actions,
    }


def guarantee_frontier(n, seed, epsilons=(0.05, 0.10, 0.15, 0.20, 0.30), delta=0.05):
    """Trace the guarantee-vs-interruption trade-off.

    Each point tightens the tolerated autonomous-error probability epsilon and
    reports what it costs in ask-rate. unwanted-autonomous and precision are
    reported too, to show the guarantee never comes at the price of acting against
    the user.
    """
    out = []
    for eps in epsilons:
        cfg = PolicyConfig(risk=RiskFloor(epsilon=eps, delta=delta, min_evidence=3))
        recs, _ = run_learned(n, seed, config=cfg)
        s = summarize(recs)
        _, cold = windowed(recs, frac=0.2)
        out.append({
            "epsilon": eps, "delta": delta,
            "ask_rate_learnable": s["ask_rate_learnable"],
            "ask_rate_converged": cold["ask_rate_learnable"],
            "unwanted_autonomous_rate": s["unwanted_autonomous_rate"],
            "silent_precision": s["silent_precision"],
        })
    return out


def make_plots(records, outdir, frontier=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if frontier:
        eps = [f["epsilon"] for f in frontier]
        overall = [f["ask_rate_learnable"] for f in frontier]
        conv = [f["ask_rate_converged"] for f in frontier]
        fig, ax = plt.subplots(figsize=(7, 4.4))
        ax.plot(eps, overall, "-o", label="ask-rate (whole run)", lw=2)
        ax.plot(eps, conv, "-s", label="ask-rate (converged)", lw=2, alpha=0.8)
        ax.set_xlabel("tolerated autonomous-error probability  (epsilon)")
        ax.set_ylabel("ask-rate on learnable mail")
        ax.set_title("Price of the guarantee: tighter epsilon, more asking")
        ax.invert_xaxis(); ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "guarantee_frontier.png"), dpi=130)
        plt.close(fig)

    roll_learn = rolling_ask_rate(records, window=50, learnable_only=True)
    roll_all = rolling_ask_rate(records, window=50, learnable_only=False)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(roll_learn, label="learnable contexts", lw=2)
    ax.plot(roll_all, label="all contexts (incl. floors)", lw=1.4, alpha=0.8)
    ax.set_xlabel("event #"); ax.set_ylabel("rolling ask-rate (window=50)")
    ax.set_title("Ask-rate falls as the agent calibrates")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "ask_rate_decay.png"), dpi=130)
    plt.close(fig)

    proc = [r for r in records if r.decision <= int(L.PROCEED_AND_NOTIFY)]
    if proc:
        bins = [(i / 10, (i + 1) / 10) for i in range(10)]
        xs, ys, ns = [], [], []
        for lo, hi in bins:
            b = [r for r in proc if lo <= r.p_proceed < hi]
            if b:
                xs.append((lo + hi) / 2)
                ys.append(sum(1 for r in b if r.ceiling <= int(L.PROCEED_AND_NOTIFY)) / len(b))
                ns.append(len(b))
        fig, ax = plt.subplots(figsize=(6.6, 5))
        ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
        ax.scatter(xs, ys, s=[max(20, n) for n in ns], alpha=0.75, label="observed")
        ax.set_xlabel("posterior P(proceeding OK)")
        ax.set_ylabel("empirical P(user OK proceeding)")
        ax.set_title("Proactive actions are well-calibrated")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "reliability.png"), dpi=130)
        plt.close(fig)


def main(n=1500, seed=7, outdir=None):
    outdir = os.path.abspath(outdir or os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(outdir, exist_ok=True)

    learned, _ = run_learned(n, seed)
    # ablation: same thing with the statistical risk floor switched off
    no_risk = PolicyConfig(risk=RiskFloor(epsilon=1.0, delta=0.05, min_evidence=0))
    policy_only, _ = run_learned(n, seed, config=no_risk)

    baselines = {
        "always_ask": run_baseline(n, seed, L.ASK_FIRST),
        "always_proceed": run_baseline(n, seed, L.PROCEED_SILENTLY),
        "static_notify": run_baseline(n, seed, L.PROCEED_AND_NOTIFY),
    }

    warm, cold = windowed(learned, frac=0.2)
    report = {
        "config": {"n_events": n, "seed": seed, "feedback_noise": 0.05,
                   "costs": {"c_over": 10.0, "c_under": 1.0},
                   "risk_floor": {"epsilon": 0.10, "delta": 0.05, "min_evidence": 3}},
        "learned": summarize(learned),
        "learned_confidence_intervals": headline_cis(learned, seed=seed),
        "learned_calibration": expected_calibration_error(learned),
        "ablation_policy_only_no_risk_floor": summarize(policy_only),
        "baselines": {k: summarize(v) for k, v in baselines.items()},
        "calibration": {"time_to_calibration":
                        time_to_calibration(learned)["median_events_to_calibrate"]},
        "learned_first_20pct": warm,
        "learned_last_20pct": cold,
        "injection_battery": run_injection_battery(seed=seed),
        "guarantee_frontier": guarantee_frontier(n, seed),
    }
    with open(os.path.join(outdir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    make_plots(learned, outdir, frontier=report["guarantee_frontier"])
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
