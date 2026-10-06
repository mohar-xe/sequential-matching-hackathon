"""Validate the recovered probability model against the real simulator."""
import itertools
import math
import random
import statistics
import sys


import _bootstrap  # noqa: F401  (sets sys.path portably)
from kit import Simulator, generate, eligibility, HARD, baseline_asks, baseline_match
from prob_model import p_msmi_truth, fit_term, e_shared_prod, sigmoid, _truth_feasible
from ceiling import matchable


def check_quadrature():
    """Compare the 21-node rule against a dense trapezoid integral."""
    def dense(za, zb, sig=0.45, n=4001):
        lo, hi = -6 * sig, 6 * sig
        h = (hi - lo) / (n - 1)
        tot = 0.0
        for i in range(n):
            x = lo + i * h
            g = math.exp(-x * x / (2 * sig * sig)) / (sig * math.sqrt(2 * math.pi))
            tot += g * sigmoid(za + x) * sigmoid(zb + x) * h
        return tot
    worst = 0.0
    for za in (-3, -1.5, -0.5, 0, 0.5, 1.5, 3):
        for zb in (-3, -1.5, -0.5, 0, 0.5, 1.5, 3):
            q = e_shared_prod(za, zb)
            d = dense(za, zb)
            worst = max(worst, abs(q - d))
    print('quadrature max abs error over grid: %.2e' % worst)


def check_pair_level(variant='development', seed=41, npairs=60, reps=40):
    """Replay specific pairs one at a time through the real simulator."""
    world = generate(seed, 200, 'eval', variant)
    ms = [m for m in world['members'] if matchable(m)]
    pairs = [(a, b) for a, b in itertools.combinations(ms, 2) if _truth_feasible(a, b)]
    random.Random(0).shuffle(pairs)
    pairs = pairs[:npairs]

    obs, pred = [], []
    for pair_index, (a, b) in enumerate(pairs):
        sub = []
        for m in (a, b):
            m = dict(m)
            m['arrived_day'] = 0
            m['exit_day'] = 10000
            for k in HARD:
                m['fields'][k] = m['truth'][k]
                m['field_status'][k] = 'observed'
            sub.append(m)
        for rep in range(reps):
            # rand_for() is deterministic in (world seed, a, b, day), so each
            # repetition must use a fresh world seed or we replay one outcome.
            sw = dict(world)
            sw['seed'] = seed * 100003 + rep * 7919 + pair_index
            sw['members'] = sub
            sim = Simulator(sw)
            sim.advance([[a['member_id'], b['member_id']]])
            for _ in range(40):
                sim.advance([])
            ev = sim.receive_feedback()
            es = [e for e in ev if e['event'] != 'pause_after_mutual_interest']
            rs = [e for e in es if e['event'] == 'introduction_response']
            ds = [e for e in es if e['event'] == 'date_happened' and e['value'] is True]
            ss = [e for e in es if e['event'] == 'second_meeting_intention']
            ok = bool(ds) and ds[0]['occurred_day'] <= 30 \
                and len(ss) == 2 and all(e['value'] == 'yes' for e in ss) \
                and all(e['occurred_day'] - ds[0]['occurred_day'] <= 3 for e in ss)
            obs.append(1.0 if ok else 0.0)
            pred.append(p_msmi_truth(a, b, day=0, variant=variant, full=True))
    n = len(obs)
    emp = statistics.mean(obs)
    pr = statistics.mean(pred)
    corr = statistics.correlation(obs, pred)
    print('%-12s %d draws: empirical P(MSMI)=%.4f  model=%.4f  corr=%.3f'
          % (variant, n, emp, pr, corr))
    lo = emp - 1.96 * math.sqrt(max(emp * (1 - emp), 1e-9) / n)
    hi = emp + 1.96 * math.sqrt(max(emp * (1 - emp), 1e-9) / n)
    print('   95%% CI [%.4f, %.4f] -> model %s' % (max(0, lo), min(1, hi),
                                                   'INSIDE' if lo <= pr <= hi else 'OUTSIDE'))
    # high vs low predicted decile
    z = sorted(zip(pred, obs))
    k = max(1, n // 10)
    print('   top decile predicted P=%.4f -> empirical %.4f | bottom decile P=%.4f -> %.4f'
          % (statistics.mean(p for p, _ in z[-k:]),
             statistics.mean(o for _, o in z[-k:]),
             statistics.mean(p for p, _ in z[:k]),
             statistics.mean(o for _, o in z[:k])))


if __name__ == '__main__':
    check_quadrature()
    for v in ('development', 'shift', 'drift'):
        check_pair_level(v)