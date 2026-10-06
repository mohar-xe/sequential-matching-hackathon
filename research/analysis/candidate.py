"""Candidate policy v1, scored against the three baselines on all six families.

Design rationale, all of it measured in the sibling scripts:

1. Volume is capped by the REACHABLE FEASIBLE EDGE SET (~103 edges in
   development, ~21 in sparse). Clarification bundles are supply-limited, not
   budget-limited, so spend them on every member that has no declined hard
   field. That is what the baseline already does; there is little to win here.
2. Value ordering is the only real lever: sorting reachable edges by P(MSMI)
   lifts the realised rate ~1.4x. The only reliable pair-level value signal
   is soft-field agreement, because per-member latents (bias, second_bias,
   response_rate) get at most 1-2 observations each and are therefore not
   learnable at this data rate.
3. Therefore: buy the four decision-relevant soft fields (1 unit each) for
   every reachable member, and score pairs by fit marginalised over the fields
   that were declined or never asked.
4. Learn the four fit weights ONLINE from revealed directional responses, so
   the policy adapts to the 'shift' family without being told the variant.
"""
import itertools
import math
import statistics
import sys


import _bootstrap  # noqa: F401  (sets sys.path portably)
from kit import Simulator, generate, eligibility, baseline_asks, baseline_match, HARD, SOFT, OPTIONS
from prob_model import run_episode

KEYF = ['relationship_goal', 'relationship_pace', 'lifestyle', 'conversations']
KS = {k: len(OPTIONS[k]) for k in KEYF}          # |S| per key field
BASE_W = {'relationship_goal': .7, 'relationship_pace': .4,
          'lifestyle': .25, 'conversations': .2}


def marginal_expected(w, k, va, vb):
    """E[match contribution] for one key field, marginalising unknowns.

    Observed+equal -> +w ; observed+different -> -w/2 ; unknown -> uniform
    prior over |S| options gives w*(1.5/|S| - 0.5).
    """
    if va is not None and vb is not None:
        return w if va == vb else -0.5 * w
    return w * (1.5 / KS[k] - 0.5)


def pair_fit(a, b, w):
    fa, fb = a['fields'], b['fields']
    return sum(marginal_expected(w[k], k, fa.get(k), fb.get(k)) for k in KEYF)


# ------------------------------------------------------------------ ask
def make_ask(soft=True):
    def ask(state, day):
        budget = state['ask_budget_remaining']
        mem = [m for m in state['members'] if m['available']]
        out = []
        # 1. constraint bundles first: they gate the feasible graph entirely
        hard = [m for m in mem
                if any(m['fields'].get(k) is None for k in HARD)
                and not any(m['field_status'][k] == 'declined' for k in HARD)]
        hard.sort(key=lambda m: (m['arrived_day'], m['member_id']))
        while budget >= 3 and hard:
            out.append({'member_id': hard.pop(0)['member_id'], 'field': 'constraints'})
            budget -= 3
        if not soft:
            return out
        # 2. spend what is left on the four decision-relevant soft fields
        softq = []
        for m in mem:
            for k in KEYF:
                if m['fields'].get(k) is None and m['field_status'][k] != 'declined':
                    softq.append((m['arrived_day'], m['member_id'], k))
        softq.sort()
        for _, mid, k in softq[:budget]:
            out.append({'member_id': mid, 'field': k})
        return out
    return ask


# ----------------------------------------------------------------- match
def make_match(weights, local=True):
    def match(state, day):
        w = weights
        mem = [m for m in state['members'] if m['available']]
        past = {tuple(sorted((i['user_a'], i['user_b']))) for i in state['introductions']}
        edges = []
        for a, b in itertools.combinations(mem, 2):
            key = tuple(sorted((a['member_id'], b['member_id'])))
            if key in past or eligibility(a, b)['status'] != 'feasible':
                continue
            edges.append((pair_fit(a, b, w) + 0.75, key))
        if not edges:
            return []
        edges.sort(key=lambda e: (-e[0], e[1]))
        used, out = set(), []
        for _, key in edges:
            if used.isdisjoint(key):
                out.append(list(key))
                used.update(key)
        return out
    return match


# ------------------------------------------------------- online weight fit
class Weights:
    """Online logistic regression on directional revealed preference.

    Feature for member x responding about partner y: agreement on the key
    fields where BOTH are observed. Label: recorded Yes. Partially observed
    pairs are used with the known subset (missing-data by restriction, which is
    unbiased for a correctly specified main-effects model).
    """

    def __init__(self, w0=None, lr=0.35, l2=0.02):
        self.w = dict(w0 or BASE_W)
        self.lr, self.l2 = lr, l2
        self.samples = 0

    def features(self, fa, fb):
        out = []
        for k in KEYF:
            va, vb = fa.get(k), fb.get(k)
            if va is not None and vb is not None:
                out.append((k, 1.0 if va == vb else -0.5))
        return out

    def update(self, obs_by_intro):
        for intro_id, actor, other, fdict in obs_by_intro:
            feats = self.features(actor, other)
            if not feats:
                continue
            z = -0.25 + sum(self.w[k] * x for k, x in feats)
            p = 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
            for k, x in feats:
                g = (p - fdict) * x + self.l2 * self.w[k]
                self.w[k] -= self.lr * g
            self.samples += 1


def harvest(state, world_by_id):
    """Pull (features, label) rows out of observable feedback only."""
    rows = []
    intro = {i['introduction_id']: i for i in state['introductions']}
    for e in state['feedback']:
        if e['event'] != 'introduction_response' or e.get('value') is None:
            continue
        i = intro.get(e['introduction_id'])
        if not i:
            continue
        by = world_by_id
        a, b = by.get(i['user_a']), by.get(i['user_b'])
        if a is None or b is None:
            continue
        actor, other = (a, b) if e['member_id'] == i['user_a'] else (b, a)
        y = 1.0 if e['value'] == 'yes' else 0.0
        rows.append((e['introduction_id'], actor['fields'], other['fields'], y))
    return rows


# ------------------------------------------------------------------- eval
def run(variant, seeds, policy, learn=False, verbose=False):
    rows = []
    for s in seeds:
        world = generate(s, 200, 'evaluation', variant)
        by = {m['member_id']: m for m in world['members']}
        wmodel = Weights()
        ask, match = policy(wmodel)
        step = {'d': 0}

        def mf(state, day, _w=wmodel, _by=by, _vm=variant):
            if learn and day > 0:
                wmodel.update(harvest(state, _by))
            return match(state, day)

        def af(state, day, _a=ask):
            return _a(state, day)

        r, _ = run_episode(world, af, mf)
        r['weights'] = dict(wmodel.w)
        rows.append(r)
    return rows


def policy_v1(wmodel):
    return make_ask(True), make_match(wmodel.w)


def policy_v1_fixed(wmodel):
    return make_ask(True), make_match(dict(BASE_W))


def policy_greedyish(wmodel):
    return make_ask(False), make_match(dict(BASE_W))


KEYS = ('assign_per_100', 'mutual_acceptance', 'msmi', 'msmi_per_100', 'coverage', 'ask_cost')


def show(name, rows):
    agg = {k: statistics.mean(r[k] for r in rows) for k in KEYS}
    ms = [r['msmi'] for r in rows]
    m = statistics.mean(ms)
    se = statistics.stdev(ms) / math.sqrt(len(ms)) if len(ms) > 1 else 0.0
    print('  %-26s assign=%5.1f mutual=%5.1f  MSMI=%5.2f +- %.2f  per100=%5.3f  cov=%.3f ask=%5.0f'
          % (name, agg['assign_per_100'], agg['mutual_acceptance'], m, se,
             agg['msmi_per_100'], agg['coverage'], agg['ask_cost']))


if __name__ == '__main__':
    import kit
    SEEDS = list(range(101, 113))
    VARIANTS = ['development', 'sparse', 'cold_start', 'delayed', 'shift', 'drift']
    finals = {}
    print('=== baselines (kit reference implementations) ===')
    for v in VARIANTS:
        rows = []
        for s in SEEDS:
            world = generate(s, 200, 'evaluation', v)

            def af(st, d):
                return baseline_asks(st)

            def mf(st, d):
                return baseline_match(st)
            r, _ = run_episode(world, af, mf)
            rows.append(r)
        show('greedy/' + v, rows)
        finals.setdefault('greedy', {})[v] = statistics.mean(r['msmi_per_100'] for r in rows)

    print()
    print('=== policy v1: soft acquisition + fixed dev weights ===')
    for v in VARIANTS:
        rows = run(v, SEEDS, policy_v1_fixed)
        show('v1-fixed/' + v, rows)
        finals.setdefault('v1fixed', {})[v] = statistics.mean(r['msmi_per_100'] for r in rows)

    print()
    print('=== policy v1 + ONLINE weight learning ===')
    for v in VARIANTS:
        rows = run(v, SEEDS, policy_v1, learn=True)
        show('v1-learn/' + v, rows)
        finals.setdefault('v1learn', {})[v] = statistics.mean(r['msmi_per_100'] for r in rows)
        print('       learned weights: %s' % {k: round(x, 3) for k, x in rows[-1]['weights'].items()})

    print()
    print('=== ablations ===')
    for v in ['development', 'shift']:
        show('no-soft-ask/' + v, run(v, SEEDS, policy_greedyish))
        show('soft-ask-nolearn/' + v, run(v, SEEDS, policy_v1_fixed))

    print()
    print('=== PRIMARY SCORE (mean of 6 family means) ===')
    for k, d in finals.items():
        print('  %-10s %.4f   %s' % (k, statistics.mean(d.values()),
                                     {v: round(x, 2) for v, x in d.items()}))