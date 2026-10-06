"""Where should the 3-unit constraint bundles go?

The trace from headroom.py showed day 0 has 134 visible members but only
~10 feasible pairs. This script prices the ask-allocation question directly.
"""
import itertools
import math
import statistics
import sys


import _bootstrap  # noqa: F401  (sets sys.path portably)
from kit import Simulator, generate, eligibility, baseline_asks, baseline_match, HARD, SOFT
from prob_model import run_episode, greedy_match, p_msmi_truth

SEEDS = list(range(21, 33))
VARIANTS = ['development', 'sparse', 'cold_start', 'delayed', 'shift', 'drift']
KEYS = ('assign_per_100', 'mutual_acceptance', 'msmi', 'msmi_per_100', 'coverage', 'ask_cost')


def show(name, rows):
    agg = {k: statistics.mean(r[k] for r in rows) for k in KEYS}
    se = statistics.mean(r['msmi'] for r in rows) / max(1, math.sqrt(len(rows)))
    print('%-30s assign=%6.1f  mutual=%5.1f  MSMI=%5.2f(+-%.2f)  per100=%5.3f  cov=%.3f  ask=%5.1f'
          % (name, agg['assign_per_100'], agg['mutual_acceptance'], agg['msmi'],
             se, agg['msmi_per_100'], agg['coverage'], agg['ask_cost']))


# ------------------------------------------------------------ ask strategies
def ask_greedy(state, day):
    return baseline_asks(state)


def ask_zone_cluster(state, day, variant):
    """Clarify only inside the largest zones -> dense feasible subgraph."""
    budget = state['ask_budget_remaining']
    mem = [m for m in state['members'] if m['available']]
    counts = {}
    for m in mem:
        counts[m['zone']] = counts.get(m['zone'], 0) + 1
    if variant == 'sparse':
        top = {z for z, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:3]}
    else:
        top = {z for z, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:2]}
    out, want = [], [m for m in mem if m['zone'] in top
                     and any(m['fields'].get(k) is None for k in HARD)
                     and not any(m['field_status'][k] == 'declined' for k in HARD)]
    want.sort(key=lambda m: -sum(1 for k in HARD if m['fields'].get(k) is None))
    while budget >= 3 and want:
        out.append({'member_id': want.pop(0)['member_id'], 'field': 'constraints'})
        budget -= 3
    return out


def ask_eager(state, day):
    """Drain the askable supply at full budget, nothing smarter."""
    budget = state['ask_budget_remaining']
    mem = [m for m in state['members'] if m['available']]
    want = [m for m in mem
            if any(m['fields'].get(k) is None for k in HARD)
            and not any(m['field_status'][k] == 'declined' for k in HARD)]
    want.sort(key=lambda m: m['arrived_day'])
    out = []
    while budget >= 3 and want:
        out.append({'member_id': want.pop(0)['member_id'], 'field': 'constraints'})
        budget -= 3
    return out


def ask_soft_mix(state, day):
    """Half the budget on hard bundles, half on the 4 decision-relevant soft fields."""
    budget = state['ask_budget_remaining']
    mem = [m for m in state['members'] if m['available']]
    hard = [m for m in mem
            if any(m['fields'].get(k) is None for k in HARD)
            and not any(m['field_status'][k] == 'declined' for k in HARD)]
    out = []
    n_hard = 2
    for m in hard[:n_hard]:
        out.append({'member_id': m['member_id'], 'field': 'constraints'})
    budget -= 3 * n_hard
    soft = []
    for m in mem:
        for k in ('relationship_goal', 'relationship_pace', 'lifestyle', 'conversations'):
            if m['fields'].get(k) is None and m['field_status'][k] != 'declined':
                soft.append((m['member_id'], k))
    for mid, k in soft[:budget]:
        out.append({'member_id': mid, 'field': k})
    return out


# --------------------------------------------------------- match strategies
def match_greedy(state, day):
    return baseline_match(state)


def make_mwm(mode, truth_by_id, variant):
    """Max-weight matching on P(MSMI); mode picks the weight model."""
    def match(state, day):
        mem = [m for m in state['members'] if m['available']]
        past = {tuple(sorted((i['user_a'], i['user_b']))) for i in state['introductions']}
        edges = []
        for a, b in itertools.combinations(mem, 2):
            key = tuple(sorted((a['member_id'], b['member_id'])))
            if key in past or eligibility(a, b)['status'] != 'feasible':
                continue
            if mode == 'blind':
                w = 1.0
            elif mode == 'soft':
                fa, fb = a['fields'], b['fields']
                n = sum(1 for k in SOFT
                        if fa.get(k) is not None and fb.get(k) is not None and fa[k] == fb[k])
                w = 1.0 + n
            else:  # oracle
                w = p_msmi_truth(truth_by_id[a['member_id']], truth_by_id[b['member_id']],
                                 day=day, variant=variant, full=True)
            edges.append((w, key))
        return [list(p) for _, p in greedy_match(edges, mem)]
    return match


def combo(ask_fn, match_fn, variant, seeds=SEEDS, truth=False):
    rows = []
    for s in seeds:
        world = generate(s, 200, 'evaluation', variant)
        tbi = {m['member_id']: m for m in world['members']}
        orig = Simulator.observe

        def obs(self, _v=variant, _o=orig):
            st = _o(self)
            st['_variant'] = _v
            return st
        Simulator.observe = obs
        try:
            af = (lambda st, d, _v=variant: ask_fn(st, d, _v)) if ask_fn is ask_zone_cluster else ask_fn
            r, _ = run_episode(world, af, match_fn)
        finally:
            Simulator.observe = orig
        rows.append(r)
    return rows


if __name__ == '__main__':
    print('=== A. ask strategy, greedy match (development, %d seeds) ===' % len(SEEDS))
    show('ask=baseline', combo(ask_greedy, match_greedy, 'development'))
    show('ask=eager', combo(ask_eager, match_greedy, 'development'))
    show('ask=zone_cluster', combo(ask_zone_cluster, match_greedy, 'development'))
    show('ask=soft_mix', combo(ask_soft_mix, match_greedy, 'development'))

    print()
    print('=== B. ask x match grid (development) ===')
    for aname, afn in (('baseline', ask_greedy), ('eager', ask_eager),
                       ('zone_cluster', ask_zone_cluster)):
        for mname, mfn in (('greedy', match_greedy),
                           ('mwm-soft', make_mwm('soft', None, 'development')),
                           ('mwm-oracle', make_mwm('oracle', {}, 'development'))):
            if mname == 'mwm-oracle':
                rows = []
                for s in SEEDS:
                    world = generate(s, 200, 'evaluation', 'development')
                    tbi = {m['member_id']: m for m in world['members']}
                    orig = Simulator.observe

                    def obs(self, _o=orig):
                        return _o(self)
                    Simulator.observe = orig
                    af = (lambda st, d: afn(st, d, 'development')) if afn is ask_zone_cluster else afn
                    r, _ = run_episode(world, af, make_mwm('oracle', tbi, 'development'))
                    rows.append(r)
                show('%s + %s' % (aname, mname), rows)
            else:
                show('%s + %s' % (aname, mname), combo(afn, mfn, 'development'))