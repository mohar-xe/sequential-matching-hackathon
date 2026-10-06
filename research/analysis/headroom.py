"""Headroom analysis: decompose the metric, then price each lever."""
import itertools
import statistics
import sys


import _bootstrap  # noqa: F401  (sets sys.path portably)
from kit import Simulator, generate, eligibility, baseline_asks, baseline_match, HARD, SOFT
from prob_model import (p_msmi_truth, fit_term, drift, greedy_match, run_episode,
                        decompose, _truth_feasible)

SEEDS = [11, 12, 13]
VARIANTS = ['development', 'sparse', 'cold_start', 'delayed', 'shift', 'drift']


def show(name, rows, keys=('assign_per_100', 'both_responded', 'mutual_acceptance',
                           'date', 'sec_both_yes', 'sec_both_on_time', 'msmi_per_100',
                           'coverage', 'ask_cost')):
    agg = {k: statistics.mean(r[k] for r in rows) for k in keys}
    print('%-34s ' % name + '  '.join('%s=%7.3f' % (k[:14], v) for k, v in agg.items()))


# ------------------------------------------------------------------ baselines
def ep_greedy(world):
    return run_episode(world, lambda s, d: baseline_asks(s), lambda s, d: baseline_match(s))


def ep_noasks(world):
    return run_episode(world, lambda s, d: [], lambda s, d: baseline_match(s))


def ep_random(world):
    import random as _r

    def match(state, day):
        cand = [m for m in state['members'] if m['available']]
        past = {tuple(sorted((i['user_a'], i['user_b']))) for i in state['introductions']}
        edges = [tuple(sorted((a['member_id'], b['member_id'])))
                 for a, b in itertools.combinations(cand, 2)
                 if eligibility(a, b)['status'] == 'feasible'
                 and tuple(sorted((a['member_id'], b['member_id']))) not in past]
        _r.Random(17 + day).shuffle(edges)
        out, used = [], set()
        for p in edges:
            if used.isdisjoint(p):
                out.append(list(p))
                used.update(p)
        return out

    return run_episode(world, lambda s, d: baseline_asks(s), match)


# -------------------------------------------------------------------- oracles
def full_clear_ask(world_members_by_variant):
    pass


def make_oracle(kind, truth_by_id):
    """kind: 'full' (truth) | 'fit' (fit only) | 'flat' (no value signal)."""
    def ask(state, day):
        mem = [m for m in state['members'] if m['available']]
        budget = state['ask_budget_remaining']
        # Spend on constraint bundles first: they gate feasibility entirely.
        out = []
        want = [m for m in mem
                if any(m['fields'].get(k) is None for k in HARD)
                and not any(m['field_status'][k] == 'declined' for k in HARD)]
        want.sort(key=lambda m: (-sum(1 for k in HARD if m['fields'].get(k) is None), m['arrived_day']))
        while budget >= 3 and want:
            m = want.pop(0)
            out.append({'member_id': m['member_id'], 'field': 'constraints'})
            budget -= 3
        return out

    def match(state, day):
        variant = state.get('_variant', 'development')
        mem = [m for m in state['members'] if m['available']]
        past = {tuple(sorted((i['user_a'], i['user_b']))) for i in state['introductions']}
        edges = []
        for a, b in itertools.combinations(mem, 2):
            key = tuple(sorted((a['member_id'], b['member_id'])))
            if key in past:
                continue
            if eligibility(a, b)['status'] != 'feasible':
                continue
            if kind == 'flat':
                w = 1.0
            else:
                w = p_msmi_truth(truth_by_id[a['member_id']], truth_by_id[b['member_id']],
                                 day=day, variant=variant, full=(kind == 'full'))
            edges.append((w, key))
        return [list(p) for _, p in greedy_match(edges, mem)]

    return ask, match


def run_variant(kind, variant, seeds=SEEDS):
    rows = []
    for s in seeds:
        world = generate(s, 200, 'evaluation', variant)
        truth_by_id = {m['member_id']: m for m in world['members']}
        ask, match = make_oracle(kind, truth_by_id)
        orig = Simulator.observe

        def obs(self, _v=variant, _o=orig):
            st = _o(self)
            st['_variant'] = _v
            return st
        Simulator.observe = obs
        try:
            r, sim = run_episode(world, ask, match)
        finally:
            Simulator.observe = orig
        rows.append(r)
    return rows


def throughput(world):
    """Why only ~80 assignments? Track available / feasible / used each day."""
    sim = Simulator(world)
    trace = []
    for d in range(60):
        sim.resolve_asks(baseline_asks(sim.observe()))
        st = sim.observe()
        mem = [m for m in st['members'] if m['available']]
        feas = 0
        for a, b in itertools.combinations(mem, 2):
            if eligibility(a, b)['status'] == 'feasible':
                feas += 1
        pairs = baseline_match(st)
        trace.append((d, len(st['members']), len(mem), feas, len(pairs),
                      len(sim.introductions)))
        sim.advance(pairs)
    return trace


if __name__ == '__main__':
    import statistics as _st

    print('=== daily trace, greedy, seed 11 ===')
    print('day  visible avail feaspairs batch cum_intro')
    for row in throughput(generate(11, 200, 'evaluation', 'development'))[:24]:
        print('%3d  %7d %5d %9d %5d %9d' % row)

    print()
    print('=== baselines (development, seeds %s) ===' % SEEDS)
    for name, fn in (('greedy', ep_greedy), ('no_asks', ep_noasks), ('random', ep_random)):
        show(name, [fn(generate(s, 200, 'evaluation', 'development'))[0] for s in SEEDS])

    print()
    print('=== oracle ladder (development) ===')
    for kind in ('flat', 'fit', 'full'):
        show('oracle-' + kind, run_variant(kind, 'development'))

    print()
    print('=== oracle-full across variants ===')
    for v in VARIANTS:
        show('oracle-full ' + v, run_variant('full', v))
    print()
    print('=== oracle-fit (policy-legal value signal) across variants ===')
    for v in VARIANTS:
        show('oracle-fit ' + v, run_variant('fit', v))