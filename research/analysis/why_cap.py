"""What actually caps the number of introductions? Measure, don't guess."""
import itertools
import statistics
import sys


import _bootstrap  # noqa: F401  (sets sys.path portably)
from kit import Simulator, generate, eligibility, HARD, SOFT
from prob_model import _truth_feasible

VARIANTS = ['development', 'sparse', 'cold_start', 'delayed', 'shift', 'drift']


def cleared(m):
    return all(m['fields'].get(k) is not None for k in HARD)


def truth_cleared(m):
    return True


def density_report(variant, n=200, seeds=range(31, 39)):
    """Rejection cascade on a fully-observed world (the best case)."""
    print('--- %s: rejection cascade on TRUTH (n=%d) ---' % (variant, n))
    reasons = {}

    def bump(r):
        reasons[r] = reasons.get(r, 0) + 1

    tot = 0
    for s in seeds:
        w = generate(s, n, 'eval', variant)
        ms = w['members']
        for a, b in itertools.combinations(ms, 2):
            tot += 1
            # per-direction reason attribution, count pair once
            fa, fb = a['truth'], b['truth']
            bad = None
            for x, y, fx, fy in ((a, b, fa, fb), (b, a, fb, fa)):
                if y['gender'] not in fx['who_to_meet']:
                    bad = bad or 'who_to_meet'
                if y['age'] < fx['age_min'] or y['age'] > fx['age_max']:
                    bad = bad or 'age'
                if y['zone'] not in fx['acceptable_zones']:
                    bad = bad or 'geography'
                if fx['partner_smoking'] == 'no_smoking' and fy['smoking'] in ('yes', 'occasionally'):
                    bad = bad or 'smoking'
                if fx['partner_children'] == 'no_children' and fy['has_children'] is True:
                    bad = bad or 'children_present'
            if fa['relationship_structure'] != fb['relationship_structure']:
                bad = bad or 'relationship_structure'
            if {fa['wants_children'], fb['wants_children']} == {'yes', 'no'}:
                bad = bad or 'children_plans'
            if not (set(fa['schedule']) & set(fb['schedule'])):
                bad = bad or 'schedule'
            bump(bad or 'FEASIBLE')
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print('  %-24s %7d  %6.2f%% of all pairs' % (k, v, 100 * v / tot))
    print()


def observed_density(variant, seeds=range(31, 39)):
    """How many members are actually cleared, and what density does that give?"""
    print('--- %s: observed-state density during a greedy episode ---' % variant)
    from kit import baseline_asks, baseline_match
    world = generate(next(iter(seeds)), 200, 'eval', variant)
    sim = Simulator(world)
    # state members are redacted copies; keep a truth side-table by id
    truth = {}
    for m in world['members']:
        truth[m['member_id']] = dict(m)
    rows = []
    asks_per_day = []
    for d in range(60):
        a = baseline_asks(sim.observe())
        asks_per_day.append(len(a))
        sim.resolve_asks(a)
        st = sim.observe()
        mem = [m for m in st['members'] if m['available']]
        cl = [m for m in mem if cleared(m)]
        feas_obs = 0
        feas_locked = 0
        for x, y in itertools.combinations(mem, 2):
            if eligibility(x, y)['status'] == 'feasible':
                feas_obs += 1
            elif _truth_feasible(truth[x['member_id']], truth[y['member_id']]):
                feas_locked += 1
        pairs = baseline_match(st)
        rows.append((d, len(mem), len(cl), feas_obs, feas_locked, len(pairs)))
        sim.advance(pairs)
    print('  day avail cleared feasObs feasTruthLocked batch')
    for r in rows[:4] + rows[10:14] + rows[25:29] + rows[-3:]:
        print('  %3d %5d %7d %7d %15d %6d' % r)
    tot_obs = sum(r[3] for r in rows)
    tot_lock = sum(r[4] for r in rows)
    tot_bat = sum(r[5] for r in rows)
    print('  daily ask counts: %s' % asks_per_day)
    print('  TOTALS feasible-visible=%d  truth-feasible-but-locked=%d  assigned=%d'
          % (tot_obs, tot_lock, tot_bat))
    if tot_obs + tot_lock:
        print('  -> observation layer sees %.1f%% of feasible pairs'
              % (100 * tot_obs / (tot_obs + tot_lock)))
    print()


def supply(variant, seeds=range(31, 39)):
    """How many members can EVER be matched (no declined hard field)?"""
    tot = nc = nask = ndecl = 0
    for s in seeds:
        w = generate(s, 200, 'eval', variant)
        for m in w['members']:
            tot += 1
            miss = [k for k in HARD if m['fields'].get(k) is None]
            if not miss:
                nc += 1
            elif not any(m['field_status'][k] == 'declined' for k in miss):
                nask += 1
            else:
                ndecl += 1
    print('  %-12s cleared_now=%3d (%4.1f%%)  clarifiable=%3d  UNREACHABLE=%3d (%4.1f%%)  '
          'P(matchable)=%.3f'
          % (variant, nc / len(list(seeds)), 100 * nc / tot,
             nask / len(list(seeds)), ndecl / len(list(seeds)), 100 * ndecl / tot,
             (nc + nask) / tot))


if __name__ == '__main__':
    for v in VARIANTS:
        density_report(v)
    for v in ('development', 'sparse', 'cold_start'):
        observed_density(v)
    print('--- clarification supply ---')
    for v in VARIANTS:
        supply(v)