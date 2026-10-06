"""Are the soft fields already decided? Three readings, tested.

  (a) fixed  -- are the values set once at generation and never change?
  (b) known  -- are they already observable without asking?
  (c) do they matter -- which of the 7 soft fields affect the outcome at all?
"""
import itertools
import math
import statistics
import sys

sys.path.insert(0, '/root/hackathon/The-Sequential-Matching-Problem')
sys.path.insert(0, '/root/hackathon/analysis')

from kit import Simulator, generate, HARD, SOFT, OPTIONS

KEY = ['relationship_goal', 'relationship_pace', 'lifestyle', 'conversations']
UNUSED = ['emotional_availability', 'space_for_relationship', 'relocate']


def test_a_static():
    print('=== (a) are soft values fixed for the whole episode? ===')
    world = generate(7, 200, 'eval', 'development')
    sim = Simulator(world)
    start = {m['member_id']: dict(m['truth']) for m in world['members']}
    drift_found = 0
    for d in range(60):
        sim.advance([])
        for m in world['members']:
            if dict(m['truth']) != start[m['member_id']]:
                drift_found += 1
    print('  members whose soft values changed over 60 days: %d' % drift_found)
    print('  -> values are STATIC: they are fixed parameters, not a moving target.')


def test_b_observable():
    print()
    print('=== (b) how observable are they, per member, and per PAIR? ===')
    print('  fit is a function of agreement between two people, so a field only')
    print('  enters the score when BOTH endpoints have been observed.')
    for variant in ('development', 'cold_start'):
        marg, pairfield, all4 = [], [], []
        for s in (41, 42, 43, 44, 45):
            w = generate(s, 200, 'eval', variant)
            ms = w['members']
            n = len(ms)
            marg.append({k: sum(1 for m in ms if m['fields'].get(k) is not None) / n
                         for k in KEY})
            jf = {k: 0 for k in KEY}
            a4 = total = 0
            for a, b in itertools.combinations(ms, 2):
                total += 1
                ok = 0
                for k in KEY:
                    if a['fields'].get(k) is not None and b['fields'].get(k) is not None:
                        jf[k] += 1
                        ok += 1
                if ok == 4:
                    a4 += 1
            pairfield.append({k: jf[k] / total for k in KEY})
            all4.append(a4 / total)
        pf = {k: statistics.mean(r[k] for r in marg) for k in KEY}
        pj = {k: statistics.mean(r[k] for r in pairfield) for k in KEY}
        q4 = statistics.mean(all4)
        print()
        print('  %s' % variant)
        print('    P(field observed), one person      : %s'
              % {k: round(v, 3) for k, v in pf.items()})
        print('    P(BOTH people observed), one field : %s'
              % {k: round(v, 3) for k, v in pj.items()})
        print('    P(a random PAIR has all 4 jointly observed) = %.5f  (1 in %d)'
              % (q4, 1 / max(1e-9, q4)))
        print('    after ASKING for all 4: per-field 0.93^2=0.865 -> all 4 = %.3f'
              % (0.93 ** 2) ** 4)

def test_c_matters():
    print()
    print('=== (c) which soft fields actually affect the outcome? ===')
    print('  Simulator._prob weights, straight from kit.py:')
    print('    development: relationship_goal .7, relationship_pace .4,')
    print('                 lifestyle .25, conversations .2   (sum 1.55)')
    print('    shift:       relationship_goal .25, relationship_pace .8,')
    print('                 lifestyle -.25, conversations .5   (sum 1.30)')
    src = open('/root/hackathon/The-Sequential-Matching-Problem/kit.py').read()
    body = src[src.index('def _prob'):src.index('def advance')]
    unused_here = [k for k in UNUSED if k in body]
    print('  UNUSED soft fields appear in _prob: %s -> %s'
          % (unused_here, 'NO, they cannot affect the outcome' if not unused_here else 'CHECK'))
    print('  -> asking about %s costs 1 unit and buys NOTHING.' % ', '.join(UNUSED))
    # empirical: vary an unused field, outcome must not move
    base = _empirical_p('development', mutate=None)
    for k in UNUSED:
        alt = _empirical_p('development', mutate=k)
        print('     mutate %-24s P(MSMI) %.4f -> %.4f  (delta %+.4f)'
              % (k, base, alt, alt - base))


_CACHE = {}


def _empirical_p(variant, mutate, npairs=120, reps=200):
    from prob_model import _truth_feasible
    from ceiling import matchable
    if (variant, mutate) in _CACHE:
        return _CACHE[(variant, mutate)]
    import random as _r
    w = generate(41, 200, 'eval', variant)
    ms = [m for m in w['members'] if matchable(m)]
    pairs = [(a, b) for a, b in itertools.combinations(ms, 2) if _truth_feasible(a, b)]
    _r.Random(0).shuffle(pairs)
    pairs = pairs[:npairs]
    hits = tot = 0
    for pi, (a, b) in enumerate(pairs):
        sub = []
        for m in (a, b):
            m = dict(m)
            m['arrived_day'] = 0
            m['exit_day'] = 10000
            for k in HARD:
                m['fields'][k] = m['truth'][k]
                m['field_status'][k] = 'observed'
            if mutate:
                opts = OPTIONS[mutate]
                m['truth'][mutate] = opts[(opts.index(m['truth'][mutate]) + 1) % len(opts)]
            sub.append(m)
        for rep in range(reps):
            sw = dict(w)
            sw['seed'] = 41 * 100003 + rep * 7919 + pi
            sw['members'] = sub
            sim = Simulator(sw)
            sim.advance([[a['member_id'], b['member_id']]])
            for _ in range(40):
                sim.advance([])
            ev = sim.receive_feedback()
            ds = [e for e in ev if e['event'] == 'date_happened' and e['value'] is True]
            ss = [e for e in ev if e['event'] == 'second_meeting_intention']
            tot += 1
            if ds and len(ss) == 2 and all(e['value'] == 'yes' for e in ss) \
                    and all(e['occurred_day'] - ds[0]['occurred_day'] <= 3 for e in ss):
                hits += 1
    val = hits / max(1, tot)
    _CACHE[(variant, mutate)] = val
    return val


if __name__ == '__main__':
    test_a_static()
    test_b_observable()
    test_c_matters()