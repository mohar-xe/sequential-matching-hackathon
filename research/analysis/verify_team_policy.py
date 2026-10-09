"""Independent verification of the team-policy branch findings.

Oviya's implementation lives on the `team-policy-experiment` branch
(unmerged at Round 1 freeze to keep `The-Sequential-Matching-Problem/`
verbatim). This script loads `team_policy.py` from that ref (or `--path`
to a checkout) and checks the two load-bearing claims that feed the
Round 1 submission:

1. MWM exactness: fuzz `max_weight_matching` against brute-force optima.
2. v1 soft-ask waste: re-run candidate-v1 asks on seed 101 / development
   and count soft queries to members with a declined HARD field.

Usage:
  cd research/analysis
  python3 verify_team_policy.py [--path /path/to/team_policy.py] [--trials 300]
"""
import argparse
import importlib.util
import itertools
import random
import subprocess
import sys

import _bootstrap  # noqa: F401  (sets sys.path portably)


def load_team_policy(path=None):
    if path is None:
        raw = subprocess.run(
            ['git', 'show',
             'team-policy-experiment:The-Sequential-Matching-Problem/team_policy.py'],
            capture_output=True, check=True, text=True).stdout
        import os
        import tempfile
        fd, tmp = tempfile.mkstemp(suffix='_team_policy.py')
        with os.fdopen(fd, 'w') as fh:
            fh.write(raw)
        path = tmp
    spec = importlib.util.spec_from_file_location('team_policy_under_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def brute_opt(edges):
    best = 0.0
    for mask in range(1 << len(edges)):
        used, tot, ok = set(), 0.0, True
        for i in range(len(edges)):
            if mask >> i & 1:
                w, (u, v) = edges[i]
                if u in used or v in used:
                    ok = False
                    break
                used.add(u)
                used.add(v)
                tot += w
        if ok and tot > best + 1e-9:
            best = tot
    return best


def fuzz_mwm(mod, trials=300, seed=0):
    r = random.Random(seed)
    fails = done = 0
    for _ in range(trials):
        nodes = [chr(65 + i) for i in range(r.randint(2, 7))]
        edges = [(round(r.random() * 2, 2), (a, b))
                 for a, b in itertools.combinations(nodes, 2)
                 if r.random() < 0.5]
        if not edges:
            continue
        done += 1
        got = mod.max_weight_matching(edges)
        gw = sum(next(w for w, e in edges if set(e) == set(p)) for p in got)
        if abs(gw - brute_opt([(w, tuple(e)) for w, e in edges if w > 0])) > 1e-6:
            fails += 1
    print('MWM fuzz: %d/%d optimal' % (done - fails, done))
    return fails == 0


def waste_check():
    from kit import generate, Simulator, HARD
    import candidate
    world = generate(101, 200, 'evaluation', 'development')
    ask = candidate.make_ask(True)
    match = candidate.make_match(dict(candidate.BASE_W))
    sim = Simulator(world)
    nsoft = nwasted = 0
    for d in range(60):
        st = sim.observe()
        by = {m['member_id']: m for m in st['members']}
        aq = ask(st, d)
        for a in aq:
            if a['field'] != 'constraints':
                nsoft += 1
                if any(by[a['member_id']]['field_status'][k] == 'declined'
                       for k in HARD):
                    nwasted += 1
        sim.resolve_asks(aq)
        sim.advance(match(sim.observe(), d))
    print('v1 soft asks: %d, to unmatchable members: %d (%.1f%%)'
          % (nsoft, nwasted, 100 * nwasted / max(1, nsoft)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', default=None)
    ap.add_argument('--trials', type=int, default=300)
    args = ap.parse_args()
    mod = load_team_policy(args.path)
    ok = fuzz_mwm(mod, trials=args.trials)
    waste_check()
    sys.exit(0 if ok else 1)
