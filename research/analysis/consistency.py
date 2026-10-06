"""Consistency check: assigned pairs must be a subset of the reachable edge set."""
import itertools
import statistics
import sys


import _bootstrap  # noqa: F401  (sets sys.path portably)
from kit import Simulator, generate, eligibility, baseline_asks, baseline_match, HARD
from prob_model import _truth_feasible, run_episode
from ceiling import matchable


def check(variant, seeds=range(41, 61)):
    tot_assigned = tot_in_reach = tot_unexplained = 0
    per = []
    for s in seeds:
        world = generate(s, 200, 'eval', variant)
        ms = {m['member_id']: m for m in world['members'] if matchable(m)}
        reach = set()
        for a, b in itertools.combinations(ms.values(), 2):
            if _truth_feasible(a, b):
                reach.add(tuple(sorted((a['member_id'], b['member_id']))))

        def af(st, d):
            return baseline_asks(st)

        def mf(st, d):
            return baseline_match(st)
        r, sim = run_episode(world, af, mf)
        assigned = {tuple(sorted((i['user_a'], i['user_b']))) for i in sim.introductions}
        bad = assigned - reach
        # is every assigned pair still truth-feasible if we relax matchability?
        loose = set()
        allm = {m['member_id']: m for m in world['members']}
        for a, b in itertools.combinations(allm.values(), 2):
            if _truth_feasible(a, b):
                loose.add(tuple(sorted((a['member_id'], b['member_id']))))
        per.append((len(reach), len(assigned), len(bad), len(assigned - loose)))
        tot_assigned += len(assigned)
        tot_in_reach += len(assigned & reach)
        tot_unexplained += len(assigned - loose)
    print('%-12s mean |reachable E|=%.1f  mean assigned=%.1f  assigned-but-unreachable=%d  '
          'assigned-and-not-even-truth-feasible=%d'
          % (variant, statistics.mean(p[0] for p in per), statistics.mean(p[1] for p in per),
             sum(p[2] for p in per), sum(p[3] for p in per)))


def edge_budget(variant, seeds=range(41, 61), rounds=8):
    """How many of the reachable edges can a perfect scheduler actually use?
    Greedily peel maximum matchings; that is the natural 8-round schedule."""
    used_tot = []
    for s in seeds:
        world = generate(s, 200, 'eval', variant)
        ms = {m['member_id']: m for m in world['members'] if matchable(m)}
        edges = set()
        for a, b in itertools.combinations(ms.values(), 2):
            if _truth_feasible(a, b):
                edges.add(tuple(sorted((a['member_id'], b['member_id']))))
        left = set(edges)
        tot = 0
        for _ in range(rounds):
            # greedy max matching on the remaining graph
            left_sorted = sorted(left)
            adj = {}
            for u, v in left_sorted:
                adj.setdefault(u, []).append(v)
                adj.setdefault(v, []).append(u)
            mate = {}

            def aug(u, seen):
                for v in adj[u]:
                    if v in seen:
                        continue
                    seen.add(v)
                    if v not in mate or aug(mate[v], seen):
                        mate[v] = u
                        mate[u] = v
                        return True
                return False

            for u in adj:
                aug(u, set())
            got = set()
            for u, v in mate.items():
                k = tuple(sorted((u, v)))
                got.add(k)
            left -= got
            tot += len(got)
        used_tot.append((len(edges), tot))
    print('%-12s |E|=%.1f  best 8-round peeling uses %.1f edges (%.0f%% of E)'
          % (variant, statistics.mean(a for a, _ in used_tot),
             statistics.mean(b for _, b in used_tot),
             100 * statistics.mean(b for _, b in used_tot) / max(1e-9, statistics.mean(a for a, _ in used_tot))))


if __name__ == '__main__':
    for v in ['development', 'sparse', 'cold_start']:
        check(v)
    print()
    for v in ['development', 'sparse', 'cold_start', 'delayed', 'shift', 'drift']:
        edge_budget(v)