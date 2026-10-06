"""The definitive ceiling.

A member is matchable only if every HARD field can be observed, i.e. no
declined hard field. Among matchable members the feasible graph is what the
policy is allocating over. |E| bounds total introductions for the episode
(pairs may not repeat); the maximum matching bounds any single day.
"""
import itertools
import math
import statistics
import sys


import _bootstrap  # noqa: F401  (sets sys.path portably)
from kit import generate, HARD
from prob_model import _truth_feasible


def matchable(m):
    miss = [k for k in HARD if m['fields'].get(k) is None]
    return not any(m['field_status'][k] == 'declined' for k in miss)


def max_matching_size(nodes, edges):
    """Cardinality max matching (general graph, small n) via simple augmenting."""
    adj = {n: [] for n in nodes}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    mate = {}

    def try_aug(u, seen):
        for v in adj[u]:
            if v in seen:
                continue
            seen.add(v)
            if v not in mate or try_aug(mate[v], seen):
                mate[v] = u
                mate[u] = v
                return True
        return False

    for u in nodes:
        try_aug(u, set())
    return len(mate) // 2


def greedy_worst_sizes(nodes, edges):
    """Edge-colouring upper bound: you need >= max_degree rounds to use all edges."""
    deg = {n: 0 for n in nodes}
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return max(deg.values()) if deg else 0


def ceiling(variant, seeds=range(41, 61), n=200):
    rows = []
    for s in seeds:
        w = generate(s, n, 'eval', variant)
        ms = [m for m in w['members'] if matchable(m)]
        ids = {m['member_id'] for m in ms}
        edges = []
        for a, b in itertools.combinations(ms, 2):
            if _truth_feasible(a, b):
                edges.append((a['member_id'], b['member_id']))
        mm = max_matching_size(ids, edges)
        md = greedy_worst_sizes(ids, edges)
        rows.append((len(ms), len(edges), mm, md))
    print('%-12s matchable=%5.1f  |E|=%6.1f  max_matching=%5.1f  max_degree=%4.1f  '
          'matchable%%=%.2f'
          % (variant, statistics.mean(r[0] for r in rows),
             statistics.mean(r[1] for r in rows), statistics.mean(r[2] for r in rows),
             statistics.mean(r[3] for r in rows),
             100 * statistics.mean(r[0] for r in rows) / n))
    return rows


def schedulable_ceiling(variant, seeds=range(41, 61), n=200, rounds=8):
    """Each round can add at most one max matching; 8 rounds fit in 60 days
    given the 8-day post-introduction lockout."""
    rows = ceiling(variant, seeds, n)
    vals = [min(r[1], rounds * r[2]) for r in rows]
    print('   -> 8-round schedulable ceiling on introductions: %.1f'
          % statistics.mean(vals))


if __name__ == '__main__':
    print('=== reachable feasible graph (TRUTH, matchable members only) ===')
    for v in ['development', 'sparse', 'cold_start', 'delayed', 'shift', 'drift']:
        schedulable_ceiling(v)
        print()