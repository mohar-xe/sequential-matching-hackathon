"""Is value-ordering worth anything when the edge set is scarce?

|E| reachable is ~103 edges for 133 matchable members. Each edge is consumed
forever (no repeats). So the policy is not choosing among many pairs, it is
sequencing a scarce, non-renewable edge set. Question: how much does ordering
that set by P(MSMI) buy over an arbitrary feasible matching?
"""
import itertools
import math
import statistics
import sys

sys.path.insert(0, '/root/hackathon/The-Sequential-Matching-Problem')
sys.path.insert(0, '/root/hackathon/analysis')

from kit import generate, HARD
from prob_model import _truth_feasible, p_msmi_truth, greedy_match
from ceiling import matchable


def reachable_edges(variant, n=200):
    w = generate(n and 41, n, 'eval', variant)
    ms = [m for m in w['members'] if matchable(m)]
    by = {m['member_id']: m for m in ms}
    edges = []
    for a, b in itertools.combinations(ms, 2):
        if _truth_feasible(a, b):
            edges.append((a, b))
    return by, edges


def value_spread(variant, seeds=range(41, 61)):
    allv, summax, sumall, sumtop = [], [], [], []
    for s in seeds:
        w = generate(s, 200, 'eval', variant)
        ms = [m for m in w['members'] if matchable(m)]
        edges = [(a, b) for a, b in itertools.combinations(ms, 2) if _truth_feasible(a, b)]
        v = [p_msmi_truth(a, b, day=0, variant=variant, full=True) for a, b in edges]
        if not v:
            continue
        allv += v
        ids = [m['member_id'] for m in ms]
        keyed = [(p_msmi_truth(a, b, 0, variant, True),
                  (a['member_id'], b['member_id']))
                 for a, b in edges]
        # greedy max-weight matching (single batch, ignores time)
        picked = greedy_match(keyed, ids)
        summax.append(sum(w_ for w_, _ in picked))
        sumall.append(sum(w_ for w_, _ in keyed))
        k = max(1, len(picked))
        sumtop.append(sum(sorted((w_ for w_, _ in keyed), reverse=True)[:k]))
    if not allv:
        return
    allv.sort()

    def q(p):
        return allv[int(p * (len(allv) - 1))]

    print('%-12s n_edges=%5.1f  P(MSMI): p05=%.4f p25=%.4f med=%.4f p75=%.4f p95=%.4f max=%.4f'
          % (variant, len(allv) / 20, q(.05), q(.25), q(.50), q(.75), q(.95), allv[-1]))
    print('   best-per-batch max-weight matching = %.3f  |  all edges = %.3f  |  '
          'top-k by value = %.3f   (max/all = %.2f)'
          % (statistics.mean(summax), statistics.mean(sumall),
             statistics.mean(sumtop),
             statistics.mean(summax) / max(1e-9, statistics.mean(sumall))))


def by_id_factory(variant, world):
    return None


def by_id(a):
    return a


def decompose_by_latent(variant, seeds=range(41, 61)):
    """How much of P(MSMI) is explainable from what a policy can observe?"""
    buckets = {'fit only': [], 'fit+response': []}
    for s in seeds:
        w = generate(s, 200, 'eval', variant)
        ms = [m for m in w['members'] if matchable(m)]
        for a, b in itertools.combinations(ms, 2):
            if not _truth_feasible(a, b):
                continue
            truth = p_msmi_truth(a, b, 0, variant, True)
            fit_only = p_msmi_truth(a, b, 0, variant, False)
            fit_resp = fit_only * a['response_rate'] * b['response_rate'] / 0.585
            buckets['fit only'].append((fit_only, truth))
            buckets['fit+response'].append((fit_resp, truth))
    for k, pairs in buckets.items():
        pairs.sort(key=lambda t: -t[0])
        n = len(pairs)
        top = pairs[:n // 2]
        bot = pairs[n // 2:]
        print('   %-14s top-half mean truth P=%.4f   bottom-half=%.4f   lift=%.2fx'
              % (k, statistics.mean(t for _, t in top),
                 statistics.mean(t for _, t in bot),
                 statistics.mean(t for _, t in top) / max(1e-9, statistics.mean(t for _, t in bot))))


if __name__ == '__main__':
    print('=== oracle P(MSMI) spread over the reachable feasible edge set ===')
    for v in ['development', 'sparse', 'cold_start']:
        value_spread(v)
        decompose_by_latent(v)
        print()