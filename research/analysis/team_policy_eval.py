"""Reproducible evaluation of Team Policy vs all baselines and ablations.

Compares:
1. Starter Greedy Baseline (kit reference)
2. Candidate v1 Prototype (research/analysis/candidate.py)
3. Policy v2 Prototype (research/analysis/policy_v2.py)
4. Team Policy (VOI Clarification + General-Graph Exact MWM)
5. Ablation 1: Team Policy with Greedy Matching (isolates MWM impact)
6. Ablation 2: Team Policy with Fixed Priors (isolates online learning impact)

Usage:
  cd research/analysis
  python team_policy_eval.py [--seeds 101,102,103,104,105,106]
"""
import argparse
import itertools
import math
from pathlib import Path
import statistics
import sys
import time

repo = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo / "The-Sequential-Matching-Problem"))
sys.path.insert(0, str(repo / "research" / "analysis"))

from kit import generate, Simulator, baseline_asks, baseline_match
from prob_model import run_episode
import candidate
import policy_v2
import team_policy

VARIANTS = ['development', 'sparse', 'cold_start', 'delayed', 'shift', 'drift']


def run_policy_wrapper(world, decide_mode='team'):
    pol_mem = {}

    def af(state, day):
        req = {'schema_version': '1.0.0', 'phase': 'ask', 'state': state, 'memory': pol_mem}
        res = team_policy.decide(req, mode=decide_mode)
        pol_mem.update(res['memory'])
        return res['asks']

    def mf(state, day):
        req = {'schema_version': '1.0.0', 'phase': 'match', 'state': state, 'memory': pol_mem}
        res = team_policy.decide(req, mode=decide_mode)
        pol_mem.update(res['memory'])
        return res['pairs']

    r, _ = run_episode(world, af, mf)
    return r


def run_greedy(world):
    return run_episode(world, lambda st, d: baseline_asks(st), lambda st, d: baseline_match(st))[0]


def run_candidate_v1(world):
    ask = candidate.make_ask(True)
    match = candidate.make_match(dict(candidate.BASE_W))
    return run_episode(world, ask, match)[0]


def run_ablation_greedy_matching(world):
    """Team policy clarification but with GREEDY matching instead of MWM."""
    pol_mem = {}

    def af(state, day):
        req = {'schema_version': '1.0.0', 'phase': 'ask', 'state': state, 'memory': pol_mem}
        res = team_policy.decide(req, mode='team_fixed')
        pol_mem.update(res['memory'])
        return res['asks']

    def mf(state, day):
        # Use greedy allocation
        mem = [m for m in state['members'] if m['available']]
        past = {tuple(sorted((i['user_a'], i['user_b']))) for i in state['introductions']}
        edges = []
        for a, b in itertools.combinations(mem, 2):
            key = tuple(sorted((a['member_id'], b['member_id'])))
            if key in past or team_policy.eligibility(a, b)['status'] != 'feasible':
                continue
            w = team_policy.pair_weight(a, b, team_policy.BASE_W, state['day'])
            edges.append((w, key))
        edges.sort(key=lambda e: (-e[0], e[1]))
        used, out = set(), []
        for _, key in edges:
            if used.isdisjoint(key):
                out.append(list(key))
                used.update(key)
        return out

    return run_episode(world, af, mf)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='101,102,103,104,105,106',
                        help='Comma-separated seeds')
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]

    print("=" * 80)
    print(f"SEQUENTIAL MATCHING BENCHMARK (Seeds: {seeds})")
    print("=" * 80)

    models = [
        ("Organizer Greedy", run_greedy),
        ("Candidate v1 (Prototype)", run_candidate_v1),
        ("Ablation 1 (Smart VOI + Greedy Matching)", run_ablation_greedy_matching),
        ("Team Policy (VOI + MWM + Adaptive)", lambda w: run_policy_wrapper(w, 'team_adaptive')),
        ("Team Policy (VOI + MWM + Fixed)", lambda w: run_policy_wrapper(w, 'team_fixed')),
    ]

    all_results = {}
    for name, runner in models:
        print(f"\nEvaluating: {name}...")
        t0 = time.time()
        v_scores = {}
        v_mutual = {}
        for v in VARIANTS:
            ms, mu = [], []
            for s in seeds:
                world = generate(s, 200, 'evaluation', v)
                r = runner(world)
                ms.append(r['msmi_per_100'])
                mu.append(r['mutual_acceptance'])
            v_scores[v] = statistics.mean(ms)
            v_mutual[v] = statistics.mean(mu)
        elapsed = time.time() - t0
        all_results[name] = {
            'primary_score': statistics.mean(v_scores.values()),
            'mutual_acceptances': statistics.mean(v_mutual.values()),
            'variant_scores': v_scores,
            'elapsed': elapsed
        }
        print(f"  Primary Score: {all_results[name]['primary_score']:.4f}  |  Mutual Acceptances: {all_results[name]['mutual_acceptances']:.2f}  |  Time: {elapsed:.1f}s")
        print(f"  Per-variant MSMI: { {k: round(v_scores[k], 3) for k in VARIANTS} }")

    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Policy Name':<42} {'MSMI/100':<10} {'Rel Lift':<10} {'Mutual':<10}")
    print("-" * 80)
    base_score = all_results["Organizer Greedy"]["primary_score"]
    for name, res in all_results.items():
        score = res['primary_score']
        lift = ((score - base_score) / max(1e-9, base_score)) * 100
        print(f"{name:<42} {score:<10.4f} {lift:>+7.1f}%   {res['mutual_acceptances']:<10.2f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
