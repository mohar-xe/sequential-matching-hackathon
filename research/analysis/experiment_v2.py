"""Focused experiment: compare v1 vs v2 on drift and development worlds.

Runs 12 seeds for each policy on just 2 variants to get results quickly.
This is the targeted ablation that isolates the CUSUM + stabilized SGD contribution.
"""
import math
import statistics
import sys

import _bootstrap  # noqa: F401  (portable kit import; replaces author-local paths)

from kit import generate, baseline_asks, baseline_match
from prob_model import run_episode
from candidate import (
    make_ask, make_match, Weights,
    BASE_W, KEYS, policy_v1, policy_v1_fixed,
    run as run_v1, show as show_v1
)
from policy_v2 import (
    StableWeights, CUSUMDriftDetector,
    make_ask as make_ask_v2, make_match_v2,
    harvest, run as run_v2, show as show_v2,
    sigmoid
)

SEEDS = list(range(101, 113))  # 12 seeds
VARIANTS_FOCUS = ['development', 'shift', 'drift']  # most relevant to our changes


def run_greedy(variant, seeds):
    rows = []
    for s in seeds:
        world = generate(s, 200, 'evaluation', variant)
        r, _ = run_episode(world,
                           lambda st, d: baseline_asks(st),
                           lambda st, d: baseline_match(st))
        rows.append(r)
    return rows


def summarize(label, rows):
    ms = [r['msmi'] for r in rows]
    m = statistics.mean(ms)
    se = statistics.stdev(ms) / math.sqrt(len(ms)) if len(ms) > 1 else 0.0
    mut = statistics.mean(r['mutual_acceptance'] for r in rows)
    asgn = statistics.mean(r['assign_per_100'] for r in rows)
    return label, m, se, mut, asgn


print('=' * 80)
print('FOCUSED EXPERIMENT: v1 (original) vs v2 (stabilized SGD + CUSUM)')
print('%d seeds per policy per variant' % len(SEEDS))
print('=' * 80)

# Header
print('\n%-36s %6s  %5s  %6s  %6s' % ('Policy/Variant', 'MSMI', 'SE', 'Mutual', 'Assign'))
print('-' * 65)

results = {}

for variant in VARIANTS_FOCUS:
    print()
    print('--- %s ---' % variant)

    # 1. Greedy baseline
    rows = run_greedy(variant, SEEDS)
    label, m, se, mut, asgn = summarize('greedy/' + variant, rows)
    results[('greedy', variant)] = {'msmi': m, 'mutual': mut, 'assign': asgn}
    print('  %-34s %6.3f ±%5.3f  %6.1f  %6.1f' % (label, m, se, mut, asgn))

    # 2. V1 with fixed weights
    rows = run_v1(variant, SEEDS, policy_v1_fixed)
    label, m, se, mut, asgn = summarize('v1-fixed/' + variant, rows)
    results[('v1fixed', variant)] = {'msmi': m, 'mutual': mut, 'assign': asgn}
    print('  %-34s %6.3f ±%5.3f  %6.1f  %6.1f' % (label, m, se, mut, asgn))

    # 3. V1 with online learning (original unstable SGD)
    rows = run_v1(variant, SEEDS, policy_v1, learn=True)
    label, m, se, mut, asgn = summarize('v1-learn/' + variant, rows)
    results[('v1learn', variant)] = {'msmi': m, 'mutual': mut, 'assign': asgn}
    print('  %-34s %6.3f ±%5.3f  %6.1f  %6.1f' % (label, m, se, mut, asgn))
    last_weights = rows[-1].get('weights', {})
    print('    v1 learned weights: %s' % {k: round(v, 3) for k, v in last_weights.items()})

    # 4. V2 with stabilized SGD + CUSUM
    rows = run_v2(variant, SEEDS, learn=True)
    label, m, se, mut, asgn = summarize('v2/' + variant, rows)
    results[('v2', variant)] = {'msmi': m, 'mutual': mut, 'assign': asgn}
    print('  %-34s %6.3f ±%5.3f  %6.1f  %6.1f' % ('v2/' + variant, m, se, mut, asgn))
    last_weights = rows[-1].get('weights', {})
    print('    v2 learned weights: %s' % {k: round(v, 3) for k, v in last_weights.items()})
    if variant == 'drift':
        n_alert = sum(1 for r in rows if r.get('drift_alerted'))
        alert_days = [r['drift_alert_day'] for r in rows if r.get('drift_alert_day') is not None]
        print('    CUSUM alerts: %d/%d  |  alert days: %s' % (
            n_alert, len(rows),
            [round(d, 0) for d in sorted(alert_days)] if alert_days else []))
    if variant == 'development':
        n_fa = sum(1 for r in rows if r.get('drift_alerted'))
        print('    CUSUM false alarms: %d/%d' % (n_fa, len(rows)))

print()
print('=' * 80)
print('COMPARISON: V2 vs GREEDY vs V1-LEARN (delta = v2 - baseline)')
print('=' * 80)
for variant in VARIANTS_FOCUS:
    g = results.get(('greedy', variant), {})
    v1l = results.get(('v1learn', variant), {})
    v2 = results.get(('v2', variant), {})
    print('\n%s:' % variant)
    print('  MSMI:   greedy=%.3f  v1-learn=%.3f  v2=%.3f  (v2-greedy=%+.3f, v2-v1=%+.3f)'
          % (g.get('msmi', 0), v1l.get('msmi', 0), v2.get('msmi', 0),
             v2.get('msmi', 0) - g.get('msmi', 0),
             v2.get('msmi', 0) - v1l.get('msmi', 0)))
    print('  Mutual: greedy=%.1f  v1-learn=%.1f  v2=%.1f  (v2-greedy=%+.1f)'
          % (g.get('mutual', 0), v1l.get('mutual', 0), v2.get('mutual', 0),
             v2.get('mutual', 0) - g.get('mutual', 0)))

print()
print('Note: SE ≈ 0.10-0.15 per variant at 12 seeds. Differences < 0.10 are within noise.')
print('The signal-to-noise ratio improves on mutual_acceptance (more events per episode).')
