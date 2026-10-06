"""Would Differential Evolution work on this problem?

DE needs three things: a fixed-dimensional continuous vector, a scalar fitness,
and enough evaluations to search. This measures all three against reality.

Runs the episodes in parallel across available cores. Safe because each episode
is a pure function of (world seed, variant, weights) with no shared state.
"""
import _bootstrap  # noqa: F401  (sets sys.path portably)

import itertools
import json
import math
import multiprocessing as mp
import statistics
import sys
import time

from kit import generate
from prob_model import run_episode
from candidate import make_ask, make_match, BASE_W

VARIANTS = ['development', 'sparse', 'cold_start', 'delayed', 'shift', 'drift']

# A deliberately wrong weight vector: correct ordering for 'shift', wrong for
# every other family. Used to measure how big a *real* effect looks against noise.
SHIFT_W = {'relationship_goal': .25, 'relationship_pace': .8,
           'lifestyle': -.25, 'conversations': .5}
BAD_W = {'relationship_goal': .7, 'relationship_pace': .4,
         'lifestyle': -.25, 'conversations': .5}
SCRAMBLED_W = {'relationship_goal': -.7, 'relationship_pace': -.4,
               'lifestyle': .25, 'conversations': -.2}


def one_episode(job):
    weights, seed, variant = job
    world = generate(seed, 200, 'evaluation', variant)
    r, _ = run_episode(world, make_ask(True), make_match(weights))
    return (variant, seed, r['msmi_per_100'])


def run_grid(weight_sets, seeds, pool):
    """weight_sets: list of (label, weights). Returns {(label, variant, seed): score}."""
    jobs = [(lab, w, v, s) for lab, w in weight_sets for v in VARIANTS for s in seeds]
    out = {}
    for lab, variant, seed, score in pool.map(_work, jobs, chunksize=1):
        out[(lab, variant, seed)] = score
    return out


def _work(job):
    lab, w, variant, seed = job
    world = generate(seed, 200, 'evaluation', variant)
    r, _ = run_episode(world, make_ask(True), make_match(w))
    return lab, variant, seed, r['msmi_per_100']


def primary(scores, label, seeds):
    fam = [statistics.mean(scores[(label, v, s)] for s in seeds) for v in VARIANTS]
    return statistics.mean(fam), dict(zip(VARIANTS, fam))


def main():
    ncpu = _bootstrap.available_cpus()
    print('cores visible: %d' % ncpu, flush=True)
    # env override so a smoke test can run in seconds without editing code
    import os
    nseed = int(os.environ.get('DE_SEEDS', '12'))
    seed0 = int(os.environ.get('DE_SEED0', '301'))
    seeds = list(range(seed0, seed0 + nseed))
    sets = [('base', dict(BASE_W)), ('shift', dict(SHIFT_W)),
            ('bad', dict(BAD_W)), ('scrambled', dict(SCRAMBLED_W))]
    total_runs = len(sets) * len(VARIANTS) * len(seeds)

    with mp.Pool(ncpu) as pool:
        t0 = time.perf_counter()
        scores = run_grid(sets, seeds, pool)
        elapsed = time.perf_counter() - t0

    print()
    print('=== 1. cost of one fitness evaluation ===')
    per_ep = elapsed / total_runs
    print('  %d episodes (%d weight sets x %d families x %d seeds) in %.1f s'
          % (total_runs, len(sets), len(VARIANTS), len(seeds), elapsed))
    print('  %.4f s per episode wall-clock on %d cores' % (per_ep, ncpu))
    print('  one fitness eval (12 seeds x 6 families = 72 episodes) = %.1f s' % (72 * per_ep))
    print()
    print('  DE budget, population 50 x 100 generations = 5000 evaluations')
    print('    at 72 episodes/eval, 4 seeds/eval: %6.1f h' % (5000 * 4 * 6 * per_ep / 3600))
    print('    at 72 episodes/eval, 8 seeds/eval: %6.1f h' % (5000 * 8 * 6 * per_ep / 3600))
    print('    note: no accelerator applies; this is pure-Python CPU work')

    print()
    print('=== 2. per-family primary scores for four weight vectors ===')
    prim = {}
    for lab, _ in sets:
        p, fam = primary(scores, lab, seeds)
        prim[lab] = p
        print('  %-10s primary %.4f   %s'
              % (lab, p, {k: round(v, 3) for k, v in fam.items()}))

    print()
    print('=== 3. common random numbers: does pairing reduce the noise? ===')
    flat = {lab: [scores[(lab, v, s)] for v in VARIANTS for s in seeds]
            for lab, _ in sets}
    a, b = 'base', 'shift'
    diffs = [scores[(b, v, s)] - scores[(a, v, s)]
             for v in VARIANTS for s in seeds]
    ind_a, ind_b = flat[a], flat[b]
    sd_d = statistics.stdev(diffs)
    sd_a = statistics.stdev(ind_a)
    sd_b = statistics.stdev(ind_b)
    se_unpaired = math.sqrt(sd_a ** 2 + sd_b ** 2) / math.sqrt(len(diffs))
    se_paired = sd_d / math.sqrt(len(diffs))
    print('  A=%s primary %.4f (sd %.4f)   B=%s primary %.4f (sd %.4f)'
          % (a, statistics.mean(ind_a), sd_a, b, statistics.mean(ind_b), sd_b))
    print('  true mean difference            : %+.4f' % (statistics.mean(ind_b) - statistics.mean(ind_a)))
    print('  unpaired standard error         : %.4f   <- what DE effectively does' % se_unpaired)
    print('  PAIRED standard error (CRN)     : %.4f   <- variance reduction %.1fx'
          % (se_paired, se_unpaired / se_paired if se_paired else float('inf')))

    print()
    print('=== 4. how many seed-family runs to resolve a real effect? ===')
    for target in (0.02, 0.05, 0.10, 0.20):
        n = math.ceil((1.96 * sd_d / target) ** 2)
        print('  detect %+.2f  ->  %5d paired runs  (%.0f full 72-episode evals)'
              % (target, n, n / len(VARIANTS)))
    print('  the real assessment gives 20 seeds x 6 families = 120 paired runs')

    print()
    print('=== 5. verdict inputs ===')
    print('  spread across the four weight vectors: %.4f .. %.4f  (range %.4f)'
          % (min(prim.values()), max(prim.values()),
             max(prim.values()) - min(prim.values())))
    print('  best-vs-worst paired sd: %.4f' % statistics.stdev(
        [scores[('shift', v, s)] - scores[('scrambled', v, s)]
         for v in VARIANTS for s in seeds]))
    print('  -> DE searches a 4-dim space whose total achievable range here is ~%.3f,'
          % (max(prim.values()) - min(prim.values())))
    print('     while ONE paired episode already carries noise of similar size.')

    out = {'cores': ncpu, 'episodes': total_runs, 'seconds': elapsed,
           'per_episode_s': per_ep, 'seeds': seeds,
           'primary': {k: v for k, v in prim.items()},
           'family': {lab: primary(scores, lab, seeds)[1] for lab, _ in sets},
           'se_unpaired': se_unpaired, 'se_paired': se_paired,
           'paired_sd': sd_d}
    with open('de_feasibility_result.json', 'w') as fh:
        json.dump(out, fh, indent=2)
    print('\nwrote de_feasibility_result.json')


if __name__ == '__main__':
    main()