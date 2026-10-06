#!/usr/bin/env bash
# Assemble a flat, CPU-only bundle for a Kaggle kernel.
#
# Why flat: the analysis scripts resolve kit.py relative to their own directory
# (see analysis/_bootstrap.py), so a flat layout needs no path surgery and the
# same files run unchanged on Kaggle, Colab, or the phone.
#
# Deliberately EXCLUDED: The-Sequential-Matching-Problem/data/ (11 MB). Nothing
# in these experiments reads it -- every world is synthesised by kit.generate().
set -euo pipefail

SRC=/root/hackathon
KIT="$SRC/The-Sequential-Matching-Problem"
OUT="${1:-$SRC/kaggle-kernel}"
BUNDLE="$OUT/bundle"

rm -rf "$OUT"
mkdir -p "$BUNDLE"

# kit + the modules the experiments import
cp "$KIT/kit.py" "$BUNDLE/"
cp "$KIT/LICENSE" "$BUNDLE/KIT_LICENSE"

for f in _bootstrap.py prob_model.py candidate.py ceiling.py \
         de_feasibility.py consistency.py soft_fields.py; do
  cp "$SRC/research/analysis/$f" "$BUNDLE/"
done

cat > "$OUT/kernel-metadata.json" <<'JSON'
{
  "title": "Sequential Matching - DE feasibility and ceiling checks",
  "code_file": "run_all.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_tpu": false,
  "enable_internet": false
}
JSON

cat > "$OUT/run_all.py" <<'PY'
"""Entry point for the Kaggle kernel.

Pure-Python CPU work, so no accelerator is requested. Uses all visible cores.
"""
import os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bundle'))

def _run(mod):
    print('\n' + '=' * 70)
    print('RUNNING %s' % mod)
    print('=' * 70, flush=True)
    t0 = time.time()
    import importlib
    m = importlib.import_module(mod)
    try:
        m.main()
    except AttributeError:
        import runpy
        runpy.run_module(mod, run_name='__main__')
    print('[%s finished in %.1f s]' % (mod, time.time() - t0), flush=True)

if __name__ == '__main__':
    try:
        import _bootstrap
        print('cores: %d' % _bootstrap.available_cpus(), flush=True)
    except Exception as e:
        print('cpu probe failed: %s' % e, flush=True)

    # 1. the question that was asked: is black-box parameter search viable here?
    _run('de_feasibility')

    # 2. the open inconsistency in research/README.md 6.1 -- unblocks that claim
    if '--with-consistency' in sys.argv:
        _run('consistency')

    print('\nALL DONE')
PY

cp "$SRC/problem.md" "$OUT/problem.md" 2>/dev/null || true
du -sh "$BUNDLE"
echo "bundle contents:"
ls "$BUNDLE"