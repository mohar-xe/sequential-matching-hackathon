#!/usr/bin/env bash
# Assemble a SELF-CONTAINED, CPU-only Kaggle kernel script.
#
# Why self-contained: a Kaggle *script* kernel ships only its single
# code_file. Sibling .py files in the kernel directory are NOT made available
# at runtime (verified: version 2 failed with ModuleNotFoundError for a module
# that was present next to the entry point). So we embed every module as
# base64, materialise them into a temp dir at startup, then import normally.
#
# Why CPU-only: the workload is pure-Python simulation. A GPU accelerator would
# not help and would burn weekly quota for nothing.
#
# Deliberately EXCLUDED: The-Sequential-Matching-Problem/data/ (11 MB). Nothing
# in these experiments reads it -- every world is synthesised by kit.generate().
set -euo pipefail

SRC=/root/hackathon
KIT="$SRC/The-Sequential-Matching-Problem"
OUT="${1:-$SRC/kaggle-kernel}"
TMP="$OUT/.pack"

rm -rf "$OUT"
mkdir -p "$TMP"

cp "$KIT/kit.py" "$TMP/kit.py"
for f in _bootstrap.py prob_model.py candidate.py ceiling.py \
         de_feasibility.py consistency.py soft_fields.py; do
  cp "$SRC/research/analysis/$f" "$TMP/$f"
done

python3 - "$TMP" "$OUT" <<'PY'
import base64, hashlib, json, pathlib, sys

tmp, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
mods = {}
for p in sorted(tmp.glob('*.py')):
    raw = p.read_bytes()
    mods[p.name] = base64.b64encode(raw).decode()
    print('  embed %-24s %6d bytes' % (p.name, len(raw)))

manifest = {name: hashlib.sha256(base64.b64decode(b)).hexdigest()[:16]
            for name, b in mods.items()}

meta = {
    "title": "Sequential Matching DE Check",
    "id": "malow007/sequential-matching-de-check",
    "code_file": "run_all.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": False,
    "enable_tpu": False,
    "enable_internet": False,
}
(out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')

body = '''"""Sequential Matching Hackathon - offloaded experiments.

Self-contained by necessity: Kaggle script kernels ship only the code_file, so
the modules below are embedded as base64 and materialised at startup.

CPU-only: pure-Python simulation, so a GPU would not help.
"""
import base64
import hashlib
import importlib
import os
import runpy
import sys
import tempfile
import time

# sha256[:16] of each embedded module, for integrity checking after unpacking.
MANIFEST = %(manifest)r

MODULES = {
%(payload)s}


def materialise():
    root = tempfile.mkdtemp(prefix='seqmatch_')
    for name, blob in MODULES.items():
        raw = base64.b64decode(blob)
        got = hashlib.sha256(raw).hexdigest()[:16]
        if got != MANIFEST[name]:
            raise SystemExit('checksum mismatch for %%s: %%s != %%s'
                             %% (name, got, MANIFEST[name]))
        with open(os.path.join(root, name), 'wb') as fh:
            fh.write(raw)
    sys.path.insert(0, root)
    return root


def _run(mod):
    print('\\n' + '=' * 72)
    print('RUNNING %%s' %% mod)
    print('=' * 72, flush=True)
    t0 = time.time()
    m = importlib.import_module(mod)
    if hasattr(m, 'main'):
        m.main()
    else:
        runpy.run_module(mod, run_name='__main__')
    print('[%%s finished in %%.1f s]' %% (mod, time.time() - t0), flush=True)


if __name__ == '__main__':
    root = materialise()
    print('modules unpacked to %%s' %% root, flush=True)
    print('files: %%s' %% sorted(os.listdir(root)), flush=True)
    import _bootstrap
    print('cores: %%d' %% _bootstrap.available_cpus(), flush=True)

    _run('de_feasibility')

    # Second target: the unresolved sparse-ceiling inconsistency documented in
    # research/README.md section 6.1.
    _run('consistency')

    print('\\nALL DONE')
'''

payload = ''.join(
    '    %r: (\n%s\n    ),\n' % (name, '\n'.join(
        '        %r' % line for line in blob.splitlines()) or "        ''")
    for name, blob in mods.items())

(out / 'run_all.py').write_text(body % {'manifest': manifest, 'payload': payload})
print('\nwrote %s (%.0f KB)' % (out / 'run_all.py', (out / 'run_all.py').stat().st_size / 1024))
PY

rm -rf "$TMP"
cp "$KIT/LICENSE" "$OUT/KIT_LICENSE"
echo "kernel dir:"
ls "$OUT"