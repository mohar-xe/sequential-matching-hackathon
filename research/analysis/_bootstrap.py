"""Portable import bootstrap.

Originally every analysis script hard-coded two absolute paths under
/root/hackathon, so the code only ran on this one phone. This resolves the kit
and the sibling modules relative to wherever the script actually lives, which
is what lets the same files run unchanged on Kaggle/Colab.

Resolution order for kit.py:
  1. this directory            (flat bundle, e.g. the Kaggle kernel)
  2. ../The-Sequential-Matching-Problem   (repo layout)
  3. this directory's parent
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

KIT_DIR = None
for _base in (HERE, HERE.parent, HERE.parent.parent):
    for _cand in (_base, _base / 'The-Sequential-Matching-Problem'):
        if (_cand / 'kit.py').is_file():
            KIT_DIR = _cand
            break
    if KIT_DIR:
        break

if KIT_DIR is None:
    raise ImportError('kit.py not found near %s' % HERE)

for _p in (str(HERE), str(KIT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def available_cpus():
    """Report usable CPU count without importing anything exotic."""
    import os
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1