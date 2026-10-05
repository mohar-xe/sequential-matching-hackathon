# Kit provenance

The starter kit in `The-Sequential-Matching-Problem/` is **not our code**. It is
vendored verbatim from the organisers so the team has one offline, self-contained
checkout on a phone.

| | |
|---|---|
| Upstream | `https://github.com/RomeoJulietLove/The-Sequential-Matching-Problem` |
| Pinned commit | `a8e26b35118cfa8e886a02f93984923e43ab64f` |
| Release | Participant release **1.0.0** (5 October 2026) |
| Vendored on | 5 October 2026 |

Licences are preserved in place and govern that subtree:

- `The-Sequential-Matching-Problem/LICENSE` — MIT, starter code, © 2026 RomeoJulietLove
- `The-Sequential-Matching-Problem/DATA_LICENSE.md` — synthetic dataset permission

The upstream README states the starter-code and synthetic-data permissions permit
reuse of the kit. Nothing in that subtree is ours to relicense, and our own work is
confined to `research/`.

## Why it is vendored instead of a submodule

A submodule would re-download ~1 MB of git history and needs network access on every
fresh clone — awkward on a phone. Vendoring costs 11 MB once and works offline.

The trade-off is that upstream updates are no longer a `git pull`. To re-sync:

```bash
rm -rf The-Sequential-Matching-Problem
git clone https://github.com/RomeoJulietLove/The-Sequential-Matching-Problem
rm -rf The-Sequential-Matching-Problem/.git
# then verify against a new pinned SHA recorded in this file
python -m unittest -v          # expect 22 tests, OK
python verify_data.py
```

## Verification after any re-sync

```bash
cd The-Sequential-Matching-Problem
python3 -m unittest -v        # 22 tests
python3 verify_data.py        # checksum verification of data/
```

If the test count or checksum output differs from the pinned commit above, the
release has moved and our measurements in `research/` must be re-run before any
number is quoted.