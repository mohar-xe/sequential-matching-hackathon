# Sequential Matching Hackathon — team repo

Private working repo for the Vouchsafe **Sequential Matching Hackathon**.

> **Round 1 research submission closes 9 Oct 2026, 23:59 IST.**
> Round 2 build window is 12–18 Oct 2026.

## Start here

**[`research/README.md`](research/README.md)** — our main document. It explains the
challenge from first principles in plain language, then with measurements: why
**only 1.24% of pairs are even legal**, why **a third of people can never be
matched**, why **98.5% of every introduction fails for reasons no policy can
control**, and why the supplied baseline already sits at **89% of the achievable
ceiling**. Ends with our hypothesis, the proposed solution, and what we got wrong.

[`research/RESEARCH_NOTE.md`](research/RESEARCH_NOTE.md) — the technical Round 1
submission draft: full derivations, the recovered generative law, funnel
decomposition, and citations.

## Layout

| Path | What it is |
|---|---|
| `research/README.md` | Main explanation: layman → first principles → hypothesis → solution |
| `research/RESEARCH_NOTE.md` | Technical submission draft with derivations and citations |
| `research/analysis/` | The 9 stdlib-only scripts that produced every number cited |
| `The-Sequential-Matching-Problem/` | **Organisers' starter kit, vendored verbatim.** Not our code. |
| [`KIT_PROVENANCE.md`](KIT_PROVENANCE.md) | Upstream URL, pinned commit, licences, how to re-sync |

## Reproducing the numbers

```bash
cd research/analysis
python3 prob_model.py      # recovered generative law + oracle scoring
python3 validate_model.py  # our model vs the real simulator, per world
python3 why_cap.py         # why only 1.24% of pairs are legal; daily density
python3 ceiling.py         # reachable edge set, ceiling, max matching
python3 value_spread.py    # what value-ordering is actually worth
python3 candidate.py       # prototype vs all three baselines, all six worlds
```

Everything is **pure standard library** — `numpy`/`scipy`/`networkx` are not
installed, and assessed inference is capped at 2 cores / 1 GiB / 10 s per call, so
heavier dependencies are off the table anyway.

Verify the kit itself:

```bash
cd The-Sequential-Matching-Problem
python3 -m unittest -v      # 22 tests, OK
python3 verify_data.py
```

## Current status — read before quoting numbers

- Round 1 note is drafted and submission-ready.
- **No policy is implemented yet.** Nothing here satisfies the required JSON
  ask/match interface; there is no `results/` and no Dockerfile.
- The one prototype scored **0.417 vs the baseline's 0.403** — *inside the noise
  band* (SE ≈ 0.04). Mechanism metrics moved clearly (mutual acceptances +10% in
  `development`, +17% in `shift`); the headline metric is simply underpowered at
  12 seeds. Do not present that as an improvement.
- **Two unresolved inconsistencies** are documented in `research/README.md` §6.1:
  the `sparse` ceiling contradicts observed baseline volume, and the aggregate
  model under-predicts MSMI by ~2×. `research/analysis/consistency.py` was written
  to localise the first and **has never been run**. Do not publish the `sparse`
  ceiling until it is.

## Contribution

`research/` is ours. `The-Sequential-Matching-Problem/` belongs to the organisers
under MIT + the synthetic-data licence — please don't relicense it or edit it in
place. If you change the kit, do it in a branch and say why in the PR.