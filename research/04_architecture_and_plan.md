# 04 · Proposed Architecture & Experiment Plan

## 4.1 System architecture (one repo, offline inference, no extra deps)

```
policy.py                          # the shipped interface entry-point; KEEP JSON protocol
├── policy/pipeline.py             # request parsing, phase dispatch, memory (de)serialisation
├── policy/features.py             # pair-feature extraction, signed agreement vector, cache
├── policy/scorer.py               # Bayesian logistic pair scorer (online-updateable)
├── policy/member_bandit.py        # per-member Beta posteriors + pool-structured priors
├── policy/clarify.py              # VOI clarification ranking under 12/day budget
├── policy/allocate.py             # greedy + single-pair-replacement local search over
│                                  # the general (non-bipartite) eligible graph
├── policy/pretrain.py             # OFFLINE: generate self-logging corpus, fit initial priors
└── assets/priors.json             # small JSON artifact packed into the Docker image (<2 GiB)
```

All modules **pure stdlib** (`itertools`, `math`, `random`, `json`) — the kit environment
has no numpy/scipy/networkx, so exact Blossom is off the table; greedy + single-pair
replacement local search is the allocation engine (Udwani, arXiv:2403.18059 gives an
honest 1/2-competitive bound for non-adaptive greedy).

### Data flow per invocation
```
stdin JSON -> validate schema_version -> dispatch:
  phase=ask   -> load memory, restore counters, run clarify.py, save memory, return
  phase=match -> load memory, restore scorer state, extract features, run scorer +
                 member_bandit, apply θ threshold, run allocate.py, save memory, return
```

### Pretraining / asset budget
`assets/priors.json` ≈ a 12-dim logistic prior (mean vector, covariance) fitted on
hand-labelled rollout outcomes from *our own* greedy+random logging policies run on
the 6 training pools only. Estimated ~50 KB. Trivially inside the 2 GiB image budget.

> **Reconciliation with the prior empirical deep-dive (`RESEARCH_NOTE.md` + `analysis/`):**
> that note already measured the achievable headroom: candidate policy primary score
> 0.417 vs greedy 0.403 (within noise, SE ≈ 0.03), while intermediate metrics move
> clearly (+10% mutual acceptances in development, +17% in shift). It also established:
> (i) primary-score differences below ~0.1 are not resolvable at 20 seeds; (ii) the
> `sparse` ceiling number is unverified (one consistency check was interrupted);
> (iii) early online weight-learning was unstable (needs step decay, prior anchoring,
> clipping); (iv) DE-based weight tuning is dominated by noise and is worse than
> maximum-likelihood online logistic regression. **This plan keeps those findings and
> adjusts expectations accordingly: the honest headline is the ceiling analysis plus
> intermediate-metric gains, not a large primary-score jump.**

## 4.2 Development pipeline (do in this order)

### Phase 0 — setup (day 0)
1. Working baseline pipeline clone (greedy + no_asks + random) on 1 seed × 6 variants.
2. `python evaluate.py --baseline greedy ...` sanity, confirm ~0.5 primary.
3. Verify the interrupted `analysis/consistency.py` run to settle the `sparse` ceiling.
   NOTE: the kit environment is **stdlib-only — `numpy`/`scipy`/`networkx` are
   unavailable**, so allocation is pure-Python greedy + single-pair-replacement local
   search (Udwani, arXiv:2403.18059 gives an honest 1/2-competitive bound for
   non-adaptive greedy; Drummond & Boutilier IJCAI 2013 justify targeting minimax
   regret over exact stability under a query budget).

### Phase 1 — allocation upgrade (day 1)  *highest low-risk win*
Replace `baseline_match` greedy-fill with **general-graph matching** (greedy +
single-pair-replacement local search), scored by the *baseline's own* soft-count (no
learned model yet). Expect: sparse + cold_start variants improve; development variant
roughly flat (its feasible graph is dense and greedy already realises ~89% of the edge
cap there).
- deliverable A: 6-seed × 6-variant results table, greedy-fill vs local-search matcher.

### Phase 2 — clarification VOI (day 2)
Implement §3.5. Expect: no_asks coverage 0.17 → toward greedy's 0.46 and maybe past it,
primary score should follow coverage up.
- deliverable B: ask-budget ablation 0 / 4 / 8 / 12 units/day.

### Phase 3 — pair scorer (day 3–4)
Implement §3.3 offline pretraining + online update, with the stabilisation measured in
RESEARCH_NOTE.md §10 (step decay, prior anchoring to the development weights, gradient
clipping). Evaluate with allocation already in.
- deliverable C: scorer vs baseline soft-count; report ranking discrimination (top vs
  bottom decile), not absolute calibration — the model is known to be miscalibrated in
  level but sound for ranking, and probabilities do not affect ranking (PS §10).

### Phase 4 — member bandit (day 5)
Implement §3.4 as pool-prior shrinkage only. Per RESEARCH_NOTE.md §0.5, expect no
resolvable primary-score gain from this arm (1–2 observations per member); keep it for
the ablation table so the claim "member latents are not learnable at this data rate" is
itself evidenced.
- deliverable D: ablation with bandit disabled (fixed pool mean instead).

### Phase 5 — sequencing + drift-aware scheduling (day 6)
Implement θ as a *sequencing* device (front-load high-value edges), plus the drift
response (front-load before day 35, detected from the observed acceptance-rate drop).
- deliverable E: θ ∈ {0.15, 0.25, 0.35} as a soft-ordering parameter across all 6
  variants, plus the disabled ablation.

### Phase 6 — full harness + honest reporting (day 7–8)
Add the rolling posterior-predictive coverage check (fraction of matured introductions
whose observed acceptance falls outside the 90% posterior interval) as both the
drift-detection trigger and the report's answer to "how would the system notice a wrong
belief?" — see `05_explore_exploit_notes.md` §4.
Run the official `evaluate.py --seeds <our-declared-seeds> --variants all`;
compose results tables; write the technical note per examples/REPORT_GUIDE.md (#1–#8).

## 4.3 Ablations required by the rules (and our mapping)

| Rule requires ≥1 ablation | Our choice |
|---|---|
| Hypothesis-driven ablation | **Disable online learner updates** (fixed `development` prior weights) — isolates the `shift` handling, which RESEARCH_NOTE.md identified as the single largest defensible win. |
| We will also include | Disable soft-field acquisition (value model starved); disable hard-bundle clarification (no_asks); greedy-fill allocation vs local-search matcher; hard-only vs hard+soft acquisition; no drift-aware scheduling (front-loading off); θ=0 (match everybody). |

## 4.4 Honest-evaluation choices (avoiding known traps)

- **Leakage**: features stored in memory at *assignment* time; scoring a pending outcome
  uses that snapshot, not today's state. This is verified by a test that walks a
  fake `field_observed_day` forward and asserts no hard field flips sign post-hoc.
- **Right-censoring**: only matured outcomes (both responses or deadline elapsed, date
  emitted + 3d window, or 30d window elapsed) enter gradient/Beta updates; pending ones sit
  in a `pending` list keyed on expected observation day.
- **Selective labels**: never assume "not introduced ⇒ bad pair". Never impute a Yes for
  unobserved second-meeting intention (that is *missing*, not *negative*).
- **No propensity assumptions**: because `propensity = null` on the supplied data, we do
  NOT claim IPS. All "offline evaluation" is done as full policy-vs-policy simulator
  comparisons on public seeds; the note explicitly states we do not run IPS on the
  shipped snapshots (citing the Hayashi et al. RecSys 2025 argument that reciprocal-market
  OPE without propensities is unreliable).
- **Pool hygiene**: all offline fitting and hyperparameter tuning uses the 6 training
  pools only. Validation pools used once for a single final sanity check. Dev-test pools
  used exactly once at the end for the reported number.
- **Small-N honesty**: at 20 seeds the MSMI denominator counts average ~1 MSMI+ per
  episode ⇒ primary score noisy at ±0.2 per seed. Report ±1σ over seeds as instructed.

## 4.5 Risks and mitigations

| Risk | Mitigation |
|---|---|
| 10 s/invocation violated by matching on dense 200-v graphs | Cap to top-K edges by score (K=400) before matching; measured <1 s even for greedy on 200 members (shipped `inference_seconds` ≈ 7.7 with process startup). No numpy/networkx available — pure Python only. |
| Scorer overfits to logging policy (our own corpus) | Two logging policies (greedy and random-feasible) each produce 1 corpus; report both; use **common random numbers** (same seeds across candidates — measured 1.7× SE reduction) |
| Online weight learning unstable (observed: `pace = −0.69` on cold_start at lr=0.35 with weak L2) | Step decay, prior anchoring to the development weights, gradient clipping; the signal is real (`pace = 1.19, lifestyle ≈ 0` recovered on shift) — the optimiser needs taming |
| Model miscalibration breaks threshold rules | Rank-only decisions; no hard value bar (probabilities do not affect ranking, per PS §10) |
| Memory >1 MiB for late episodes | Bounded at ~4 × 200 introductions × 200 B = 160 KB; unit test verifies |
| Process restart loses in-flight state | All state is in `memory` JSON; validated by simulating two policy invocations in a row |
| Private worlds differ from public in variant weights | Keep hyperparameters variant-agnostic; derive drift response from observed acceptance-rate drop, not variant labels |
| Small-N noise misread as improvement | Primary-score differences below ~0.1 are not resolvable at 20 seeds (SE ≈ 0.03); report ±1σ and the funnel decomposition so volume-vs-ordering gains are distinguishable |

## 4.6 What to write into the technical report

Include the **Θ / ψ / computed** parameter-hygiene table from
`06_hard_rules_soft_goals.md` §5 (world-estimated vs chosen-and-ablated vs solver-computed
items) and a residual-verification unit test (no proposed pair ever violates
`eligibility()` under any scorer output; each feedback event enters at most one posterior
update) — the deck's feasibility-workflow and no-double-counting disciplines.

Python outline of the note (markdown → PDF per submission guide):

1. Hypothesis — a reciprocal pair scorer learned online from delayed/censored feedback,
   paired with global max-weight allocation and VOI-ranked clarification, raises MSMI/100
   over greedy. Mechanism: (i) learned pair preference vs uniform soft-count, (ii) global
   allocation vs greedy fill, (iii) budgeted clarification vs none, (iv) online member
   posteriors for cold-start and drift robustness.
2. Feasibility definition — exactly `eligibility()` semantics, no re-parameterisation.
3. Allocation — general-graph matching (degree cap 1 per person; the graph is not
   bipartite), greedy + single-pair-replacement local search, weight = P̂(MSMI) used for
   ranking/sequencing.
4. Clarification — VOI-per-unit, with declined/unavailable deferral.
5. Online learner — §3.1–§3.5.
6. Baselines — greedy / no_asks / random on identical seeds and variants.
7. Results tables + funnel charts + uncertainty.
8. Ablations + failure analysis — where it loses (sparse geography, extreme cold start
   with < few members still arrives after day 50, etc.).
9. Limitations — invented outcomes; no external deployment evidence; clarify what a
   production version would additionally validate before shipping.
