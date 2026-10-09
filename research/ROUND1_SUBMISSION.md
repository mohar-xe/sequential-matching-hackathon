# Sequential Matching Hackathon — Round 1 Research Submission

**Vouchsafe Sequential Matching Hackathon · Participant release 1.0.0 · 9 Oct 2026
· Rev 3 (adds Oviya's implemented team policy + ablation ladder; Rev 2 fixed
same-seed ceilings, joint probability, power analysis, weight-learning fix)**

**Authorship note:** §§1–7, 10 are joint analysis (analysis scripts on main).
§8's measured ladder, the `team_policy.py` implementation, and the §12
appendix of the branch `RESEARCH_NOTE` are Oviya's work
(`team-policy-experiment` branch: `a5ebf9a`, `c81afc0`), independently
verified by the procedure in `research/analysis/verify_team_policy.py`.

This is our frozen Round 1 approach note. It covers problem interpretation,
hypothesis, reciprocal feasibility, probability analysis, allocation,
clarification policy, missing/delayed-data handling, planned baselines,
ablations, and failure cases, per `docs/SUBMISSION.md`.

All numbers were produced by the 9 stdlib-only scripts in `research/analysis/`
against the supplied public simulator. Every person, conversation, and outcome
is synthetic. Nothing here is evidence about real relationships.

**Kit provenance:** organiser starter kit vendored verbatim in
`The-Sequential-Matching-Problem/` (upstream
`RomeoJulietLove/The-Sequential-Matching-Problem`, pinned commit
`a8e26b3`, release 1.0.0, MIT + synthetic-data licence). Our work is confined
to `research/`. Kit verified with `python -m unittest -v` (22 tests, OK) and
`verify_data.py`. Analysis and any future policy are pure Python standard
library — `numpy`/`scipy`/`networkx` are unavailable, and assessed inference is
2 cores / 1 GiB / 10 s per call, offline, CPU-only.

---

## 0. Executive summary

1. **The metric is a near-fixed constant of the world, not a tunable quantity.**
   The reachable feasible-edge set caps introductions at ~103 (`development`) /
   ~21–23 (`sparse`, seed-dependent); MSMI per reachable edge averages ~1–1.5%.
   The primary score (MSMI per 100 arrived members, mean over 6 families × 20
   seeds) lives in the range 0.2–1.0 because a single episode yields ~1 MSMI
   event. Greedy is at 86–99% of the corrected ceiling in `development` and
   `sparse` on identical seeds — the remaining headroom is ordering
   (`shift`, `drift`) and `cold_start` volume, not volume in general.
2. **98.5% of every introduction is lost to mechanisms no policy controls**
   (response rates, fixed 0.78 date rate, and above all a 3-day answer window
   that discards ~64% of otherwise-successful outcomes).
3. **Feasibility, not desirability, binds.** Only **1.24%** of all pairs are
   hard-feasible in `development` (0.24% in `sparse`); one third of members can
   never be matched at all (a `declined` hard field is permanent).
4. **The only reliable pair-level value signal is soft-field agreement on 4 of
   7 fields.** Per-member latents get 1–2 observations per episode and are not
   learnable at this data rate.
5. **Therefore this is an acquisition problem, not an RL problem.** Buy
   hard bundles to expose the graph, spend leftover budget on the 4
   decision-relevant soft fields, score by fit marginalised over the unknown,
   allocate by maximum-weight (general-graph) matching, learn the 4 weights
   online so `shift` is handled without knowing the variant.
6. **Honest headroom is 1.3–1.6×, not 5× — and the headline is underpowered
   ~20×.** Our prototype scores 0.417 vs greedy's 0.403 — *inside the noise
   band* (SE ≈ 0.03–0.04; paired SD ≈ 0.375 ⇒ minimum detectable effect at
   72 episodes is ~0.12, and detecting our +0.014 would take ~470 full
   evaluations). The powered evidence is intermediate metrics on high-count
   events (+10% mutual acceptances in `development`, +17% in `shift`, paired
   on identical seeds). The ceiling analysis itself is our strongest
   deliverable: it tells the judges exactly why the metric barely moves.
7. **We now have an implemented policy, not just analysis.** Oviya's
   `team_policy.py` (stdlib-only, POLICY_INTERFACE-conformant) fixes two
   confirmed v1 bugs — 47.7% of v1 soft asks go to permanently unmatchable
   members, and v1's cumulative `harvest()` retrains each feedback event up
   to 55× — and adds exact general-graph MWM (268/268 fuzz-optimal) with
   tiered VOI acquisition. Measured ladder (6 seeds × 6 families, paired):
   greedy 0.333 → v1 0.347 → VOI+greedy-match 0.389 → full adaptive 0.444 →
   full fixed-prior **0.514**. Absolute gains sit at the resolvability
   threshold (min detectable ~0.175 at n=36), so these are directional, not
   proven — but the mechanism metrics and the fixed-prior-beats-adaptive
   finding (predicted by §5.1's noise analysis) corroborate the structure.

---

## 1. Problem interpretation

**The job in one line:** twice a day for 60 days (+40 follow-up days with no
new actions), decide (a) which questions to ask and (b) which pairs to
introduce, to maximise mutual second-meeting intentions (MSMI) per 100 arrived
members.

**Policy loop (per day):**

```python
asks  = policy.ask(state)        # ≤ 12 units: constraints bundle = 3, named soft field = 1
state = sim.observe()            # refreshed with ask results
pairs = policy.match(state)      # disjoint feasible 2-tuples; empty list = wait
sim.advance(pairs)
feedback = sim.receive_feedback()  # only matured feedback; the rest is censored
```

The process is killed and restarted on every call: everything remembered must
fit in an explicit JSON `memory` blob (≤ 1 MiB). JSON in/out ≤ 1 MiB per
message, 10 s per call including startup. No network, no GPU.

**Scoring.** MSMI = both said yes → date within 30 days of assignment → both
said "yes, meet again" both within 3 days of the date. One qualifying pair
counts once. Primary score = equally weighted mean over 6 scenario families
(standard, sparse geography, cold start, delayed dates, shifted weights,
changing response conditions) × 20 private seeds. Tiebreakers: coverage, then
mutual acceptances, then lower ask cost, then lower inference time.

**What is withheld:** world seed, who has not arrived, hidden preferences,
reply likelihoods. Only observable state, paid answers, matured feedback, and
self-carried memory cross the boundary. Reading simulator internals or
reconstructing hidden answers from public seeds is forbidden.

**Three facts about soft fields that reframe the challenge**
(`analysis/soft_fields.py`):

- (a) **Fixed:** values are drawn once and never change (0 of 200 members
  changed over 60 days). Asking is a one-time, lossless purchase. The `drift`
  world shifts the *scoring formula* (−0.5 logit from day 35), not preferences.
- (b) **Mostly invisible:** P(one person observed) ≈ 0.41 (normal) / 0.28
  (cold start), but a field enters the score only when *both* sides are
  observed — so only 11.4% of pairs (3.8% cold start) have complete fit before
  asking, vs 56% after asking the four fields (0.93 observed each). Asking is
  5× (normal) to 15× (cold start) more coverage, and flattens both worlds.
- (c) **Only 4 of 7 matter:** `relationship_goal` (0.70 / 0.25 in shift),
  `relationship_pace` (0.40 / **0.80**), `lifestyle` (0.25 / **−0.25**),
  `conversations` (0.20 / 0.50). `emotional_availability`,
  `space_for_relationship`, `relocate` never enter the scorer (force-changing
  them leaves MSMI bit-identical). **Never ask about those three.** The
  negative `lifestyle` weight in `shift` is why weights must be *learned*, not
  hard-coded, and why "match on everything" is wrong there.

---

## 2. Reciprocal feasibility: why introductions are capped

`analysis/why_cap.py` (first failing hard rule per pair, truth, 20 seeds):

| first rejection | development | sparse |
|---|---|---|
| `who_to_meet` | 57.07% | 54.14% |
| `age` | 26.06% | 25.60% |
| `geography` | 12.04% | 19.37% |
| smoking / children / structure / schedule | ~3.6% combined | ~0.6% combined |
| **FEASIBLE** | **1.24%** | **0.24%** |

`who_to_meet` dominates because `wants` is size-1 70% of the time; mutual
acceptance needs it in both directions. Age rejects 26% (each member accepts
only age ± U{3..10}).

`analysis/ceiling.py` restricts to the **reachable** set (no declined hard
field) and computes the feasible graph on truth (seeds 41–60):

| family | matchable / 200 | |E| reachable | max matching | best 8-round schedule |
|---|---|---|---|---|
| development (+delayed/shift/drift, structurally identical) | 133.4 | 103.1 | 34.5 | 99.8 (97%) |
| sparse | 131.2 | 20.9 | 14.7 | 20.9 (100%) |
| cold_start | 120.0 | 90.5 | 31.8 | 89.2 (99%) |

P(matchable) = 0.667 (development), 0.656 (sparse), 0.600 (cold start) — one
third of people can never be matched because `resolve_asks` never resolves a
declined field. `|E|` is a hard ceiling (repeats forbidden all episode).
The observation layer sees only ~30% of truth-feasible pairs at a time
(1445 visible vs 3369 locked in development).

**Resolved: the apparent sparse contradiction was a seed-set mismatch, not a
simulator bug.** The draft compared `|E| = 20.9` (seeds 41–60) against greedy
realising ~22.6 (seeds 101–112) — different worlds. Recomputed on identical
seeds: sparse `|E| = 22.8` on seeds 101–112 vs greedy 22.6 realised there
(**99% saturated**); on seeds 101–103 every assigned pair was verified a
subset of the reachable truth-feasible set (unreachable = 0, not-even-truth =
0 on all three). This must hold generally: an observed-feasible pair copies
truth values exactly, and feasibility requires fully cleared endpoints, hence
no declined hard field on either side — i.e. membership in the reachable set
by construction. **Consequence for the narrative:** `sparse` is saturated, so
it is *not* a volume opportunity; quote only same-seed ceiling pairs
(development ~89%, sparse ~99%, cold_start ~86%: 77.6 realised of 90.5).
Remaining headroom is `cold_start` volume + `shift`/`drift` ordering.

**Clarification is supply-limited, not budget-limited.** Total budget is
720 units (240 bundles); askable supply is ~80 members, exhausted by ~day 20
(baseline bundles/day: `4×12, then 1,0,0,2,…`). There is no budget headroom to
exploit. The policy consequence is deliberately narrow (see §5): clear every
clarifiable member early in decision-relevance order; the only real
acquisition decision is the leftover spend on the 4 soft fields (~311 units,
fits with room to spare).

---

## 3. Probability analysis (optional estimates — ranking only)

**Recovered generative law** (`analysis/prob_model.py`). Per assigned pair,
one shared shock `shared ~ N(0, 0.45²)`, then per direction
`accept ~ Bernoulli(σ(−0.25 + bias_a + fit(i,j) + shared + drift(t)))`,
`respond ~ Bernoulli(response_rate_a)`. Date iff both responded and accepted
(p = 0.78); `date_day = t + max(delay) + U{1..14}`; second yes
`σ(0.15 + second_bias + shared + 0.4·[goal match])`, answered with
`response_rate`, delay `U{1..5}`. `fit` is a weighted sum over the 4 soft
fields only — hard constraints gate feasibility and contribute nothing to
value.

The two directions are **correlated** (ρ ≈ 0.43 via `fit` + `shared`), so
multiplying marginals understates mutual acceptance. The joint must integrate
over `shared` (41-node midpoint rule, validated to 4.8e-6). The problem
statement explicitly warns against the independence assumption; it is
quantitatively wrong here. A second, subtler correlation: kit.py draws **one**
`shared` per pair driving **both** the acceptance stage and the second-meeting
stage, and each stage independently requires a `response_rate` coin. The
factorized `p_msmi_truth` (E[acc]·E[sec], one response product) therefore
over-predicts absolute levels by ~1.4×; the corrected joint form
`p_msmi_joint` (single E over `shared` of all four sigmoids, both response
products) is in `prob_model.py`. Ranking is identical under either form.

**Validation** (`analysis/validate_model.py`: 60 reachable pairs × 40 fresh
seeds):

| family | empirical P(MSMI) | model | 95% CI | verdict |
|---|---|---|---|---|
| development | 0.0121 | 0.0155 | [0.0077, 0.0165] | inside |
| shift | 0.0079 | 0.0128 | [0.0044, 0.0115] | marginally outside |
| drift (day 0) | 0.0121 | 0.0155 | same | inside |

**Discrimination is what matters** (probabilities do not affect ranking):
top decile 0.0409 predicted / 0.0333 empirical vs bottom 0.0022 / 0.0000. Usable
for ranking despite slight level over-prediction.

**Funnel** (`prob_model.py::decompose`, greedy, development, 12 seeds):

| stage | count | cond. rate | controllable? |
|---|---|---|---|
| introductions | 91.6 | — | yes, capped ~103 |
| both responses | 52.0 | 57% | no (latent, 1 sample/person) |
| mutual acceptance | 12.6 | 24% | **yes, via fit + ordering** |
| date happened | 10.3 | 82% | no (fixed 0.78) |
| both second-yes | 3.7 | 36% | marginally |
| both within 3-day window | 1.3 | 36% | **no — U{1..5} ⇒ (3/5)²** |
| **MSMI** | **~1.0** | | |

The 3-day window alone destroys ~64% of fully successful outcomes and admits
no action.

**Value spread** (`analysis/value_spread.py`, reachable edges): p25/median/p75/
p95 ≈ 0.0012/0.0031/0.0076/0.0172 (development); top-half vs bottom-half
realised lift 1.38× on fit alone → 1.48× with oracle `response_rate`. Cheap
legal signal captures most of it; member latents are not worth the risk at
1–2 obs/member. Ordering helps by *sequencing* (early high-value edges matter
for `drift`/`delayed`), not by manufacturing volume — 89% of edges are already
consumed.

**Why not black-box weight search** (`analysis/de_feasibility.py`, Kaggle
4-core run, 288 episodes, 1143 s, log in `results/kaggle_de_feasibility.log`):
one episode 3.97 s; one fitness eval (12 seeds × 6 families) ≈ 4.8 min; DE at
50×100 = 132–265 h. Achievable weight-space range 0.132 < single-run noise
0.356; a deliberately sign-flipped vector scores 0.410 vs 0.424 (correct
shift) and 0.340 (correct development) — the landscape is unidentified.
Common random numbers cut SE 1.7× (0.073 → 0.044) and are mandatory for any
tuning; fitting 4 params on fixed seeds is the overfitting mode the problem
statement warns about. **Maximum likelihood on revealed responses (online
logistic regression) dominates DE:** it uses gradients, needs no repeated
simulation, and returns a calibrated quantity.

> ✅ **Resolved — the "2× under-prediction" was a mislabelled comparison,
> and its direction was flipped.** The draft's "Σ ≈ 0.56" is the *single-batch*
> max-weight-matching sum (verified: 0.57–0.69 on seeds 41–44), not Σ over all
> reachable edges. True Σ_all ≈ 1.59 (seeds 41–60, factorized form) vs greedy
> realised ~1.0 — i.e. the model *over*-predicts ~1.6×, exactly as the two
> identified formula gaps predict (missing second-stage response product
> 1/0.585 ≈ 1.7× over, partly offset by ~1.2–1.3× shared-reuse boost measured
> by Monte Carlo). The corrected joint gives Σ ≈ 1.07–1.15 (seeds 41–42) vs
> ~1.0 realised: greedy is at **~87–93% of the corrected absolute ceiling** in
> development. Rankings — the only thing the policy uses — are unaffected
> under either form, which is why single-pair replay validated while levels
> did not.

---

## 4. Hypothesis and proposed policy

**Hypothesis.** A reciprocal pair scorer learned online from delayed/censored
feedback, paired with global maximum-weight allocation and VOI-ranked
clarification, raises MSMI/100 over greedy via (i) learned pair weights vs
uniform soft-count, (ii) global allocation vs greedy fill, (iii) budgeted
soft-field acquisition vs none, (iv) drift-aware sequencing. Full RL is
explicitly rejected (Learn2Match's friction-loss result predicts its failure
when welfare is saturated and the gap is friction); external LLM scoring is
rejected (no grounding, templates only, offline CPU).

Three separable modules (keep normalisation, modelling, allocation separate):

**(a) Acquisition — hard-first, matchability-gated.** Constraint bundles
(3 units) for available members with any hard field `not_asked` and no
declined hard field — this alone exposes the graph. **Gate (Oviya's fix):
v1's soft-ask step checked only the soft field's own status, so 183 of 384
soft asks (47.7%, seed 101 development — independently replicated via
`verify_team_policy.py`) went to permanently unmatchable members. No soft
query is now issued outside the matchable set.** Remaining budget (1 unit
each) on the 4 decision-relevant fields (never the 3 dead fields),
prioritised by tier (§5). ~311 units for all reachable members fits inside
720 with room to spare. Never ask declined / unavailable members.

**(b) Value model.** Per field `k` observed on both sides: `+w_k` if equal
else `−w_k/2`. Unobserved/declined: marginalise under the uniform prior
(`E[contribution_k] = w_k·(1.5/|S_k| − 0.5)` — exact here since missingness is
uniform-random; removes the `None`-equals-`None` artefact). Score
`logistic(−0.25 + Σ E[contribution])`, intercept calibrated from observed
response rates. Report directions separately; integrate over `shared` for the
joint.

**(c) Weights — online logistic regression** on matured directional responses
(features = agreement on observed subset, label = recorded Yes; unbiased for a
main-effects model). Regularise toward the development prior, clip steps, decay
/ re-widen on shift detection. First attempt (100+ SGD steps, lr 0.35, weak
L2-to-zero) was unstable (`pace = −0.69` in cold_start) but recovered the
right direction on shift (`pace = 1.19, lifestyle ≈ 0`) — signal exists,
optimiser needs taming. **Fix validated on synthetic revealed-preference data
(true shift weights):** on clean iid data (n=400) the old hyperparams recover
well (pace 0.83 vs true 0.80 — confirming the signal is real); in a
cold-start-like small-n regime (n=33, 60% missingness) old-style updates
overshoot and sign-flip (`conversations` −0.19 vs true +0.50), while the fixed
rule (lr 0.1 with 1/t decay, L2 anchored to the development prior instead of
zero, gradient clip 1.0) holds smaller excursions with no sign flips on the
two dominant weights. Bias-for-variance trade is correct here since noise
dominates. **Streaming fix (Oviya): v1's `harvest(state)` re-fed cumulative
feedback every day, so a day-5 event was SGD-updated ~55 times by day 60 —
part of the observed instability. The implemented learner is strictly
single-pass (`seen_events` in memory, one update per matured response) with
`η = 0.25/√(1+n)`, anchor λ = 0.10 to BASE_W, clip 1.0.** Episode-level
confirmation is Round 2 work.

**(d) Allocation — exact maximum-weight matching on the general
(non-bipartite) eligible graph.** ~47% of women accept meeting women, so
bipartite solvers are invalid. **Implemented (Oviya): branch-and-bound over
connected components with greedy warm-start, stdlib-only, <1 ms/day —
verified 268/268 optimal against brute force on fuzz graphs
(`verify_team_policy.py`).** Weights = P̂(MSMI), used for
**ranking/sequencing**, not hard thresholds (model is miscalibrated in level;
probabilities do not affect ranking). The −0.10 post-day-35 edge offset is a
declared fixed ψ (mild late-edge filter via the w > 0 rule), *not* a drift
detector — triggering from the observed acceptance-rate drop is Round 2 work.

**Per-family handling:** sparse → zone-clustered clearing (whole zones, so
cleared members concentrate in dense sub-blocks); cold_start → clear bundles
as early as possible (4/day cap binds ~25 days); shift → online weights (only
defensible large win, tightly anchored); drift → front-load high-value edges
before day 35 (design: trigger from the observed acceptance-drop; implemented:
fixed day-35 offset — detector is Round 2 work, see §4d); delayed →
flat few-percent loss (extra delay independent of assignment day — the
"introduce early" intuition is wrong).

---

## 5. Clarification policy (narrow by design — supply binds, not budget)

Because askable supply (~80 members) exhausts by ~day 20 of a 720-unit budget,
there is no VOI problem in the classic "spend or save" sense. The policy is
three fixed rules, in order — the only ranked choice is *which* member to
clear first:

1. **Clear every clarifiable member early** (constraint bundles first).
   Uncleared members are ordered by zone-clustered score (2 × cleared
   same-zone peers + uncleared same-zone peers) so cleared members concentrate
   in dense sub-blocks — the sparse prescription (Oviya).
2. **Order the queue by decision-relevance** (`score(m) = n_decision-relevant
   friends(m) × Ê[agreement] / 3`, counting only pairs that could flip the
   day's argmax — `VOI(q) = 0` whenever no answer changes the best action).
   Time-adaptive bar: buy freely at day 0, require `n_friends ≥ 2` by day 20.
3. **Spend all leftover budget on the 4 decision-relevant soft fields**
   (never the 3 dead fields), tiered: Tier 0 (members in a currently feasible
   pair) → Tier 1 (hard-cleared pool) → Tier 2 (uncleared matchable); within
   tiers by |w_k|, preferring fields whose partner side is already observed
   (one query resolves fit between +w_k and −w_k/2). This leftover — not the
   hard-bundle ranking — is the real acquisition lever: it lifts pair
   coverage with complete fit from 11% (4% cold start) to 56%.

No other machinery: soft asks never unblock, so they are never prioritised
over bundles; declined/unavailable members are never asked.

---

## 6. Missing, delayed, and selective data

- **Declined ≠ not_asked:** declined hard fields are permanent and make the
  member unmatchable; never spend on them. Null value is never zero/rejection.
- **Missing responses are censored, not negative:** `introduction_response`
  null, un-dated pairs, unanswered second intentions are *unresolved until the
  window matures* (~day-43 horizon), never imputed as No.
- **Leak-free updates:** store the feature snapshot at assignment day
  (`field_observed_day` proves later fields were unknowable); score pending
  outcomes from that snapshot. Each feedback event enters at most one posterior
  update (directional yes → member posteriors; derived mutual acceptance never
  re-fed as independent sample); no feature re-models what `shared` already
  induces.
- **Right-censoring:** pending queue keyed on expected observation day; only
  matured outcomes enter gradients/Beta updates. Late-episode introductions
  resolve inside the evaluator's 40-day follow-up but are never seen by the
  policy — track outstanding introductions explicitly.
- **Selective labels / no IPS:** outcomes exist only for pairs we chose;
  `propensity` is null throughout, so no inverse-propensity claims on shipped
  snapshots. All comparisons are controlled simulator episodes on identical
  seeds. Pool-level splits only (6 train / 2 val / 2 dev-test); never row-level.
- **Memory:** per-member Beta counters + 12-dim scorer posterior + pending
  queue ≈ low KBs (≈10 KB JSON), comfortably inside 1 MiB. All state in
  `memory`; validated across simulated restarts.

Exploration: Thompson sampling over pool-prior posteriors (κ ≈ 4) — Thompson
noise *is* the exploration budget (course Table 2: greedy locks out the best
option 35% of the time while looking calibrated; TS regret 13.5 vs greedy
91.5). Rank-only, no hard cutoff (self-transitions explore→exploit as σ
shrinks). Rolling posterior-predictive check (fraction outside 90% interval)
is both drift trigger and wrong-belief detector: sustained one-sided
miscoverage → re-widen + temporarily lower sequencing bar. Member posteriors
kept as cheap shrinkage only — headline claim never rests on them.

---

## 7. Baselines and experimental plan

- **Baselines (identical seeds, all 6 variants):** greedy (soft-match count),
  no-clarification, random-feasible via `evaluate.py`.
- **Seeds:** declared training seeds disjoint from validation; dev-test used
   once. MSMI ≈ 1/episode, paired SD ≈ 0.375 (measured, `de_feasibility`) ⇒
   at our 72 episodes (12 seeds × 6) the minimum detectable primary-score
   effect (80% power) is **~0.12**; our +0.014 would need **~470 full
   evaluations** (even +0.05 needs ~37). **The headline metric cannot
   demonstrate an edge at any feasible seed count** — report ±1σ, never rank
   on noise, and treat mutual acceptances (~12–14/episode, paired on identical
   seeds) as the powered learning signal. A zero MSMI is not evidence of
   equivalence. Round 2 evaluation uses common random numbers throughout
   (measured 1.7× SE cut, 0.073 → 0.044) and leads with the funnel, not MSMI.
- **Report separately:** assignments, mutual acceptances, dates, MSMI,
  coverage, ask cost, missing feedback, first-introduction waits + unserved
  count, funnel decomposition (volume vs ordering), runtime. Common random
  numbers throughout.

---

## 8. Ablations (≥1 required; we run five + threshold sweep)

1. **No soft-field acquisition** (value model on ~1 observed field) — the
   "no value signal" arm; also hard-only vs hard+soft (isolates
   feasibility/value split).
2. **Fixed development weights vs online weights** — isolates `shift`; the
   single largest defensible win.
3. **Greedy-fill vs maximum-weight matching** — isolates allocation (expect
   wins in cold_start/drift-sequencing, flat in development *and* sparse —
   both saturated at 89–99% of ceiling on identical seeds).
4. **No drift-aware scheduling** (front-loading off) — isolates day-35 response.
5. **Member bandit disabled** (fixed pool mean) — evidences the claim that
   member latents are unlearnable at this data rate.
6. **θ sweep {0.15, 0.25, 0.35} as soft sequencing + θ=0 (match everybody)** —
   grid mean across variants, never hand-fit per variant.

### Measured ladder (Oviya, `team_policy_eval.py`, seeds 101–106, paired)

| Configuration | MSMI/100 | vs greedy | Mutual acc. | Isolates |
|---|---|---|---|---|
| Organiser greedy | 0.333 | — | 10.14 | reference |
| Candidate v1 | 0.347 | +4.2% | 10.56 | soft acquisition + fixed weights |
| VOI ask + greedy match | 0.389 | +16.7% | 10.50 | matchability gate + VOI tiers |
| Full (VOI + MWM + adaptive) | 0.444 | +33.3% | 10.81 | + exact matching + streaming SGD |
| Full (VOI + MWM + fixed prior) | **0.514** | **+54.2%** | **11.20** | + weight prior (no online updates) |

Per-variant MSMI (fixed-prior config): development 0.667, sparse 0.167,
cold_start 0.500, delayed 0.583, shift 0.500, drift 0.667 (greedy: 0.417,
0.083, 0.333, 0.333, 0.417, 0.417).

**Reading this honestly.** At n=36 episodes the minimum detectable effect is
~0.175: only the fixed-prior headline (+0.181) reaches it, and per-variant
cells (n=6, ±0.3 noise) are illustrative, not claims. Three things corroborate
the structure anyway: (i) the ladder is monotone across cumulative mechanisms;
(ii) mutual acceptances (+10%, high-count metric) move with the headline;
(iii) **fixed-prior beats adaptive everywhere on these seeds — including
`shift` (0.500 vs 0.417)** — exactly what §5.1's noise analysis predicts
(~50 Bernoulli trials cannot carry 4 weights; adaptation chases noise
off-shift). The adaptive arm is retained for Round 2 with tighter anchoring,
not dropped: theory says it is the only mechanism that can handle `shift`
without variant labels. Cost note: the team policy spends ~409 ask units vs
greedy's ~198 (leftover soft budget by design) — it currently *loses* the
3rd tiebreaker (lower cost); a cost-capped variant is a Round 2 ablation.
Harness smoke test (official `evaluate.py`, seed 101, development): greedy
msmi=1 → team msmi=3, both valid, 0.15 s/call — replicated independently.

---

## 9. Failure cases and limitations

- **Withheld answers:** ~1/3 of members permanently unmatchable; they stay in
  coverage/unserved denominators forever — explained, not dropped.
- **Sparse supply:** |E| ≈ 21–23 ⇒ family contributes ~0.05–0.10 for everyone
   and greedy is already at ~99% of ceiling — do not over-read differences and
   do not pitch sparse as headroom. Weight pressure re-prices scarce slots,
   never creates supply.
- **Coupling:** one intro/person at a time + shared 8-day cadence couples the
  schedule (days t, t+8, t+16…) — why global matching beats per-day greedy.
- **Delayed feedback + censoring:** day-55 intros resolve ~day 77 (seen by
  evaluator, never by policy); must not be credited as failures.
- **Instability / miscalibration:** early weight run overshot (see §4c);
   fixes (decay + prior-anchoring + clipping + single-pass streaming)
   validated on synthetic data and by code-level diagnosis; episode-level
   confirmation pending. Level miscalibration ⇒ ranking-only decisions, no
   hard value bar.
- **No open blockers.** Both former caveats are closed above (same-seed
  subset checks, 0 violations; joint-probability correction with measured
  0.70 ratio). Neither affects rankings.
- **Provenance / merge hygiene.** Oviya's branch is merged (`db51aa1`,
  history preserved, no rewrites) and the policy files relocated out of the
  vendored subtree to `research/analysis/` (`team_policy.py`,
  `test_team_policy.py`, `team_policy_eval.py`; the 5-line dispatch hook in
  the kit's `policy.py` reverted — baselines untouched, kit byte-verbatim).
  The two `scratch/` scripts cited in the branch §12 were never committed;
  both findings were independently replicated (`verify_team_policy.py`:
  waste 183/384 = 47.7%; MWM 268/268 optimal). The branch's
  `policy_v2.py`/`experiment_v2.py` (hardcoded local paths) are superseded
  iteration history, not cited.

---

## 10. What this cannot establish; reproducibility

Simulator success is a first engineering test only. Synthetic probabilities
are not calibrated for real members; conversation fragments are authored
templates for input handling, not psychological evidence. Real deployment needs
authorised inputs, current constraint checks, and human review. External
datasets/models, if any, will be declared with licences; no production data is
used. Parameters follow Θ (world-fixed, estimated: 4 weights, member latents)
/ ψ (our choices, declared + ablated: κ, λ, sequencing θ, VOI τ) / computed
(posteriors, matching, drift trigger) hygiene, with residual-verification
(no proposed pair violates `eligibility()`) and no-double-counting tests.

**Reproduce:**

```bash
cd The-Sequential-Matching-Problem && python3 -m unittest -v && python3 verify_data.py
cd ../research/analysis
python3 prob_model.py; python3 validate_model.py; python3 why_cap.py
python3 ceiling.py; python3 value_spread.py; python3 candidate.py
python3 verify_team_policy.py   # MWM fuzz (268/268) + v1 waste replication;
                                # loads team_policy.py from team-policy-experiment ref
# full ladder (needs branch checkout; ~15 min, 180 episodes):
python3 team_policy_eval.py --seeds 101,102,103,104,105,106
```

Prototype: 0.417 vs 0.403 primary (12 seeds × 6 families, within noise —
underpowered ~20×, see §7); mutual acceptances 12.6 → 13.9 development
(+10%), 11.5 → 13.5 shift (+17%), equal volume/coverage, paired on identical
seeds. **The headline is not claimed as an improvement; the powered
intermediate metrics plus the ceiling analysis are the evidence.**
