# The Sequential Matching Problem — Deep-Dive Research Note

**Round 1 research submission draft · Vouchsafe Sequential Matching Hackathon · 5 Oct 2026**

All numbers below were produced by the scripts in `analysis/` against the supplied
public simulator. Every person, conversation and outcome is synthetic. Nothing here
is evidence about real relationships.

---

## 0. Summary of findings

1. **The scoring metric is a near-fixed constant of the world, not a tunable
   quantity.** The reachable feasible-edge set caps introductions at ~103
   (development) / ~21 (sparse); MSMI per *reachable edge* averages ~0.5–1.5%.
   Introductions are therefore capped, and per-edge probability is fixed, so the
   primary score is a small residual of scheduling and ordering choices.
2. **The dominant term in the funnel is not controllable by any policy.** Of an
   introduction, ~64% get both directional responses, ~28% of those become mutual
   acceptances, 78% of those produce a date, ~33% of those get two "yes" to a
   second meeting, and only ~36% of *those* land inside the 3-day window. The
   3-day window alone destroys ~64% of otherwise-successful outcomes.
3. **Feasibility, not desirability, is the binding constraint.** Only **1.24%** of
   all 200-member pairs are hard-constraint feasible in `development`, and
   **0.24%** in `sparse`. `who_to_meet` (57% of pairs), `age` (26%) and
   `geography` (12%) do essentially all the rejecting.
4. **One third of the population can never be matched at all.** A member with any
   `declined` hard field is permanently unmatchable — `resolve_asks` never
   resolves a declined field. P(matchable) = 0.667 (development), 0.656 (sparse),
   0.600 (cold start).
5. **The only reliable pair-level value signal is soft-field agreement.** Per-member
   latents (`bias`, `second_bias`, `response_rate`) are observed at most 1–2 times
   per member over an episode, so they are *not learnable* at this data rate — a
   single Bernoulli observation leaves a posterior no tighter than the prior.
6. **Therefore the winning design is an acquisition problem, not an RL problem.**
   Spend the 12-unit daily budget to buy soft fields (1 unit each) for members
   already unblocked by a `constraints` bundle (3 units), score pairs by fit
   marginalised over unobserved fields, allocate by maximum-weight matching, and
   learn the four fit weights online so the `shift` family is handled without
   knowing the variant.

---

## 1. The problem as a POMDP

This is the formalisation implied by the kit, and it is precisely the setting of
Zong et al., *Learn to Match: Two-Sided Matching with Temporally Extended
Feedback* (arXiv:2606.06744), with two additions the benchmark does not have:
reciprocal hard-constraint gates, and an outcome definition that includes a
tight *response-latency* condition.

**State (observable).** Day `t`; arrived members with age, gender, zone; for each
of 18 questionnaire fields a value, a status in `{observed, not_asked, declined}`,
and an observation day; introduction history; matured feedback events.

**Latent state (never observed).** Per member `i`:
`bias_i ~ N(0, 0.7²)`, `second_bias_i ~ N(0, 0.5²)`, `response_rate_i ~ U(0.55,
0.98)`, `exit_day`, plus the 18 field values themselves.

**Action.** `(A_t, M_t)`: a set of clarification requests costing ≤ 12 units
(`constraints` = 3, one named soft field = 1), and a set of feasible
vertex-disjoint pairs on the eligible graph.

**Transition.** Per assigned pair `(i,j)` at day `t`:
`shared ~ N(0, 0.45²)` — **one shock shared by both directions**; then for each
actor `a ∈ {i,j}` independently,
`accept ~ Bernoulli(σ(z_a))` and `respond ~ Bernoulli(response_rate_a)`, where

```
z_a = -0.25 + bias_a + fit(i,j) + shared + drift(t)
```

Date occurs iff both responded and both accepted, with probability 0.78;
`date_day = t + max(delay_i, delay_j) + U{1..14}`; then for each actor
`yes₂ ~ Bernoulli(σ(0.15 + second_bias_a + shared + 0.4·[goal_i = goal_j]))`,
answered with probability `response_rate_a`, and reported `delay₂ ~ U{1..5}`.

**Reward.** `MSMI` requires both directional yes → a date within 30 days of
assignment → both second-meeting yes *within 3 days of the date*.

Two structural features deserve emphasis, because they drive the whole design:

- **`fit` is the only pair-level driver, and it is a function of four soft fields
  only.** Hard constraints contribute *nothing* to the success probability. They
  only decide whether the pair may be formed at all. This separates the problem
  cleanly into a *feasibility* stage (worth 3 units/member) and a *value* stage
  (worth 1 unit/member) — a distinction the greedy baseline does not make.
- **The `shared` shock is common to both directions**, so the two directional
  acceptance probabilities are correlated through `fit` and `shared` only:
  `Cov(z_i, z_j) = Var(fit) + Var(shared) = 0.166 + 0.203 = 0.368`,
  `Var(z) = 0.858`, so `ρ = 0.43`. Multiplying marginals therefore **understates**
  mutual acceptance; the joint probability must be integrated over `shared`:
  `E_shared[σ(z_i + s)·σ(z_j + s)]`. The PS explicitly warns against the
  independence assumption, and here it is quantitatively wrong.

---

## 2. The recovered generative law, and its validation

`analysis/prob_model.py` re-implements the law above in closed form, including the
`shared` integral by a 41-node midpoint rule over ±4.5σ (validated to 4.8e-6
against a 4001-point trapezoid rule).

`analysis/validate_model.py` replays individual pairs through the real simulator
(60 random reachable pairs × 40 fresh world seeds, because `rand_for` is
deterministic in `(seed, a, b, day)` and naive repetition replays one outcome):

| family | empirical P(MSMI) | model | 95% CI | verdict |
|---|---|---|---|---|
| development | 0.0121 | 0.0155 | [0.0077, 0.0165] | inside |
| shift | 0.0079 | 0.0128 | [0.0044, 0.0115] | marginally outside (under-predicts) |
| drift (day 0) | 0.0121 | 0.0155 | [0.0077, 0.0165] | inside (drift inactive before day 35) |

**Discrimination is the operative property, not calibration.** Sorting the same
pairs by the model and comparing deciles:

| | predicted P | empirical |
|---|---|---|
| top decile | 0.0409 | 0.0333 |
| bottom decile | 0.0022 | 0.0000 |

So the model is usable for *ranking* even though it slightly over-predicts the
level. Since probability estimates do not affect ranking (§10 of the PS), this is
sufficient.

---

## 3. Why introductions are capped: the feasibility cascade

`analysis/why_cap.py` attributes the first failing hard rule per pair, on truth,
over 20 seeds × 200 members:

| first rejection reason | development | sparse |
|---|---|---|
| `who_to_meet` | 57.07% | 54.14% |
| `age` | 26.06% | 25.60% |
| `geography` | 12.04% | 19.37% |
| `smoking` | 1.63% | 0.31% |
| `children_present` | 0.93% | 0.16% |
| `children_plans` | 0.44% | 0.08% |
| `relationship_structure` | 0.38% | 0.07% |
| `schedule` | 0.20% | 0.03% |
| **FEASIBLE** | **1.24%** | **0.24%** |

`who_to_meet` dominates because `wants` is a size-1 sample 70% of the time, so
P(accepts a given gender) = 0.7/3 + 0.2·2/3 + 0.1 = 0.467, and mutual acceptance
needs it in *both* directions. `age` rejects 26% because each member accepts only
`age ± U{3..10}`.

`analysis/ceiling.py` then restricts to the **reachable** set — members with no
declined hard field — and computes the feasible graph on truth:

| family | matchable / 200 | \|E\| reachable | max matching | max degree | best 8-round schedule |
|---|---|---|---|---|---|
| development | 133.4 | 103.1 | 34.5 | 10.9 | 99.8 (97%) |
| sparse | 131.2 | 20.9 | 14.7 | 3.1 | 20.9 (100%) |
| cold_start | 120.0 | 90.5 | 31.8 | 10.3 | 89.2 (99%) |
| delayed / shift / drift | 133.4 | 103.1 | 34.5 | 10.9 | 99.8 (97%) |

Verified by `analysis/consistency.py`: across all three families the number of
assigned pairs that fall *outside* the reachable set is exactly **0**, as is the
number that are not even truth-feasible. The ceiling is a true upper bound.

`delayed`, `shift` and `drift` are **structurally identical** to `development` —
they differ only in outcome parameters, so `|E|` is literally unchanged.

**`|E|` is a hard ceiling on introductions**, because `advance` rejects repeated
pairs for the whole episode. The supplied greedy baseline already realises ~92 of
103 in development — **89% of the ceiling.**

### The observation layer sees only ~30% of feasible pairs

`analysis/why_cap.py` tracks, per day, feasible pairs the observation layer can
see versus feasible-on-truth pairs blocked by an uncleared endpoint:

| family | feasible visible | feasible but locked | share visible | realised |
|---|---|---|---|---|
| development | 1445 | 3369 | 30.0% | 88 |
| sparse | 604 | 1204 | 33.4% | 20 |

### Clarification is supply-limited, not budget-limited

`baseline_asks` asks four bundles/day for twelve days and then starves:

```
[4,4,4,4,4,4,4,4,4,4,4,4,1,0,0,2,0,1,0,0,2,0,0,0, ... ]   # bundles/day, development
```

Total budget is 60 × 12 = 720 units (240 bundles); the askable supply is ~80
members and is exhausted by ~day 20. So **there is no ask-budget headroom to
exploit**, and the only efficiency question is *which* members to clear.

---

## 4. The funnel: what is and is not controllable

`analysis/prob_model.py::decompose` walks the exact metric chain. Greedy,
development, 12 seeds:

| stage | count | conditional rate | controllable? |
|---|---|---|---|
| introductions assigned | 91.6 | — | **yes**, capped at ~103 |
| both directional responses | 52.0 | 57% | no — `response_rate` latent, 1 sample/person |
| mutual acceptance | 12.6 | 24% | **yes**, via `fit` and ordering |
| date happened | 10.3 | 82% | no — fixed 0.78 |
| both second-meeting yes | 3.7 | 36% | marginally — `second_bias` latent |
| both within the 3-day window | 1.3 | 36% | **no** — `U{1..5}`, so (3/5)² |
| **MSMI** | **~1.0** | | |

**The 3-day window discards ~64% of fully successful outcomes.** This is a
property of `randint(1,5)` and admits no policy action. Combined with the 0.78
date rate and the response rate, roughly **98.5% of every introduction is lost to
mechanisms outside the policy's control** — which is why the primary score lives
in the range 0.2–1.0 and why single-episode MSMI is ~1 event.

---

## 5. The lever that actually exists: value ordering on a scarce edge set

`analysis/value_spread.py`, on the reachable edge set:

| family | P(MSMI) p25 / median / p75 / p95 | top-half vs bottom-half realised lift |
|---|---|---|
| development | 0.0012 / 0.0031 / 0.0076 / 0.0172 | 1.38× (fit only) → 1.48× (+ response rate) |
| sparse | 0.0010 / 0.0027 / 0.0065 / 0.0160 | 1.44× → 1.56× |
| cold_start | 0.0013 / 0.0035 / 0.0081 / 0.0190 | 1.19× → 1.22× |

### 5.1 Would a global optimiser help? No — measured, not assumed

Differential Evolution is the natural thing to reach for when you have a small
continuous parameter vector and an expensive black-box objective. The four fit
weights are exactly that: 4 continuous parameters, scalar fitness (the primary
score), no gradients. So it is worth testing rather than dismissing.

`analysis/de_feasibility.py`, run on Kaggle (4 cores, 288 episodes, 1143 s;
full log in [`results/kaggle_de_feasibility.log`](results/kaggle_de_feasibility.log)):

| measurement | value |
|---|---|
| cost of one episode | 3.97 s (4 cores) |
| cost of one fitness evaluation (12 seeds × 6 families) | 285.9 s ≈ 4.8 min |
| DE budget, population 50 × 100 generations | 5000 evals = **132–265 h** |
| achievable range of the weight space (4 vectors) | **0.132** primary score |
| paired standard deviation of a single seed-family run | **0.356** |
| unpaired SE / paired SE (common random numbers) | 0.0730 / 0.0442 (**1.7× reduction**) |
| paired runs needed to detect a +0.02 effect at 95% | **1353** (226 full evaluations) |

**The verdict is decisive, and it is a structural argument, not a compute one.**
The entire range that DE could possibly exploit (0.132) is *smaller than the noise
on one run* (0.356). Its selection step — replace a parent when the trial vector
scores better — would be operating almost entirely on noise, which is the standard
way black-box optimisers manufacture an apparent improvement that does not
transfer. The measured per-family scores show this directly: a **deliberately
sign-flipped** weight vector (all four weights negative, `scrambled`) scores
**0.410**, against **0.424** for the correct shifted-world weights and 0.340 for the
correct development weights. If reversing every sign costs nothing, the landscape
is not being identified.

Two secondary findings:

- **Common random numbers are mandatory, not optional.** Evaluating all candidate
  vectors on the *same* seeds cuts the standard error 1.7× (0.073 → 0.044). Any
  credible tuning of these weights has to do this; comparing means on independent
  seed sets throws away most of the available signal.
- **Fitting 4 parameters on a fixed seed set is exactly the failure mode the PS
  warns about** — §10 requires results "across multiple seeds … rather than only a
  favourable run". With 4 free parameters, 5 folds, and noise larger than the
  effect, cross-seed overfitting is close to guaranteed. The 20 private seeds are
  what the assessment uses; tuning on the public ones does not transfer.

**The correct tool for these 4 parameters is maximum likelihood on the revealed
responses**, i.e. online logistic regression — it uses the gradient information DE
throws away, needs no repeated simulator evaluation, and produces a *calibrated*
quantity rather than a bare argmax. DE would return a worse answer to the same
question, at ~130 h of compute, with a higher chance of not transferring.

Three conclusions:

1. **The spread is shallow.** p95/p25 ≈ 14×, but in absolute terms the top decile
   is only ~4% likely to produce an MSMI. Sorting cannot manufacture outcomes.
2. **The cheap, policy-legal signal captures most of it.** Scoring by *observed
   soft-field agreement alone* gives 1.38×; adding oracle knowledge of
   `response_rate` adds only 1.48×. Learning per-member latents is not worth the
   risk, because the data rate is 1–2 observations per member.
3. **Ordering has limited value precisely because edges are scarce.** If ~89% of
   the reachable edges are already consumed, reordering redistributes value
   without dropping any, so the sum is nearly invariant. Value ordering mostly
   helps by *sequencing*: put high-value edges in the earliest rounds, which
   matters for `drift` (see §6) and for `delayed`.

**Total achievable MSMI ≈ Σ over reachable edges of P(MSMI | e), i.e. roughly
0.5–1.5 events per episode in `development`, against a greedy baseline of ~1.0.
The realistic headroom on the primary score is therefore on the order of
1.3–1.6×, concentrated in `sparse` and `cold_start` where volume headroom is
largest, not in `development` where greedy is already near-saturated.**

---

## 6. Per-family structure worth exploiting

- **`sparse` (1/6 of the score, everyone scores ~0.04–0.08).** `acceptable_zones`
  is forced to `[zone]`, so pairs must share a zone; with 12 uniform zones a
  feasible pair needs both endpoints in the same ~17-member bucket *and* both
  cleared. Clarification should therefore be **zone-clustered**: clear whole
  zones rather than the whole pool, so cleared members concentrate inside the
  dense sub-blocks. This is where the largest relative gain lives, because greedy
  is furthest from `|E|` here.
- **`cold_start`.** Only 20% of members are `complete` (vs 35%), so the
  clarification ramp is the bottleneck; `|E| = 90.5` vs 77.6 introductions
  realised. Buy constraints as early as possible — the 4-bundles/day cap binds
  for the first ~25 days.
- **`shift`.** The same four fields carry very different weights
  (`goal 0.25, pace 0.8, lifestyle −0.25, conversations 0.5`). The greedy
  baseline scores pairs by an *unweighted count* of agreeing soft fields, which is
  close to worst-case here. **Online weight learning is the single largest
  defensible win on this family**, and it requires no knowledge of the variant.
- **`drift`.** A −0.5 logit shift from day 35 roughly halves every pair's
  probability. Since edges do not expire, the optimal response is to **front-load
  high-value edges before day 35** and to stop making low-value introductions
  afterwards. This is a scheduling policy, not a model change, and it is directly
  detectable from a drop in observed acceptance rate.
- **`delayed`.** The 5–12 day extra date delay is *independent of assignment day*,
  so it lowers `P(date within 30 days)` uniformly rather than penalising late
  assignments. Its effect is a flat ~few-percent loss, not a timing effect. This
  corrects an intuitive-but-wrong "introduce early under delay" hypothesis.

---

## 7. Proposed policy

Three separable modules, per PS §15 (keep normalisation, modelling and allocation
separate).

**(a) Acquisition.** Two queues, hard-first:
- Constraint bundles (3 units) for available members with any `HARD` field
  `null`/`not_asked` and no `declined` hard field. This is the only thing that
  exposes the feasible graph, and the baseline already does it correctly.
- Remaining budget (1 unit each) on the four decision-relevant soft fields
  `relationship_goal, relationship_pace, lifestyle, conversations`, prioritised by
  arrival day. These cost ~311 units for all reachable members — the whole
  acquisition plan fits inside 720 units with room to spare.

This is the ablation the PS requires: *disabling soft-field acquisition* is the
"no value signal" arm.

**(b) Value model.** For each field `k` with observed values on both sides,
contribution `+w_k` if equal else `−w_k/2`. For an unobserved or declined field,
marginalise under the uniform prior over its `|S_k|` options:

```
E[contribution_k] = w_k · (1.5/|S_k| − 0.5)
```

This is exactly right for the kit (missing fields are a uniform random subset, so
conditionally uniform) and it removes the silent `None`-equals-`None` artefact that
naïve code produces. Then `w = logistic(−0.25 + Σ_k E[contribution_k])` with the
intercept calibrated from observed response rates.

**(c) Weights.** Online logistic regression on directional revealed preference:
each matured `introduction_response` with a non-null value contributes a sample
with features (field agreement on the *observed subset*, label = recorded Yes).
Partial observation is handled by restriction to the observed subset, which is
unbiased for a correctly specified main-effects model. Regularise toward the
`development` prior, clip the step size, and reset/re-widen on detected
distribution shift.

**Critically: report the two directions separately**, as PS §3 requires, and
integrate over the shared `shared` shock for the joint — do not multiply marginals.

**(d) Allocation.** Maximum-weight matching on the eligible graph with weights
`P(MSMI | pair)`. The graph is **general, not bipartite** — ~47% of women accept
meeting women, so same-gender pairs are common and a bipartite solver is invalid.
Greedy plus single-pair-replacement local search is adequate here and cheap; the
non-adaptive-greedy result of Udwani (arXiv:2403.18059) gives an honest
1/2-competitive statement if an exact Blossom solve is too slow for the 10-second
budget.

---

## 8. Position in the literature

**Learn2Match (arXiv:2606.06744).** Formallyises two-sided matching with
temporally extended feedback as a partially observable Markov game with costly
pre-match screening, noisy post-match observations, evolving latent profiles and
endogenous dissolution — the same objects as this challenge. Its central result is
the one most relevant here: **independent PPO achieves higher cumulative social
welfare and lower cumulative regret than the bandit-style CA-ETC baseline, yet
PPO's information-friction loss converges to a strictly non-zero value** because
end-to-end MARL does not recover the coordinated exploration structure of matching
bandits. Read against our measurements, that is exactly the predicted failure:
welfare (≈ realised introductions) is nearly saturated while the *friction* term
(≈ how much of the reachable edge set we actually expose and order) is where the
remaining gap lives. We should therefore **not** reach for RL.

**Matching bandits are the right frame, but the vocabulary needs care.** The
canonical regret literature for learning stable matchings under unknown
preferences is the Liu–Mania–Jordan lineage (AISTATS 2020; JMLR 2021 CA-UCB;
Basu et al. ICML 2021 Phased-ETC; Kong & Li SODA 2023), summarised by Li, Wang &
Kong, *A Survey on Bandit Learning in Matching Markets*, IJCAI 2025. It assumes
**one** side's preferences are unknown and the other side's are given and fixed.
Two-sided reciprocal uncertainty is only now being opened up — Pokharel & Das
(arXiv:2302.06176), Pagare & Ghosh's two-sided CA-ETC (arXiv:2408.08690),
Basu's super-stability route (arXiv:2506.15926), and Athanasopoulos, George &
Dimitrakakis (UAI 2026, PMLR v337:228–254), whose regret bounds avoid dependence
on `Δ_min` — directly relevant when all pair probabilities are small and clustered,
as they are here (p25–p95 = 0.0012–0.0172).

**The clarifying-budget question has a clean theoretical answer.** Saar-Tsechansky,
Melville & Provost (Management Science 2009) frame it as *Active Feature-Value
Acquisition* with the Sampled Expected Utility rule — expected improvement in
predictive utility **per unit acquisition cost** — which is precisely the
`3 units for a bundle vs 1 unit for a field` structure. Ma et al. (NeurIPS 2023)
give the theoretically cleanest version: with cross-entropy loss the expected
one-step improvement from adding feature `i` equals exactly `I(y; x_i | x_S)`, so
the greedy `argmax_i I(y; x_i | x_S)/c_i` rule is exact rather than heuristic.
DIME's Proposition 1 (per-prediction budget constraints are Pareto-dominated by
average constraints) is the licence for spending the 12 units freely across the
population. Target the mutual information to the *pair outcome*, not to the fields —
Bickford Smith et al. (AISTATS 2023) make exactly this critique of BALD via EPIG.

**Why we must not do inverse-propensity evaluation.** `propensity` is `null`
throughout and PS §9 forbids claiming unbiased IPS from these files. Comparisons
must come from controlled simulator episodes on identical seeds. Separately,
Lancewicki et al. (COLT 2021) show that when outcome-correlated reporting latency
exists, estimates are **biased, not merely noisy** — here `declined` versus
`not_asked`, and `no_response` versus a recorded No, are exactly that mechanism.
Survival-analysis framing (Guinet, Amin & Jaillet, NeurIPS 2022, "effective
dimension under censorship") is the principled way to keep them separate.

**Query-efficient stable matching is impossible, which is why we aim at
approximate stability.** Segal-Halevi (2011) proves `Ω(n²)` queries are required
even to *verify* stability, for deterministic and randomised algorithms alike. But
Drummond & Boutilier (IJCAI 2013) show query counts can be cut to `log₂ n` per
person if one accepts **maximum/minimax regret** instead of exact stability — the
right target here, since the outcome is stochastic. Iwama et al. (ICALP 1999)
bound the complexity cliff: incomplete lists alone and ties alone are polynomial,
both together are NP-complete.

**Allocation under uncertainty.** Non-adaptive greedy is 1/2-competitive against an
adaptive offline benchmark and this is tight (Udwani, arXiv:2403.18059, to appear
in *Operations Research*), proved by a lifting argument. Blum et al. (*Operations
Research* 68(2), 2020) is the closest structural analogue: an adaptive algorithm
using `O(1)` edge queries per vertex reaches a `(1−ε)` fraction of the omniscient
optimum, while a single non-adaptive round reaches `(0.5−ε)`. **That 1/2 gap is
the quantitative justification for spending the budget adaptively rather than
clearing everything up front** — though our measurements suggest the realised gap
here is small because the edge supply, not the query schedule, binds.

---

## 9. Experimental plan

- **Declared training seeds**, disjoint from validation. Pool-level splits only —
  never row-level, per PS §4.
- **Baselines**: all three, identical seeds and variants.
- **Ablations** (PS §10 requires at least one; we will run five):
  1. no soft-field acquisition (value model on ~1 observed field),
  2. fixed `development` weights vs online weights (isolates `shift`),
  3. greedy over observed-soft-count vs maximum-weight matching (isolates allocation),
  4. hard bundles only vs hard + soft (isolates the feasibility/value split),
  5. no drift-aware scheduling (isolates the day-35 response).
- **Seed count.** MSMI is ~1 event per episode; the across-seed SD is ≈ 0.34 MSMI,
  so with 20 seeds per family the per-family SE is ≈ 0.076 and the primary-score
  SE ≈ 0.031. **A difference below ~0.1 in the primary score is not resolvable at
  this sample size**, and the report must say so rather than ranking on noise.
  The PS's own warning applies: a zero MSMI count is not evidence of equivalence.
- **Report separately**: assignments, mutual acceptances, dates, MSMI, coverage,
  clarification cost, missing feedback, first-introduction waiting times and the
  unserved count.
- **Report the funnel decomposition**, not just MSMI — it is the only way to show
  whether a gain came from volume or from ordering.

---

## 10. Anticipated failure cases

- **Withheld answers.** ~1/3 of members have a declined hard field and are
  permanently unmatchable. They will appear in the coverage denominator and in the
  unserved count forever. The report must explain them rather than drop them.
- **Sparse supply.** In `sparse`, `|E| ≈ 21`. Even perfect scheduling yields ~21
  introductions, so the family contributes ~0.05–0.10 for everyone; do not
  over-read differences there.
- **Competition for the same candidate.** Because `advance` allows at most one
  introduction per person at a time and members re-enter on a shared 8-day
  cadence, the schedule is coupled across the whole pool. Greedy day's choices
  constrain days +8, +16, … This coupling is why a proper matching (rather than
  per-day greedy) should help, and it is worth an explicit experiment.
- **Delayed feedback + right-censoring.** An introduction made on day 55 resolves
  around day 77, inside the 40-day follow-up, so the *evaluator* sees it — but the
  *policy* never does. Learning must not credit an unresolved introduction as a
  failure. Track outstanding introductions explicitly.
- **Weight-learning instability.** The first online run produced unstable weights
  (e.g. `relationship_pace = −0.69` in `cold_start`) because 100+ SGD steps at
  `lr = 0.35` with weak L2 overshoots on a 4-dimensional problem with ~1% base
  rates. Needs step decay, prior anchoring, and clipping. Interestingly the same
  run recovered `relationship_pace = 1.19, lifestyle ≈ 0.00` on `shift`, which is
  the correct direction — the signal is there, the optimiser is not.
- **Model miscalibration.** The closed-form model slightly over-predicts on
  `shift`. Since probabilities do not affect ranking, this is tolerable — but any
  *threshold* rule (e.g. "skip introductions below a value bar") would be exposed
  to this miscalibration and should be avoided in favour of ranking-only decisions.

---

## 11. Honest assessment of the achievable gain

Measured against the supplied baselines over 12 seeds × 6 families, the candidate
policy's primary score is **0.417 versus greedy's 0.403** — inside the noise band
(SE ≈ 0.03). The *intermediate* metrics move in the right direction and much more
clearly: mutual acceptances rise from 12.6 to 13.9 in `development` (+10%) and
from 11.5 to 13.5 in `shift` (+17%), at equal volume and equal coverage.

That is the honest state of play. The structural analysis says the headroom is
real but small and concentrated in two places — `sparse` and `cold_start`, where
greedy is furthest from the reachable-edge ceiling — and it says the achievable
multiplier is roughly 1.3–1.6×, not 5×. The most valuable contribution of the
final report is therefore **the ceiling analysis itself**: it explains why the
metric behaves as it does, which member of the team built the strongest claim in
Round 1, and it makes the failure analysis interpretable.

Two caveats I could not resolve before stopping, which should be checked first:

1. **An apparent inconsistency in the `sparse` ceiling.** `analysis/ceiling.py`
   reports `|E| = 20.9` for `sparse`, but the greedy baseline realises ~22.6
   introductions per episode there — more edges than the ceiling allows. One of
   the two is wrong. `analysis/consistency.py` was written to localise it (it
   checks whether assigned pairs are a subset of the reachable truth-feasible set)
   but was interrupted. Until it runs, **treat the `sparse` ceiling as unverified**;
   the other five families are internally consistent.
2. **The closed-form `P(MSMI)` under-predicts the aggregate by roughly 2×**
   (Σ over 103 edges ≈ 0.56 predicted, versus ~1.0–1.2 observed), while being
   correct on single-pair replay. The discrepancy is unexplained — possibly an
   interaction between repeated world seeds and member latents, or an artefact of
   the replay harness. It does not affect rankings (which is what matters for
   ranking) but it does mean the absolute ceiling in §5 is understated.

---

## Appendix — reproducing this

```bash
cd /root/hackathon/The-Sequential-Matching-Problem
python -m unittest -v && python verify_data.py

cd /root/hackathon/analysis
python3 prob_model.py      # recovered generative law + oracle scoring
python3 validate_model.py  # model vs simulator, per family
python3 why_cap.py         # feasibility cascade, daily density, clarification supply
python3 ceiling.py         # reachable edge set, |E|, max matching
python3 value_spread.py    # P(MSMI) spread, realised lift
python3 candidate.py       # policy v1 vs all three baselines, 6 families
python3 consistency.py     # (unrun) assigned-vs-reachable subset check
```

Environment note: the kit is standard-library only and `numpy`/`scipy`/`networkx`
are unavailable, so all analysis and the eventual policy are pure Python. That is
also a constraint for assessed inference (2 cores, 1 GiB, 10 s per invocation).