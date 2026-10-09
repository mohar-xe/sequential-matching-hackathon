# 02 · Solution Space & Prior Art

Candidate solution families, ranked by fit-per-effort for a compressed timeline under the
.release's hard constraints (CPU-only, 10 s/invocation, 1 MiB memory blobs, 1 GiB RAM).
Every family below respects the hard/soft split formalised in `06_hard_rules_soft_goals.md`:
scorers are soft (ranking only) and can never relax `eligibility()` hard checks; the
integer allocation itself is the "matching algorithm" row of the deck's constraint-enforcement
taxonomy.

---

## Template summary table

| # | Family | Fit | Key risk |
|---|--------|-----|----------|
| A | Learned pair scorer + general-graph matching allocation | ★★★★★ | overfitting / covariate shift |
| B | Pool-level online learner (LinUCB / Thompson) | ★★★★☆ | cold start vs horizon |
| C | Active Feature Acquisition (VOI clarification) | ★★★★☆ | budget spent on useless asks |
| D | Deferred-Acceptance-style bipartite structure | ★★★☆☆ | is not really needed here, single-shot matching |
| E | Full RL (PPO / model-based) | ★★☆☆☆ | no daily granularity, overkill |
| F | LLM/ensemble entertainment score | ★☆☆☆☆ | nothing real to ground it |

## Family A · Learned pair scorer + graph allocation ✅

**Core idea.** Learn "what makes a pair produce an MSMI" offline on `train` pools (using
public rollout data generated *by our own baseline policy* so logging bias is ours, not the
shipped one), then each day: score every feasible edge, solve **maximum-weight matching**
on the feasible sub-graph (degree cap 1 per person), and emit that batch.

**Why this wins.** The simulator's `_prob()` is a *known-shape* logistic over
`sum_k w_k * (1[f_a[k]==f_b[k]] − 0.5)` plus member bias + shared shock + variant drift.
A linear model over one-hot pairwise feature-agreement bits is therefore almost perfectly
specified; even a mis-specified shrinkage model improves on uniform-count greedy (which is
what the shipped baseline does).

**Allocation.** Each person in a day-batch can appear once ⇒ degree constraint = 1 per
vertex ⇒ this is exactly **maximum-weight matching in a general graph** (NOT bipartite —
~47% of members accept same-gender pairs — so bipartite solvers are invalid) weighted by
P(both-accept). The shipped `baseline_match` uses plain greedy top-by-soft-score.

⚠️ **Empirical correction (see RESEARCH_NOTE.md §5.1):** the feasible edge set is scarce
(|E| ≈ 103 in development, ≈ 21 in sparse) and greedy already realises ~89% of the
development ceiling. Allocation-side gains are therefore concentrated in `sparse` and
`cold_start`, not `development`. Also: `networkx`/`numpy` are **unavailable** in the kit
environment — use pure-Python greedy + single-pair-replacement local search (Udwani,
arXiv:2403.18059 gives an honest 1/2-competitive statement for non-adaptive greedy).

**Evidence needed in the report**: (i) model calibration on held-out pool pairs, (ii)
pool-disjoint ablation replacing the matcher with plain greedy, (iii) sparse/cold_start
variant win vs greedy.

## Family B · Online learner ✅

The shot at *online* value-add. Two natural factorisations:

### B1. Per-vertex latent quality (structural)
Model the *member bias* `b_i` and the *pair agreement bit score* as two independent
parameters; treat each *observable* introduction as a Bernoulli trial for "did this person
produce a mutual-acceptance path". Update with **Thompson sampling** over Beta
posteriors, or a shared `beta_i`-with-features linear bandit (LinUCB). Learns responsivity
and *who accepts* faster than per-pair models can.
- cold_start variant: member-specific bias is unknown ⇒ TS is the textbook medicine.
- drift variant: a forgetting factor / sliding window of ~14 days keeps pace with −0.5 logit.

### B2. Pair-level contextual bandit
Treat each day as a bandit round whose arms are *the entire batch candidate set*, reward =
Σ MSMI outcomes. This quickly becomes a combinatorial bandit — the practical version is
"pair-scorer weights as linear bandit context", i.e. update Family A's scorer *online* from
each new outcome, using an over-damped reward (e.g. `r ∈ {0, 1}` for matured MSMI,
`r = None` for pending ⇒ skip that example in the Bayesian/gradient update, exactly the
maturity rule in Section 8).

### Which to use
⚠️ **Empirical correction (RESEARCH_NOTE.md §0.5, §11):** per-member latents get only
1–2 observations per member per episode, so Beta posteriors stay ≈ the prior — B1's
*measured* lift over pool priors is small (1.48× vs 1.38× top-half lift with oracle
`response_rate`). B2 (online logistic weight learning on the four soft fields) is the
*defensible* online win: it is what carries the `shift` family (recovering
`pace=1.19, lifestyle≈0` was observed in practice — the signal exists, the optimiser
needs step decay, prior anchoring and clipping). **Recommendation: deploy B2 as the
primary learner; keep B1 only as cheap pool-prior shrinkage, never as the headline claim.**

## Family C · Active Feature Acquisition / VOI clarification ✅

The clarification budget is literally a cost-purchasing-information model. Map to known
literature:

- **Value-of-Information (VOI)**, Saar-Tsechansky & Provost's *Active Feature-Value
  Acquisition* (Management Science 2009) — rank candidate features by expected model
  improvement per unit cost.
- **Pandora's Box problem** (Weitzman '79; recent re-derivations Aouad 2025) — optimal
  ordering of costly inspections via per-box *reservation values*; the threshold form
  justifies a simple score like `Δvalue(clone the state, ask, recompute top pair)`.

Concrete version that fits our 12-unit/day budget:
```
For each candidate clarification (member, field='constraints') cost 3:
    delta = Σ over its would-be-feasible partners of (p_new - p_old)
    VOI_per_unit = delta / 3
For each (member, soft_field) cost 1:   # soft fields DON'T block pairs, so only ask
    VOI_per_unit = small drift-corrected gain / 1
```
Because `constraints` **blocks all pairs for that member** while unknown, the hard
clarification is the high-leverage one: it converts `needs_clarification` infeasible-looking
edges into real candidates. Cold_start turns this from a small win into the primary win.

⚠️ **Empirical correction (RESEARCH_NOTE.md §3):** the ask budget is **supply-limited,
not budget-limited** — askable supply (~80 members) exhausts by ~day 20 against a total
of 720 units. There is no budget headroom to exploit; the only question is *which*
members to clear. In `sparse`, clarification should be **zone-clustered** (clear whole
zones so cleared members concentrate in dense sub-blocks). The real acquisition lever is
the leftover budget on the **four decision-relevant soft fields** (`goal, pace,
lifestyle, conversations`) — hard fields only gate feasibility, soft fields are the only
pair-value signal (§1 of RESEARCH_NOTE.md).

## Family D · Deferred-acceptance / two-sided matching theory 🟡

Relevant literature worth citing even though our engine is "policy proposes, people answer
asynchronously": Gale–Shapley / Roth to explain why **proposing a pair you *expect* both to
answer yes to, otherwise wait**, beats spraying many low-quality pairs (which burns the
8-day busy window). Practically this shows up as a **non-zero score threshold** rather
than an exact GS run: matching everybody no matter what maximises coverage but not MSMI.
A simple "min expected score > θ, tune θ as an ablation" will carry this point.

## Family E · Full RL 🔴
The state space is a 200-member set with fine-grained per-day information, dynamics last
60 days, and no private reward signal exists beyond simulator runs we can generate.
Sample complexity for an RL policy to beat a learned greedy + matcher under these constraints
is, empirically, not going to materialise in the available time. Worth mentioning in the
report as "tested horizon-3 lookahead, worse than allocation" at best.

## Family F · Pretrained LLM / external 3rd party 🔴
No external data permitted at inference; CPU-only; conversations are authored templates
explicitly flagged as not for benchmarking. Not viable and not competitive.

## Prior art pointers (for the technical report)

| Topic | Anchor work | What to borrow |
|---|---|---|
| OPE for matching markets | Hayashi, Goda, Saito — *Off-Policy Evaluation and Learning for Matching Markets*, RecSys 2025 (arXiv 2507.13608) | the argument that reciprocal-recommendation feedback is sparse & high-variance; DiPS/DPR use of intermediate labels (our analogous intermediate label: *mutual acceptance*, before the final *MSMI*) |
| Delayed-feedback bandits | *Contextual Bandits under Delayed Feedback*, Academia/TIST variants | pending-action re-handling; exactly our day-0…43 right-censor window |
| Active Feature-Value Acquisition | Saar-Tsechansky & Provost, Management Science 2009 | per-unit-cost acquisition ranking as the ask-policy backbone |
| Pandora's box / optimal search | Weitzman 1979; Aouad 2025 extension | explains why per-candidate *reservation values* (upper-confidence-like) may outperform naive sorting |
| Online b-matching | Kalyanasundaram–Pruhs b-matching model | our per-day batch with degree-1 vertices, adjacent theory |
| Exploration/exploitation foundations | Thompson (1933); Russo et al. (2018); Auer et al. (2002); Lai & Robbins (1985) | TS as the exploration mechanism; Ω(log T) regret framing — see `05_explore_exploit_notes.md` §8 for the full list from the course material (incl. Lakkaraju et al. 2017 selective labels; Joulani et al. 2013 delayed feedback; Garivier & Moulines 2011 drift; Howard 1966 VOI; Das & Kamenica 2005 two-sided dating bandits) |
| Feedback loops in recommenders | Deconvolving Feedback Loops (Sinha 2016); Netflix 2023 | justifies holding an *exploration floor* on never-tried pool members; a pure greedy never introduces top-bias-implied cold candidates |
| Survey of online matching | Huang (SIGECOM Exchanges 2022) | taxonomy for our related-work section |
| Maximum-weight matching impl. | NetworkX `max_weight_matching` Blossom, O(n³ish) | practical allocation engine, <10 s safe |
