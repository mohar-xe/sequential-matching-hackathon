# 03 · Online Learning Design

Every mechanism here is implementable in pure Python stdlib and fits the process-restart
+ `memory`-blob protocol. Sizes assume 200 members/episode.

The constraint of `memory ≤ 1 MiB` (JSON, no NaN) and *process restart between calls* means
any online learner must be representable as **small sufficient statistics** in memory, not
as a pickled model on disk.

---

## 3.1 What the online learner should learn

Three unknowns drive MSMI, in order of variance:

1. **Weighted-agreement compatibility of a pair** — weight vector `w` over soft/hard fields
   (`relationship_goal(.7), pace(.4), lifestyle(.25), convs(.2)` in the
   `development` generator; rotated to `goal(.25), pace(.8), lifestyle(-.25), convs(.5)`
   in `shift`). This is the *reward function* being learned.
2. **Per-member "bias" `b_i`** — overall acceptance tendency. Random normal σ=.7, i.e. a
   ~30× wider spread than differences between two random pairs' agreement scores. Huge
   signal, low parameter count (1 per member).
3. **Per-member second_bias, and response_rate** — whether they respond at all, and how
   warm they are to second meetings. Both are per-member, learnable from feedback history.

Sufficient statistics that fit in 1 MiB comfortably:

- For each of 4 soft-agreement features: Beta posterior α_k, β_k (plus a knock-out
  "lifestyle negative" case — use a *signed* additive feature vector instead, see 3.3).
- For each member: `n_yes`, `n_no`, `n_respond`, `n_asked` response-rate signer, and
  similarly `second_*` for post-date interest (2 Beta params each).
- For the shift/drift variants: a time-index at which a legacy parameter-set is archived,
  plus a decay factor λ on all Beta counters.

## 3.2 Leak-free learning rule (what to update on which day)

> Counting discipline (from the second deck, `06_hard_rules_soft_goals.md` §2): every
> meal is counted once as a loss and once as a gain — each feedback event enters at
> most one posterior update (a directional yes feeds member response posteriors; the
> derived mutual acceptance is never also fed back as an independent sample), and no
> separate feature re-models what the shared pair shock already induces (ρ ≈ 0.43).

```
Day t, after introducing pair (a, b) with all features observed at the time of ask:
    enqueue (pair, t) with target=None in the pending queue.
    memory tracks: assignment day, feature snapshot pair (a-side, b-side), and
    observed response cutoffs

Day t+δ (δ from observed feedback):
    when introduction_response(yes/no) observed from BOTH ends:
        update Beta counters for per-member response rate and mutual-acceptance posterior
    when date_happened observed:
        update date posterior (shared)
    when second_meeting_intention both observed (or implied by deadline):
        if date within 30d AND both yes within 3d: target = 1
        elif date >30d or either no/noshow: target = 0
        now run the real fit against the pair features snapshot at assignment day***
```
*** Critical: **use the features as observed on the assignment day**, not today's. The
simulator marks `field_observed_day`, so any field learned later was not knowable at the
decision. Correct leak-free incremental learning reads the snapshot stored in memory at
assignment time.

## 3.3 Bayesian pair-scorer (the concrete model)

Represent the pair-feature vector x(a,b) ∈ ℝ^d with d ≈ 12:
- one-hot agreement features for goal, pace, lifestyle, convs: `1[f_a==f_b]`
- *signed* versions for the variants where the generator flips sign (e.g. lifestyle −w)
   learning these as signed coefficients rather than raw counts is what makes the
   `shift` scenario learnable online rather than stuck.
- numeric difference between ages |a.age − b.age| relative to overlapping-age window
- zone-availability-in-both-directions indicator (sparse variants)
- schedule overlap size
- per-member priors `b̂_a, b̂_b` from the *shared* member-latent model (3.1).

Online model: **Bayesian linear regression with Normal-inverse-gamma prior**, updated via
rank-1 Cholesky (dimension d ≤ 12 so this is trivially cheap CPU work), OR even simpler:
Bernoulli Thompson-sampling with a logistic link via Laplace approximation (i.e. one
Newton step per update). Both fit in stdlib; neither needs numpy — though pinning `numpy`
under 2 GiB image budget is fine and simpler.

Predictive mean ≈ expected P(mutual second-meeting intention). Use this directly as the
edge weight for the matching problem. Predictive **variance** feeds a one-step
`+ c · σ(x)` bonus for the Thompson / UCB flavour (c ≈ 1.0 to start, a documented ablation).

## 3.4 Member-level bandit (B1)

For each member keep `(n_yes, n_no)` Beta posterior over "will accept an intro", updated
when their `introduction_response` becomes visible. Also track `(n_date_yes, n_date_no)`.
On day t, for any member never observed, **prior from pool-wide mean**, NOT from their
unobserved covariates (avoid inventing).

```
sample b̂_i ~ Beta(α+b̂_i^pool × κ, β+(1−b̂_i^pool) × κ), κ ≈ 4 (strength of pool prior)
```
This gives *structured* cold-start exploration: an unknown member is sampled from the
pool prior scaled by κ, so their sampled acceptance remains realistic. No random
dithering needed; Thompson noise is the exploration. An explicit `ε-greedy = 0.1`
ablation should also be reported since the problem statement asks for it.

> **Empirical reconciliation with RESEARCH_NOTE.md (§0.5, §5, §11):** per-member latents
> receive only 1–2 observations per member per episode, so these Beta posteriors barely
> move beyond the pool prior — measured ceiling is 1.48× vs 1.38× top-half lift for
> fit-only scoring. Keep B1 as cheap pool-prior shrinkage; **do not base the report's
> causal headline on it.** The defensible online win is B2 (weight learning over the four
> soft fields), which carries `shift` without knowing the variant.

## 3.5 Clarification policy (VOI, online too)

> See `05_explore_exploit_notes.md` §3 for the sharpened rule from the course material:
> count only **decision-relevant** friends — pairs whose answer could change the day's
> argmax allocation — since `VOI(q) = 0` whenever no possible answer changes the best
> action (Howard 1966). "Don't ask where you are most uncertain; ask where your
> uncertainty is decision-relevant."

Budget = 12/day. Per-member cost-3 `constraints` bundle up to 3 per day? No — up to
4 per day (4×3=12) but the baseline takes only `budget//3`. Smarter:

1. Rank `needs_clarification` candidates by the **shadow value of clarity**:
   - For each such member m, let `n_friends(m)` = number of members who, if m's unknown
     hard fields turn out benign, become newly feasible.
   - `score(m) = n_friends(m) × Ê[agreement with these friends] / 3`.
2. Buy up to 3–4 such clarifications per day (11–12 units) but leave 1 unit only if
   a *soft-field ask* would settle the last unknown on a pair we will surely match today
   (rare in this generator because soft fields don't block — so mostly don't bother).

> **Empirical reconciliation with RESEARCH_NOTE.md (§3):** the ask budget is
> **supply-limited, not budget-limited** — askable supply (~80 members) exhausts by
> ~day 20 against a total of 720 units over the episode. There is no budget headroom to
> exploit; the open question is only *which* members to clear. The real acquisition
> lever is the leftover budget on the **four decision-relevant soft fields**
> (`relationship_goal, relationship_pace, lifestyle, conversations`) — hard fields gate
> feasibility but contribute nothing to pair value; soft fields are the only value
> signal. Total soft-field acquisition for all reachable members costs ~311 units, well
> inside 720. In `sparse`, cluster clarification by zone so cleared members concentrate
> inside dense sub-blocks.
3. Cap total `constraints` buys per member to once — the simulator resolves the whole
   bundle in one go anyway.
4. Never ask about members whose `field_status` says `declined` for any hard field
   (the ask would be silently refused, wasting the units), and never ask unavailable
   members (raises).

Threshold `clarify_only_if_score > τ`: at day 0 in cold_start, basically everyone is
unclear ⇒ buy as many as fit the budget; by day 20, only buy if `n_friends ≥ 2`. This
time-adaptive behaviour is itself a required ablation ("disable clarification").

## 3.6 Waiting / withholding (`Match some, not all`)

> **Exploration principle (from the course material, `05_explore_exploit_notes.md` §1, §5):**
> the self-adjusting exploration mechanism is *posterior disagreement* — rank pairs by
> posterior mean, break near-ties by posterior variance, and let the UCB-style
> acquisition `μ + κσ` transition from exploring to exploiting **by itself** as evidence
> matures. This is the formal justification for the rank-only, no-hard-cutoff rule below.
> Pure greedy is measurably unsafe: 35% of greedy runs lock out the best option forever
> while looking well-calibrated on their own data (course Table 2).
>
> **Empirical reconciliation with RESEARCH_NOTE.md:** the closed-form P(MSMI) is
> *miscalibrated in level* (over-predicts on `shift`, under-predicts aggregates ~2×) even
> though its *ranking* is sound (top decile 0.041 predicted / 0.033 empirical vs bottom
> 0.0022 / 0.0000). Since probability estimates do not affect ranking (PS §10), a hard
> score threshold θ exposes us to that miscalibration for zero ranking benefit.
> **Recommendation: rank-only decisions; treat θ as a soft sequencing device** (order
> high-value edges into the earliest rounds), not a hard cutoff. This also makes the
> `drift` response a scheduling question — front-load value before day 35 — rather than a
> threshold question.

Matching everybody is tempting (maximises coverage) but is **anti-optimal for MSMI**
because each member is then busy ~8 days. In a 60-day window a person can serve at most
~4 introductions anyway; burning one on a low yield pair costs nothing in coverage-terms
but does not count toward MSMI. Use a tunable score floor θ — **as a soft sequencing
device first, hard cutoff only if calibration allows**:

```
if p̂(MSMI | pair) < θ: defer the pair to a later round (or never)
```
- Standard: θ ≈ 0.25 (gives ~all pairs day 0, refines later)
- sparse: lower θ harder, since pairs genuinely are rare there
- drift: **front-load high-value edges before day 35** (a −0.5 logit shift then halves
  every pair's probability — this is a *scheduling* response detected from the observed
  acceptance-rate drop, not a model change)
- delayed: the extra 5–12 day date delay is *independent of assignment day*, so it
  lowers P(date within 30d) uniformly — a flat few-percent loss, NOT a reason to rush
  introductions early. The intuitive "introduce early under delay" hypothesis is wrong.

Tuning θ per variant must NOT be hand-fit per variant from private information; instead
derive θ from the learned online model with a small penalty (the greedy optimum already
has a natural gap; θ is the *small-score gap* at which adding an extra tentative pair
is worth not-taking it). A numerical grid over 3 values and reporting mean across variants
is the honest approach.

## 3.7 Memory budget sanity check

Fast arithmetic on peak episode size (200 members, 60 days):
- per-member: 3 Beta counters × 2 floats = 6 numbers ⇒ 200×6 ≈ 1.2 KB
- per-pending-introduction ≤ 4 × (2 ids + 12 pair features + 8 tracking ints) ≈ 200 B
- pair-scorer posterior: d=12 ⇒ mean vector Σ⁻¹ upper-triangle ≈ 100 floats = 800 B
- returned add-information (log/ask history can be re-derived from state each call)
Total active memory ≈ few kilobytes. JSON of ~10 KB. **Comfortably inside 1 MiB.**

## 3.8 What /cannot/ be done online here

These are honest limitations that belong in the report:

- Offline pretraining happens on our own *self-generated* rollouts (a policy oversamples
  edges it thinks look good); the standard fix is a second, deliberately *diverse*
  logging policy for the offline corpus. We plan two logging corpora and will report both.
- No propensity is available for the *supplied day-30 snapshots* at all ⇒ we cannot
  honestly do IPS on them and will state this (no temptation to overclaim).
- The `shift` variant rotates weights at evaluation time in a way nothing in deployment
  actually reveals; our forgetting factor λ = 0.97/day is tuned blind, not from labels.
