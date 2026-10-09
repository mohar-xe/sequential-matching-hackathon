# 01 · Problem Breakdown

Source: `PROBLEM_STATEMENT.md` v1.0.0, `docs/DATA_CONTRACT.md`, `docs/POLICY_INTERFACE.md`, `kit.py`.

## The ask, in one sentence
Build a **daily decision policy** that, from observable-only state, (a) picks clarification
questions under a budget, (b) proposes a batch of feasible, non-overlapping, reciprocal
introductions, and (c) updates belief from delayed, selective, right-censored feedback —
scored primarily by **MSMI per 100 arrived members** across 6 scenario families × 20 seeds.

## Concrete mechanics (frozen in release 1.0.0)

| Element | Value / rule |
|---|---|
| Decision horizon | 60 days, then 40 follow-up days (no new actions) |
| Clarification budget | 12 units/day; `constraints` bundle = 3 units, named soft field = 1 |
| Clarification latency | Immediate (v1.0); declined questions stay unknown forever |
| Concurrent intros | Max 1 per person (person is busy ~8 days after assignment) |
| Repeat pair | Forbidden within an episode |
| Response deadline | 7 days |
| Primary outcome (MSMI) | First date within **30 days** of assignment AND both return **Yes** within **3 days** of that date |
| Primary metric | mean over 6 scenario families of (MSMI_count / N_arrived × 100) |
| Tiebreakers | coverage ↓ mutual acceptances ↓ ask cost ↓ inference time |
| Compute limit | 2 CPU, 1 GiB RAM, 10 s/invocation, 1 MiB JSON, offline CPU-only |
| Data | 10 pools × 200 synthetic members; 6 train / 2 val / 2 dev-test |

## The policy loop (per day)
```python
asks  = policy.ask(state)               # budget ≤ 12
state = sim.observe()                    # refreshed with ask results
pairs = policy.match(state)              # batch of disjoint feasible 2-tuples
state = sim.advance(pairs)               # observes only past feedback
```
Process is **restarted between phases**; anything to remember must be re-serialized in
`memory` (JSON, ≤1 MiB, no filesystem, no NaN/Inf). Every invocation must return in <10 s.

## Hard checks that gate every pair (`eligibility`)
Reciprocal checks both directions:
- `who_to_meet` contains the other's gender; other's age inside `[age_min, age_max]`
- other's zone ∈ one's `acceptable_zones`
- `partner_smoking`/`smoking`, `partner_children`/`has_children`
- matching `relationship_structure`; no `wants_children` {yes,no} clash
- schedule overlap
- unknown hard field ⇒ `needs_clarification` (blocks, not fatal)
- known contradiction ⇒ `infeasible` (blocks forever)

**A missing hard value is never silent failure — it is a budget question waiting to be asked.**

> The hard/soft split has its own theory vocabulary — see
> `06_hard_rules_soft_goals.md`: hard rules are enforced exactly (multipliers computed,
> never chosen); soft goals are weighted preferences that can never bend a hard rule
> (the PS: "a high predicted outcome cannot compensate for a constraint violation").

## Feedback semantics (crucial for online learning)
- `introduction_response`: yes/no/null. **null ≠ rejection** (right-censoring).
- `date_happened`: only after two yesses; probabilistic (p≈0.78 by default; delayed variant shifts it).
- `second_meeting_intention`: only after a date; answering is itself probabilistic per member.
- Outcome maturity: an assigned intro is **unresolved**, not failed, until ~day-43 horizon.
- Selective labels: the policy only observes outcomes for the pairs *it chose*; historical
  logging propensities are **null** ⇒ no unbiased IPS from start-of-episode data alone.

## Score decomposition — where value comes from
Let MS=prob(date happens) ≈ 0.78, and per-person yes-prob ≈ 0.55–0.98 roughly 0.7 avg.
```
E[MSMI per introduction] ≈ P(both respond yes | we picked a pair)
                        × P(date within 30d)
                        × P(both reply yes to second meeting within 3d)
```
Rough baseline leak-through is ~`0.8 × 0.8 × 0.8 × 0.7⁴ × base-rate(z)` ≈ 0.15–0.25 per
introduction if we pick indiscriminately. **The policy's only real leverage on MSMI is
(a) picking pairs with high underlying mutual interest, and (b) achieving enough useful
introductions** (coverage is only a tiebreaker, but more attempts = more expected MSMI count
as long as per-intro quality is preserved).

> **Measured correction (RESEARCH_NOTE.md §4, funnel decomposition over 12 seeds):**
> introductions 91.6 → both-responses 52.0 (57%) → mutual acceptance 12.6 (24%) → date
> 10.3 (82%) → both second-yes 3.7 (36%) → within the 3-day window 1.3 (36%) → **MSMI ≈
> 1.0**. The 3-day window alone (`U{1..5}` ⇒ (3/5)²) destroys ~64% of fully successful
> outcomes and admits no policy action; ~98.5% of every introduction is lost to
> uncontrollable mechanisms. Only the mutual-acceptance stage (via `fit` and ordering) is
> meaningfully controllable. The realistic primary-score headroom is **1.3–1.6×**, not the
> multi-x an optimism-first read suggests.

Two structural facts worth memorising (both measured, RESEARCH_NOTE.md §2–§3):
1. The two directional acceptances are **correlated** (ρ ≈ 0.43 through the pair-level
   `shared` shock), so multiplying marginals understates mutual acceptance — integrate
   over the shared shock instead.
2. `fit` depends on the four soft fields only; hard constraints gate feasibility but
   contribute nothing to value. Only ~1.24% (development) / 0.24% (sparse) of all pairs
   are feasible, and ~1/3 of members have a `declined` hard field and can never be
   matched at all.

## Baseline reference points (public seed 101, shipped `examples/baseline_results`)
| Baseline | Primary | Coverage | Mutual/100 | Ask cost |
|---|---:|---:|---:|---:|
| greedy (soft-match count) | 0.5 | 0.41 | 7.6 | 205 |
| no_asks | 0.33 | 0.17 | 2.8 | 0 |
| random feasible | 0.25 | 0.41 | 6.7 | 205 |

`no_asks` confirms clarification buys coverage AND quality — asking pays for itself if it
stops hard-blocked pairs from sitting idle. `random ≈ greedy` on mutual acceptances shows
the greedy soft-score is barely better than chance on this generator; its soft feature
weights (goal/pace/lifestyle/convs) are all uniform-categorical so the "true" score is a
weightedHamming distance under `_prob()` — learnable.

## Scenario families and what they stress
| Variant | Twist | Design implication |
|---|---|---|
| development | baseline | everything |
| sparse | 12 zones instead of 4 | geographic feasibility rare ⇒ allocation quality matters more than score quality |
| cold_start | only ~20 % complete profiles at start | clarification value ↑ heavily |
| delayed | dates postponed 5–12 d | 30-day window gets tight ⇒ earlier/recall-oriented matching |
| shift | soft-field weights change (goal weight ↓, pace ↑, lifestyle negative) | needs adaptive feature weights, i.e. online learning |
| drift | after day 35, interest drops −0.5 logit | early aggressive matching, then adjust expectations |
