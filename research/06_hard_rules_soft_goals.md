# 06 · Hard Rules, Soft Goals — Course Notes Mapped to Our Policy

Distilled from the second taught deck ("Hard rules, soft goals": physics → ecosystems →
constrained learning). Its central distinction is **the** architectural principle of this
hackathon: hard rules define what is allowed (enforced exactly); soft goals express
preferences (weighted, traded off); and multipliers like λ are **computed by the solver,
never chosen**. The problem statement states it directly (PS §5): hard constraints
"cannot be overridden by a favourable score"; soft observations "inform estimates without
overriding constraints."

---

## 1 · The bead-and-magnet is our eligibility-vs-score split

| Physics deck | Our challenge |
|---|---|
| Circular wire: hard rule — every reachable position lies on it | `eligibility()` hard checks: who_to_meet, age bounds, zones, smoking/children, structure, schedule |
| Magnet below the glass: soft pull — finite, never warps the wire | Learned pair score `P̂(MSMI)`: finite, never creates or unmakes a pair |
| λ (wire's push-back): **computed, not chosen** | Not directly used, but the same discipline applies: what the *data* decides (posteriors, fit weights) vs what *we* choose (weights we own, e.g. shortfall/priority weights in the matcher) |
| Reduced coordinates: 3 positions → 1 angle; every update through q(θ) stays legal | **Parameterise search in legal coordinates**: never score all 19,900 pairs then filter — iterate only over edges that are feasible-or-clarifiable |

Two operational consequences we already follow, now with their theory name:

1. **Reduced coordinates = feasible-edge enumeration.** Scoring only pairs whose
   `eligibility()` is `feasible` or `needs_clarification` (≈1.24% of pairs in
   `development`, per RESEARCH_NOTE.md §3) is exactly "optimisation in reduced
   coordinates" — every downstream update (scoring, matching, learning) stays inside the
   hard equalities by construction, and the 10-second budget is spent only where it can matter.
2. **A high score cannot bend the wire.** Our scorer feeds *selection among* feasible
   pairs only. There is no code path where P̂(MSMI) relaxes a hard check — keep it that
   way and say so explicitly in the report (the PS's "a high predicted outcome cannot
   compensate for a constraint violation").

## 2 · The ecosystem quiz answers are our data-modelling answers

The pond quiz (slides 20–21) is a stealth test of modelling discipline, and option B's
win is directly about our feedback accounting:

> **Correct: B — competition comes through the shared food.** Every meal is counted once
> as a loss and once as a gain. Adding a *direct* competition term (A) double-counts what
> the shared resource already represents; imposing `S+I ≤ R` as a hard rule (C) turns a
> growth law into a constraint; a penalty (D) turns dynamics into optimisation.

**Our analogues:**

- **Count each event once, in both directions.** A directional yes is one person's
  acceptance; the *mutual* acceptance is derived from both — never double-count the same
  feedback in two posteriors, and never count a member-level event as also pair-level
  evidence in the same update. (Our §3.2 pending-queue maturity rules enforce this.)
- **Don't add terms for what the shared latent already carries.** The pair-level
  `shared ~ N(0, 0.45²)` shock already induces the directional correlation (ρ ≈ 0.43);
  modelling "member a tends to accept when member b accepts" as a *separate* feature
  would be option A's double-count.
- **Don't turn dynamics into a loss.** Coverage pressure, pacing, and the 8-day busy
  window are *state transitions*, not penalty terms. If we want fewer low-value
  introductions, that is a sequencing/allocation decision (our θ-as-ordering), not a
  `(S−I)²`-style penalty bolted onto the scorer.

## 3 · The interaction mask is literally our hard-eligibility structure

Slide 27's consumer×food mask B (1 = link allowed, 0 = never) is structurally identical
to hard eligibility: **a zero in the mask means no direct link, not no effect** (slide
26's quiz answer: herons never eat invertebrates but still change their number — through
the chain herons → fewer fish/frogs → less predation → more invertebrates).

Two mappings for us:

- **Hard zero ≠ no effect.** A pair blocked by a hard failure still affects the system:
  the two members occupy availability slots, compete for other partners, and their
  blocked edge constrains the matching. Reporting "unmatched ≠ harmed" and explaining
  unserved members (REPORT_GUIDE #7) is the same discipline as the heron answer.
- **Third-species effects = market-level coupling.** The deck's handling-time/rectuge
  higher-order effects (slides 41–42) have an exact analogue: introducing A–B today
  changes B's availability for C tomorrow, and the matcher's choices couple across the
  whole pool. This is why allocation is a *global* per-day optimisation (RESEARCH_NOTE
  §7d), not a pile of independent pair decisions — and why greedy-fill leaves value on
  the table in sparse/cold_start even though it saturates development.

## 4 · Hard budgets in a feeding round = our daily decision budget

Slide 30's classification quiz, transcribed to our objects:

| Deck statement | Our analogue | Hard or soft |
|---|---|---|
| "No feeding amount can be negative" | No pair reuse; no person in two pairs; distinct available adults | **Hard** — enforced by `advance`, batch-atomically |
| "Frogs cannot eat snails" (mask zero) | `eligibility()` failure or missing hard field | **Hard** — `infeasible` forever, or `needs_clarification` until resolved |
| "Fish and frogs cannot eat more than 8 available" (stock) | 12-unit/day ask budget; one concurrent intro per person | **Hard** — budget checks raise on violation |
| "A frog cannot eat more than its capacity" | 7-day response deadline; 60-day horizon; 30-day MSMI window | **Hard** — structural, not scorable |
| "Fish would *like* 6 units" (target D_F) | "This pair would be a great match" (posterior mean) | **Soft** — a wish; used for ranking only |
| "We would *prefer* pressure on snails low" (weight w_F) | Exploration vs exploitation weighting; κ prior strength; sequencing θ | **Soft** — we choose, we ablate |

**The slide-33 lesson transfers exactly:** when the budget binds (both targets exceed
supply), raising one side's weight *shrinks its shortfall but never creates a surplus*
— and the price λ rises. Our version: when the feasible edge set is scarce (sparse:
|E| ≈ 21), cranking exploration pressure cannot manufacture feasible edges; it only
re-prices which edges get the scarce slots. Hence: explore cheaply, allocate globally,
and never expect preference weighting to substitute for supply.

## 5 · Θ, ψ, or neither — the parameter-hygiene quiz for our own system

Slide 36's sorting discipline, applied to our design (useful for the report's "training,
inference assets, randomness" section, REPORT_GUIDE #3):

| Family | Our items |
|---|---|
| **Θ — fixed by the world, estimated from data** | The four soft-field fit weights; per-member latents (`bias`, `second_bias`, `response_rate`) — the deck's rule: "change these and the pond behaves differently." We *estimate* these; we never hand-tune them per variant (that would be leaking variant labels) |
| **ψ — chosen by us, declared and ablated** | κ (pool-prior strength ≈ 4); λ decay ≈ 0.97/day; exploration κ in UCB-style sequencing; θ as ordering parameter; clarification VOI threshold τ. Each must appear in the report with its default, its ablation, and why its default is defensible |
| **Neither — computed, not set** | Posterior means/variances; the day's matching; the shadow price of the ask budget (our λ-analogue); the drift-detection trigger output. "λ is not a parameter: the solver computes it" — anything downstream of our ψ and Θ must never be hand-patched mid-episode |

This is also a reproducibility table (deck slide 38): state / dynamics / hard action set /
soft objective / execution / learning — the same six rows our report needs for
"reproducible evaluation command" and declared seeds.

## 6 · Forward vs inverse fitting — and what we may claim

Slide 39: forward prediction (Θ → ŷ) vs inverse fitting (y → Θ̂), with the warning that
*different parameters can produce the same observations* — unidentifiability.

**Our exposure:** the four soft-field weights are identifiable in aggregate (the `shift`
recovery of `pace=1.19, lifestyle≈0` proves the signal exists), but per-member latents
are not (1–2 observations each; RESEARCH_NOTE §0.5). The deck's warning is the theory
behind our empirical rule: report population-level weight estimates with calibration
checks; report member-level posteriors only as *priors we act through*, never as claimed
knowledge of a person. "A learned score with model weights ω... eligibility and capacity
remain hard while scores improve" (slide 40) is, verbatim, our architecture.

## 7 · What this deck changes in our plan

1. **Add the Θ/ψ/computed table to the report** (from §5 above) — it satisfies the
   REPORT_GUIDE's modelling-description and reproducibility items with the deck's own
   vocabulary, which organisers will recognise.
2. **Feasibility-first workflow** (deck slide 16): write hard rules → check feasibility →
   encode/reduce → choose soft objective → solve → verify residuals. Our pipeline
   already runs in this order (eligibility gate → feature extraction → scoring →
   matching); make the *residual verification* explicit: a unit test asserting that no
   proposed pair ever violates `eligibility()` under any scorer output (the deck's
   "numerical feasibility requires a residual check").
3. **No double-counting audit** (from the pond quiz): one test asserting each feedback
   event enters at most one posterior update — directional yes enters member response
   posteriors; the derived mutual acceptance is never *also* fed back as an independent
   sample.

## 8 · Additions to the prior-art map (from this deck)

| Concept | Anchor | Use in our report |
|---|---|---|
| Constrained optimisation via multipliers | Karush–Kuhn–Tucker framing (deck slides 6, 14) | The hard/soft split vocabulary; λ-as-price intuition for budget-binding under scarcity |
| Constraint enforcement taxonomy (slide 14) | Elimination/KKT; LP-QP; SQP; **MILP or matching algorithm** for integer decisions | Cite for "integer decisions → matching algorithm": our per-day allocation *is* the deck's matching-algorithm row |
| Interaction masks in ecology | Deck slides 25–28 (mask B; zero = no direct link, not no effect) | Vocabulary for explaining why unmatched/unserved members still matter (coverage denominator honesty) |
| Identifiability under inverse fitting | Deck slide 39 | The formal reason member-level latents stay priors, not claims |
| Higher-order interactions → tensors | Deck slides 41–42 | Justifies pool-level (market-coupled) allocation over independent pair scoring |
