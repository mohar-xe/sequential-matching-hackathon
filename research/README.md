# The Sequential Matching Problem — from first principles

Our research write-up for the Vouchsafe Sequential Matching Hackathon.

**Read this first if you read nothing else.** The one-paragraph version:

> Imagine you run a matchmaking service for 200 people. You have to decide every
> day which pairs to introduce, and you may only spend 12 questions a day finding
> out what people actually want. It turns out almost all of the difficulty is not
> in *who should meet whom* — it is that **only 1.24% of pairs are even legal**,
> **a third of people can never be matched at all**, and **98.5% of every
> introduction you make fails for reasons you cannot control**. We measured all of
> this, and the practical consequence is that this problem has a hard ceiling
> that the supplied baseline is already within 11% of. The real work is
> understanding *why*, and squeezing the last 11%.

Everything below is derived from reading `kit.py` and from measurement against the
supplied simulator. All people, conversations and outcomes are synthetic.

| Document | What it is |
|---|---|
| **This file** | The explanation: layman → first principles → hypothesis → solution |
| [`RESEARCH_NOTE.md`](RESEARCH_NOTE.md) | The technical Round 1 submission draft, with full derivations and citations |
| [`analysis/`](analysis/) | Every script that produced the numbers here |

---

# Part 1 — What are we actually being asked to do?

## 1.1 The story version

You operate an introduction service. 200 synthetic adults arrive over the first
three weeks. Over 60 days you decide, each day:

1. **Who do you want to ask a question of?** You have a budget of 12 "clarification
   units" per day. Asking someone their entire bundle of hard constraints costs 3.
   Asking one specific soft question (their relationship goal, their preferred
   pace, …) costs 1.
2. **Which pairs do you introduce?** Each person can be in at most one
   introduction at a time. A pair that has already been introduced cannot be
   repeated. And the pair *must* satisfy every hard constraint in both directions
   — if you don't know whether a constraint is satisfied, you can't propose it.

After 60 decision days, the simulator runs 40 more days during which you do
nothing but feedback keeps arriving.

**The score:** the fraction of people who experienced a *Mutual Second-Meeting
Intention* — both sides said yes to the introduction, a date actually happened,
and both sides said they wanted to meet again, with both answers landing inside a
3-day window.

## 1.2 Why this is not just "rank all the pairs"

The problem statement gives a clean illustration. Four people, four feasible pairs:

| Pair | Value |
|---|---|
| A–B | 0.90 |
| C–D | 0.05 |
| A–D | 0.65 |
| B–C | 0.65 |

Greedy on the best pair first gives A–B + C–D = **0.95**.
The better choice is A–D + B–C = **1.30**.

This is the whole point: **every person can only be used once**, so the pairs
compete. This is a *matching* problem, not a *ranking* problem. Ranking tells you
which single pair is nicest; matching tells you which *set* of pairs is best in
total, and the answer is often not the one you'd reach for greedily.

Everything hard in this challenge is a consequence of that one structural fact.

---

# Part 2 — Six things that make this genuinely hard

I'll take these one at a time, in plain language first, then with the numbers.

## 2.1 Almost every pair is illegal

This is the single biggest surprise, and it reframes the whole problem.

> **Measurement.** I took 200 people, computed every one of the 19,900 possible
> pairs, and checked each against every hard-constraint rule using the *hidden
> ground truth* — i.e. the best possible information. Then I attributed the first
> rule each pair failed:

| First rule that rejects the pair | development | sparse |
|---|---|---|
| Won't meet their gender | **57.07%** | 54.14% |
| Outside each other's acceptable age range | **26.06%** | 25.60% |
| Won't accept the other's zone | 12.04% | 19.37% |
| Smoking conflict | 1.63% | 0.31% |
| Children conflict | 0.93% | 0.16% |
| Children *plans* conflict | 0.44% | 0.08% |
| Relationship structure mismatch | 0.38% | 0.07% |
| No overlapping free time | 0.20% | 0.03% |
| **✅ FEASIBLE** | **1.24%** | **0.24%** |

**Only about 1 pair in 80 is legal.** In the sparse-geography world, 1 in 400.

Why is gender so brutal? Because each person lists the genders they'd accept, and
70% of the time that list has exactly **one** entry, chosen uniformly at random
from {woman, man, non_binary}. So any given person accepts any given gender with
probability 0.47. For a pair to pass, *both* people have to accept *both* — about
0.22.

Age is nearly as bad: each person accepts only ages within `± a random 3–10` of
their own, so a 26-year-old and a 40-year-old are usually incompatible.

**First-principles takeaway:** the challenge is not mostly *"who is a good match"*.
It is mostly *"find the 247 legal pairs that exist in the first place, out of
19,900"*. Optimising desirability while ignoring legality is optimising a term
that is 98.76% zero.

## 2.2 A third of the people can never be matched at all

Some people answered "I'd rather not say" to at least one hard question. Those
answers are **permanently unavailable** — I checked `kit.py`, and `resolve_asks`
will never fill in a declined field. Since an introduction requires *every* hard
field known for *both* people, a person with one declined hard field is
**permanently unmatchable**, no matter how clever the policy is.

| world | matchable people out of 200 |
|---|---|
| development / delayed / shift / drift | 133.4 (67%) |
| sparse | 131.2 (66%) |
| cold_start | 120.0 (60%) |

These people still sit in the score's denominator. They will always be counted as
unserved. **Your policy can never rescue them**, and a report that quietly drops
them from its statistics is lying by omission.

## 2.3 You can't propose a pair you haven't fully checked

This is where the clarification budget bites. To propose pair (A,B) you need all
11 hard fields known for **both** A and B. So you don't get to "try and see" — you
must pay 3 units per person *before* you know whether that person is worth
introducing to anyone.

And the budget is not really the constraint. Look at how the supplied greedy
baseline spends it:

```
bundles asked per day:
[4,4,4,4,4,4,4,4,4,4,4,4,1,0,0,2,0,1,0,0,2,0,0,0, 0,0,0, ... ]
```

It asks the maximum 4 bundles/day for 12 days, then **starves**. Not because it
ran out of budget (there are 720 units available over 60 days) but because it ran
out of *people worth asking* — the pool of members without a declined hard field
is only about 80 people.

> **First-principles takeaway:** there is no ask-budget headroom to exploit. The
> budget is 720 units; the useful work is about 480. Any claim of "we spend our
> clarification budget better" must be checked against *this* fact, not against
> the budget ceiling.

## 2.4 Almost nobody answers, and the outcome definition is brutal

Now the part that makes the score look so small. Here is the full funnel for the
greedy baseline in the `development` world:

| stage | count | conditional rate | can a policy change it? |
|---|---|---|---|
| introductions made | 91.6 | — | **yes** — but capped (§2.6) |
| both people replied | 52.0 | 57% | no — personal responsiveness, 1 sample each |
| both said yes (mutual acceptance) | 12.6 | 24% | **yes** — this is the lever |
| a date actually happened | 10.3 | 82% | no — fixed at 78% |
| both said "yes, meet again" | 3.7 | 36% | marginally — personal trait, unlearnable |
| both said it within **3 days** | 1.3 | **36%** | **no** — the reply delay is random 1–5 days |
| **MSMI (the score)** | **≈ 1.0** | | |

Two things to internalise:

- **The 3-day window throws away 64% of otherwise perfect outcomes.** If both
  people genuinely wanted to meet again, but one replied on day 4 instead of day 3,
  it doesn't count. The reply delay is a uniform random 1–5 days, so the chance
  both land inside 3 days is exactly `(3/5)² = 36%`. **No policy can touch this.**
- **The whole funnel multiplies out to ~1.5% MSMI per introduction.** So the score
  is tiny (0.2–1.0 range) and, critically, **one episode produces roughly *one*
  success event**. That makes the measurement extremely noisy — see §2.7.

> **First-principles takeaway:** about 98.5% of every introduction is lost to
> mechanisms outside the policy's control. A policy is a very thin lever on this
> machine.

## 2.5 What you learn is contaminated by what you chose

You only find out how a pair went if you chose to introduce them. So the data you
collect is not a random sample — it's biased toward whatever your policy already
thought was good. This is called **selection bias**, and it's the classic trap of
learning from your own logs.

Two concrete traps in *this* kit:

- The `propensity` field (which would let you correct for this) is `null`
  everywhere, and the problem statement explicitly forbids claiming unbiased
  inverse-propensity evaluation from these files. So we can't just reweight.
- **The `declined` status vs `not_asked` status is a bias generator.** Someone who
  refuses to answer is systematically different from someone who wasn't asked. If
  you pool them into "missing", your estimates are *biased*, not just noisy. Same
  for `no_response` versus a recorded No: a silent non-answer is not evidence of
  dislike.

Also: **the per-person traits you'd most like to learn are the ones you can't.**
Each person's willingness-to-respond and personal "openness" bias is revealed only
through their own introductions — and each person appears in at most ~1.5
introductions per episode. One Bernoulli observation tells you almost nothing: the
posterior is no tighter than the prior. So the "learn a model of each person"
intuition, which is the most natural thing to try, is a dead end here.

## 2.6 The score has a hard ceiling, and the baseline is already near it

Because repeat pairs are forbidden, **every legal pair can be used at most once**.
So the total number of introductions is bounded by the number of legal pairs that
exist among *reachable* people:

| world | reachable legal pairs (edges) | largest single-day matching | supplied greedy achieves |
|---|---|---|---|
| development | **103.1** | 34.5 | ~92 |
| cold_start | 90.5 | 31.8 | ~78 |
| sparse | **20.9** | 14.8 | ~23 (⚠ see §6.1) |

Greedy is at **~89% of the theoretical ceiling** in development. There is not a
5× win hiding here. There is a **~1.13× volume win**, and then the quality lever
on top of it.

And here's the subtle sting: because greedy already uses *almost every* legal
edge, **re-ordering them by desirability barely helps**. If you use all 103 edges
in some order, the total score is the same — the sum doesn't care about the order.
Ordering only pays if you use *fewer but better*, and the bottom 11% of edges are
worth almost nothing (probability ~0.0003 each), so dropping them saves nothing
either.

> **First-principles takeaway:** this problem is structurally *saturated* in the
> baseline. Any honest estimate of achievable improvement has to be modest, and
> the ceiling analysis is worth more than a marginal score.

## 2.7 The measurement noise is the same size as the effect

From §2.4, one episode yields ~1 MSMI event, with an across-seed standard deviation
of ~0.34. So:

- with 12 seeds per world, the standard error on a family's mean is ~0.10;
- averaged over 6 worlds, the standard error on the headline score is ~0.04.

**Any difference below roughly 0.1 in the headline score is not real.** Our
prototype scored 0.417 against the baseline's 0.403 — inside the noise. We could
easily have "improved" the score by 20% by changing random seeds, and that would
have been a lie.

This is why the problem statement warns that "a zero MSMI count can occur in a
small synthetic episode; it is not evidence that two methods are equivalent."

---

# Part 3 — So what *is* the lever?

Everything from §2.6 says: volume is nearly capped, and per-edge probability is
fixed once a pair is chosen. So the only real levers are:

**(a) Convert more of the ~103 legal pairs into actual introductions.** Worth up
to ~13%. Concentrated where greedy is furthest from the ceiling: `cold_start`
(clarification ramp is slow because only 20% of people arrive pre-filled) and
`sparse`.

**(b) Raise the 24% mutual-acceptance rate by choosing better pairs.** This is the
quality lever. It depends on one thing and one thing only:

> **The outcome depends on exactly four "soft" fields — the relationship goal, the
> preferred pace, the lifestyle, and the conversation style. Nothing else. The hard
> constraints decide whether a pair is *legal*, but they have *zero* influence on
> whether the pair succeeds.**

That's a clean separation, and the greedy baseline misses it: **it spends 100% of
its clarification budget on hard constraint bundles and essentially never buys the
four soft fields that actually determine success.** It then has to guess, ranking
pairs by how many of the 7 soft fields happen to be observed and agreeing — and
only ~1 is usually observed.

Measured value of sorting by the four fields:

| world | sorting on observed fields gives | adding hidden knowledge of responsiveness gives |
|---|---|---|
| development | 1.38× | 1.48× |
| sparse | 1.44× | 1.56× |
| cold_start | 1.19× | 1.22× |

Note the second column. Knowing each person's hidden responsiveness would help
only a little more — and we established in §2.5 that we *can't* learn it anyway.
**The cheap, legal, observable signal gets most of the benefit.**

**(c) Adapt to which world you're in, without being told.** The six test worlds
differ in ways that matter:

| world | what changes | the right response |
|---|---|---|
| `sparse` | 12 zones, no second zone → legal pairs must share a zone | **clarify whole zones, not the whole pool** |
| `cold_start` | only 20% arrive pre-filled | buy constraints ASAP; the ramp is the bottleneck |
| `shift` | field weights become `goal 0.25, pace 0.8, lifestyle −0.25, conversations 0.5` | **learn the weights online** — greedy assumes all fields are equal, which is near-worst-case here |
| `drift` | every pair's odds halve from day 35 onward | **front-load good pairs before day 35** |
| `delayed` | dates slip 5–12 days | flat loss; *not* a timing problem (see below) |

A policy that hard-codes the `development` field weights will be badly miscalibrated
on `shift`. A policy that treats all soft fields equally will be badly miscalibrated
on `shift` too. **This is the cleanest, most defensible research contribution
available: learn the weights from revealed preferences and let the policy adapt
without ever being told which world it's in.**

One correction to an intuitive-but-wrong guess: under `delayed`, the extra date
slip is *independent of when you introduce*, so it does **not** reward introducing
early. It just uniformly lowers the chance the date lands inside the 30-day
window. Verified by reading the generator, not assumed.

---

# Part 4 — Our hypothesis

> **Primary hypothesis.** The dominant term in the score is *realising more of the
> reachable legal-pair set*, and the dominant controllable term in the outcome is
> *soft-field agreement*. A policy that (i) maximises legal-edge realisation via
> fast, well-targeted constraint clarification, and (ii) buys the four decision-relevant
> soft fields and ranks by fit marginalised over what remains unknown, will beat the
> supplied baseline by roughly **1.3–1.6×**, with essentially all of the gain coming
> from the `sparse` and `cold_start` worlds.

> **Secondary hypothesis.** Learning the four fit weights online from revealed
> directional responses recovers most of the `shift` family's advantage, without
> variant detection, and is worth more than any improvement to the allocation step.

> **Falsifiable prediction.** If we disable soft-field acquisition (ablation 1),
> the score should collapse back to roughly baseline. If we fix the weights instead
> of learning them, the `shift` world should lose most of its gain while the other
> five are unaffected. If the ceiling analysis is right, *no* policy — not even one
> given perfect knowledge of everyone's hidden preferences — should exceed about
> 0.8–1.5 MSMI per episode in development.

The third prediction is the important one: **it's a claim about the ceiling, and it
should be falsifiable.** We built a policy that is literally handed every hidden
preference and every hidden responsiveness, and it only reached 1.25 versus
baseline's 1.17. That is the strongest evidence we have that the problem is
saturated, and it is worth putting in the report.

---

# Part 5 — The proposed solution

Four separable modules. (The problem statement asks that normalisation, modelling
and allocation stay separate — this is that separation.)

### (a) Acquisition — what to ask, and in what order

Two queues, hard constraints first because they gate everything:

1. **Constraint bundles (3 units)** for anyone available with a missing hard field
   and no declined hard field. This is the only thing that reveals the legal-pair
   graph at all.
2. **Whatever budget is left (1 unit each)** on the four decision-relevant soft
   fields, prioritised by arrival day.

Cost check: clarifying constraints for all reachable people ≈ 259 units; the four
soft fields for all of them ≈ 311 units. **Total ≈ 570 against a 720 budget.** It
fits, with room to spare. This is the ablation arm required by the problem
statement: *disable step 2*.

### (b) Value model — scoring a pair without knowing everything

For each of the four fields, if both values are known: `+w` if they agree,
`−w/2` if they don't. **If a field is unknown or was declined, don't ignore it and
don't treat "both unknown" as "they agree" — average it out.** A missing field is
uniformly random over its options (that is how the kit generates missingness), so
its expected contribution is `w · (1.5/|S| − 0.5)`.

This small detail matters: naive code that skips missing fields systematically
*overrates* pairs where neither person answered, and *underrates* pairs where both
answered and disagree. Marginalising fixes both.

### (c) Weights — learning the fit from revealed preferences

Online logistic regression on directional responses. Every matured
`introduction_response` with a real value becomes one training sample: features
are the field agreements on the fields *both* people answered, label is "said
yes". Restricting to fully-observed field pairs is the standard unbiased way to
handle missing features in a main-effects model.

This is where the `shift` world is won: the model should discover on its own that
`pace` matters most and that a `lifestyle` *mismatch* is mildly *good*.

Two engineering warnings from our first run: the optimiser was unstable (step size
far too large for a 4-dimensional problem with ~1% base rates — it produced a
weight of `−0.69` on `pace`), so it needs step decay, anchoring to a prior, and
clipping. Encouragingly, the same unstable run *did* recover the right answer on
`shift` (`pace = 1.19`, `lifestyle ≈ 0.00`). The signal is there; the optimiser
was not.

### (d) Allocation — choosing the set of pairs

**Maximum-weight matching** on the legal-pair graph, weighted by predicted
`P(MSMI)`.

One important technical trap: **the graph is not bipartite.** About 47% of women
are willing to meet women, so same-gender pairs are common and plentiful. A
bipartite matching solver would be silently wrong here. We need a general-graph
matcher (Blossom), or greedy plus local repair.

We should also state our allocation guarantee honestly rather than overclaiming:
for this class of problem, plain greedy is 1/2-competitive against an adaptive
offline benchmark, and that bound is tight. Claiming anything stronger without a
proof would be exactly the kind of overreach this report should avoid.

### (e) Reporting directional probabilities separately

The problem statement asks us to distinguish "A accepts B", "B accepts A", and
"both accept", and warns that multiplying directional probabilities assumes
independence. **That warning is quantitatively justified here.** The two directions
share a single random shock, so their probabilities are correlated at roughly
ρ = 0.43. Multiplying the marginals *understates* mutual acceptance. The correct
calculation integrates over the shared shock.

---

# Part 6 — What we measured, and what we got wrong

## 6.1 Open inconsistencies (must be resolved before we publish numbers)

We are not going to paper over these.

1. **The `sparse` ceiling is self-inconsistent.** We computed 20.9 reachable legal
   edges, but the baseline realises ~22.6 introductions per episode there — more
   edges than should exist. One of the two numbers is wrong.
   `analysis/consistency.py` was written to localise it (it checks whether assigned
   pairs are a subset of the reachable set) but was never run. **Until it runs, do
   not publish a `sparse` ceiling.**
2. **Our aggregate model under-predicts MSMI by about 2×** (predicted ~0.56 total
   across the reachable edges; observed ~1.0–1.2), even though it is correct on
   single-pair replay against the simulator. Unexplained. It doesn't affect
   *rankings*, which is what actually drives the score, but it means the absolute
   ceiling in §2.6 is understated.

## 6.2 Current experimental standing

Over 12 seeds × 6 worlds:

| policy | primary score |
|---|---|
| supplied greedy | 0.403 |
| our prototype v1 (fixed weights) | 0.368 |
| our prototype v1 (online weights) | 0.417 |

**None of these are distinguishable.** The standard error is ~0.04.

Where the signal *is* visible is the intermediate metrics, which have far more
events behind them:

| metric | greedy | prototype | change |
|---|---|---|---|
| mutual acceptances, development | 12.6 | 13.9 | **+10%** |
| mutual acceptances, shift | 11.5 | 13.5 | **+17%** |

Same volume, same coverage, better pair selection — exactly what §2.6 predicts. The
MSMI metric is simply too small and too noisy to show it at 12 seeds. That is the
finding we should report: **the mechanism works, the headline metric is
underpowered.**

## 6.3 Reproducing

```bash
cd analysis
python3 prob_model.py      # the recovered generative law + oracle scoring
python3 validate_model.py  # our model vs the real simulator, per world
python3 why_cap.py         # why only 1.24% of pairs are legal; daily density
python3 ceiling.py         # reachable edge set, ceiling, max matching
python3 value_spread.py    # how much value-ordering is actually worth
python3 candidate.py       # prototype vs all three baselines, all six worlds
python3 consistency.py     # (never run) the sparse-ceiling check
```

The kit is standard-library only and `numpy`/`scipy`/`networkx` are not installed,
so everything here — and the eventual policy — is pure Python. That is not just
convenience: assessed inference runs on 2 CPU cores, 1 GiB RAM, 10 seconds per
invocation, so anything heavier is off the table regardless.

---

# Part 7 — Where this sits in the research literature

**The benchmark paper we're named after.** Zong, Liang, Zhou & Jaques, *Learn to
Match: Two-Sided Matching with Temporally Extended Feedback* (arXiv:2606.06744)
models two-sided matching with interviews, noisy post-match observations, evolving
latent profiles and dissolution as a partially observable Markov game — the same
objects as this challenge. Its result that matters to us: **PPO beat the
bandit-style CA-ETC baseline on social welfare and regret, yet PPO's
"information-friction loss" converged to a strictly non-zero value**, because
end-to-end RL does not recover the coordinated exploration structure that bandit
methods get for free.

Read against our measurements, that is precisely the predicted failure mode:
*welfare* (≈ introductions realised) is nearly saturated, while the *friction* term
(≈ how much of the legal-pair graph you expose and order well) is where the
remaining gap lives. **This is an argument against reaching for RL in Round 2.**

**The right theoretical frame: matching bandits.** A caveat first — the
"conservative/cascading arm matching bandit" titles sometimes cited in this space
could not be verified in any index; the real, verifiable regret literature for
learning stable matchings under unknown preferences is the Liu–Mania–Jordan
lineage (AISTATS 2020; JMLR 2021 CA-UCB; ICML 2021 Phased-ETC; SODA 2023),
surveyed in Li, Wang & Kong, IJCAI 2025. Critically, it assumes **one** side's
preferences are unknown and the other's are known and fixed. Genuinely *two-sided*
reciprocal uncertainty is only now being opened up (Pokharel & Das 2023;
Pagare & Ghosh 2024; Basu 2025; Athanasopoulos, George & Dimitrakakis, UAI 2026 —
whose regret bounds avoid depending on the minimum reward gap, which is exactly
what you need when all pair probabilities are small and clustered, as ours are).

**Why the clarifying budget is an acquisition problem.** Saar-Tsechansky, Melville
& Provost (Management Science 2009) frame exactly this as *Active Feature-Value
Acquisition* with the rule "expected gain per unit of cost" — which is literally
our 3-units-for-a-bundle vs 1-unit-for-a-field structure. Ma et al. (NeurIPS
2023) give the cleanest version: with cross-entropy loss, the expected improvement
from revealing field `i` equals exactly the conditional mutual information
`I(y; x_i | x_S)`, which makes the greedy ratio rule *exact* rather than heuristic.
Target the mutual information to the **pair outcome**, not to the fields —
Bickford Smith et al. (AISTATS 2023) make exactly that critique via EPIG.

**Why we must not do inverse-propensity evaluation.** `propensity` is `null`
throughout and the problem statement forbids claiming unbiased IPS from these
files. Separately, Lancewicki et al. (COLT 2021) prove that when reporting latency
correlates with the outcome, estimates are **biased, not merely noisy** — which is
exactly what the `declined` / `not_asked` distinction manufactures here.

**Why we should target approximate, not exact, stability.** Segal-Halevi (2011)
proves `Ω(n²)` queries are needed even to *verify* stability of a matching, for
deterministic and randomised algorithms alike. But Drummond & Boutilier (IJCAI
2013) show you can cut queries to `log₂ n` per person if you accept **maximum
regret** instead of exact stability — the right target for a stochastic outcome.
Bampis et al. (APPROX/RANDOM 2024) frame "ask one field vs resolve a bundle" as a
competitive query-minimisation problem, which is the right lens for our ablation.

**Allocation under uncertainty.** Non-adaptive greedy is 1/2-competitive against an
adaptive offline benchmark, and that is tight (Udwani, to appear in *Operations
Research*). Blum et al. (*Operations Research* 2020) is the closest structural
analogue: an adaptive algorithm using `O(1)` queries per vertex gets within `(1−ε)`
of the omniscient optimum, while one non-adaptive round gets `(0.5−ε)`. That gap is
the quantitative case for spending the budget *adaptively* — though our
measurements suggest it is small here, because the edge supply, not the query
schedule, is what binds.

---

# Part 8 — Failure cases we expect, and how we'll report them

- **Permanently unreachable people (§2.2).** ~⅓ of the population. They appear in
  the coverage denominator and the unserved count forever. We will report them
  explicitly with their reason, not quietly drop them.
- **Sparse supply.** In `sparse` there are only ~21 reachable legal pairs, so the
  world contributes ~0.05 for everyone. Differences there are meaningless and we
  will say so rather than ranking on them.
- **Competition across days.** Because a person is busy for 8 days after an
  introduction, today's choices constrain days +8, +16, +16… The schedule is
  coupled across the whole pool. This is the strongest *theoretical* argument for
  proper matching over per-day greedy, and it deserves its own experiment.
- **Right-censoring.** An introduction made on day 55 resolves around day 77 —
  inside the evaluator's 40-day follow-up, so the *scorer* sees it, but the
  *policy* never does. Our learning must not credit an unresolved introduction as
  a failure. We'll track outstanding introductions explicitly.
- **Learner instability (§5c).** Observed directly; needs damping.
- **Miscalibration.** Our closed-form model slightly over-predicts on `shift`. Since
  probabilities don't affect ranking this is tolerable — but it means we must avoid
  threshold rules like "skip anything below a value bar" and use ranking-only
  decisions, which a miscalibration can't break.

---

# Part 9 — The honest bottom line

**The most valuable thing we have is the explanation, not the score.** Anyone can
improve on 0.4 by a bit. Being able to say *"we measured the ceiling, here it is,
here is why the metric behaves the way it does, and here is proof that even an
oracle policy cannot go much beyond it"* is a genuinely strong position — and it
is also the honest one.

The concrete plan, in priority order:

1. **Resolve the two open inconsistencies (§6.1).** We do not publish a number we
   cannot defend.
2. **Raise the seed count** so the headline comparison is actually powered, or state
   plainly that it isn't. Right now a 0.04 improvement is indistinguishable from
   zero, and we should say that in the report rather than implying otherwise.
3. **Build the real policy** against the JSON interface, with the four modules from
   §5, so Round 2 has a working submission with baseline comparisons, ablations and
   machine-readable results.
4. **Lead with the ceiling analysis and the funnel decomposition** in the report.
   They are reproducible, they are the most defensible claims we have, and they
   make the failure analysis interpretable rather than hand-wavy.

Contests are usually won by whoever understands the problem best. This one looks
like it rewards exactly that.