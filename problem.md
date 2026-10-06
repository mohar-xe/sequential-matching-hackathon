# The problem, in plain language

What we are actually being asked to build. No math, minimal jargon. Companion to
[`README.md`](README.md) (why it's hard, with measurements) and
[`RESEARCH_NOTE.md`](RESEARCH_NOTE.md) (technical derivations + citations).

---

## The job in one line

**Write a program that, twice a day for 60 days, decides (a) which questions to ask
people and (b) which pairs to introduce — to maximise how many people end up with a
mutual "yes, let's meet again".**

It is a **policy**, not a model. A program that gets called, returns an answer, and
is told nothing except what it chose to ask about.

---

## What you start with

200 synthetic people in one pool. They trickle in over the first three weeks.

For each person you are **given** three things: age, gender, zone.

That is almost all you get. Each person has **18 questionnaire fields**, and at the
start of the game roughly **60% of them are blank** for each person.

A blank field has **three** possible states, and the distinction matters enormously:

| Status | Meaning |
|---|---|
| `observed` | asked, and they told you |
| `not_asked` | nobody has asked them yet |
| `declined` | you asked, and they **refused to say** |

🔑 **A `declined` field can never be revealed again.** It is permanent. No budget and
no cleverness gets around it. (This turns out to be the single most consequential
rule in the whole challenge.)

The 18 fields split into two groups:

- **11 "hard" fields** — acceptable age range, genders they'll meet, relationship
  structure, smoking (own and partner's), children (own, partner's, and whether they
  want them), acceptable zones, free-time slots.
  These decide **whether a pair is allowed to exist**.
- **7 "soft" fields** — relationship goal, preferred pace, lifestyle, conversation
  style, emotional availability, space for a relationship, relocation willingness.
  These are the **only** thing affecting **whether an allowed pair succeeds**.

---

## What you do, each day, twice

### First call — "what do you want to ask?"

Return a list of `{person, question}`.

| Question | Reveals | Cost |
|---|---|---|
| `"constraints"` | **all 11 hard fields** for that person | **3 units** |
| one named soft field | that one field | **1 unit** |

**Budget: 12 units per day.** So either 4 people fully, or 12 people one question
each. You may only ask people who are currently available, and asking more than the
budget is an error.

### Second call — "whom do you want to introduce?"

Return a list of pairs. **Every** pair must pass **all** of these, or the entire
batch is rejected and you get an error:

- both people currently available, distinct, in the same pool
- within each other's acceptable **age** range — *both* directions
- each is a gender the other said they'd meet — *both* directions
- each accepts the other's **zone** — *both* directions
- no smoking conflict; no children conflict; not "one wants kids / one doesn't"
- same relationship structure (monogamous vs not)
- at least one **overlapping free-time slot**
- **never introduced before** in this episode
- **nobody used twice in one batch**; nobody has an introduction already outstanding
- 🔑 **every hard field known for both people**

That last one is the crux: **you must pay 3 units per person before you can even
propose them to anyone.** You cannot propose-and-hope.

Returning an empty list is legal — that is how you "wait".

---

## What happens after you decide

Both people are asked. Each independently either replies or stays silent. Each
independently says yes or no. If both say yes, a date may happen. Then each is asked
whether they want to meet again, and the answer arrives 1–5 days later.

🔑 **You do not see any of this on the day it happens.** Feedback appears only once its
observation day arrives. After day 60 you are frozen: 40 more days pass where you do
nothing, but feedback still arrives.

---

## How you are scored

An introduction counts as a **win** only if *all* of this happened:

1. both people said yes to the introduction
2. a date actually happened
3. **within 30 days** of you assigning it
4. both people said "yes, meet again"
5. **both within 3 days** of the date

Then: **wins ÷ 200 × 100**.

Final score = average across **6 hidden worlds** (normal, sparse geography, cold
start, delayed dates, shifted outcome rules, drifting response conditions) × 20
hidden seeds each. You never see those worlds.

**Tiebreakers, in order:** coverage (how many distinct people got at least one
introduction), then mutual acceptances, then *lower* clarification cost, then lower
inference time.

---

## What is deliberately withheld

You never receive the world seed, who has not arrived yet, anyone's real hidden
preferences, or anyone's likelihood of replying.

You are explicitly forbidden from reading the simulator's internals, reverse-
engineering hidden answers from public seeds, or inspecting organiser files.

The only things crossing the boundary are: what you observed, the answers you paid
for, matured feedback, **and your own memory — which you carry yourself**, because
the process is killed and restarted on every single call.

**Engineering limits:** JSON in / JSON out, ≤ 1 MiB per message, **10 seconds per
call including startup**, 2 CPU cores, 1 GiB RAM, no network, no GPU.

---

# Are the soft fields already decided?

Yes — and this is the single most important thing to understand about the problem.
It has three separate readings; all three were tested (`analysis/soft_fields.py`).

## (a) Are the values fixed? **Yes, completely.**

The soft field values are drawn once when the world is generated and **never change
during the episode**. Verified: across 60 simulated days, **0 of 200** members had
any soft field value change.

So they are **static parameters**, not a moving target. There is no "people reveal
more as they get to know each other". Nobody's mind changes.

> The `drift` world does *not* mean preferences drift. It applies a −0.5 shift to the
> scoring formula from day 35 onward. The underlying preferences are still static;
> the scoring of them moves.

**Consequence:** asking is a pure information purchase. There is no decay, no
staleness, no re-ask. What you learn in week 1 is still true in week 9.

## (b) Are they already visible? **No.**

At any moment, the chance a given soft field is *observed* for one person is about
**0.41** in the normal world and **0.28** in the cold-start world.

But agreement is a property of a **pair**, so a field only enters the score when
**both** people have been observed. That roughly squares it:

| | one person observed | **both** observed | **pair has all 4** |
|---|---|---|---|
| normal | 0.41 | 0.15–0.17 | **11.4%** |
| cold start | 0.28 | 0.08 | **3.8%** |

The 11.4% is mostly pairs where *both* people happen to be fully pre-filled — so it
is **not spread evenly**; it clusters on a minority of the population.

**If you ask for the four fields**, each becomes observed with probability 0.93 (7%
decline), giving 0.865 per field jointly, and:

| | before asking | after asking |
|---|---|---|
| normal | 11.4% of pairs have complete fit | **56%** |
| cold start | 3.8% | **56%** |

So asking is worth roughly **5× more coverage** in the normal world and **15×** in the
cold-start world — and it flattens the two worlds to the same value, which is
convenient for a policy that cannot see which world it is in.

## (c) Do all 7 soft fields matter? **No — only 4. Three are dead weight.**

Only these four enter the outcome:

| Field | weight, normal | weight, shifted world |
|---|---|---|
| `relationship_goal` | 0.70 | 0.25 |
| `relationship_pace` | 0.40 | **0.80** |
| `lifestyle` | 0.25 | **−0.25** |
| `conversations` | 0.20 | 0.50 |

These three **cannot** affect the outcome in any world:

- `emotional_availability`
- `space_for_relationship`
- `relocate`

Verified two ways: none of them appear in the simulator's scoring function, and
force-changing each one to a different value left the measured MSMI **bit-identical**
(0.0098 → 0.0098, delta +0.0000).

🔑 **Never ask about these three.** Each costs 1 unit and buys literally nothing. In
a 720-unit total budget that is pure waste.

Note also the shifted world: `lifestyle` gets a **negative** weight, so a lifestyle
*mismatch* is mildly **good**. A policy that assumes "match on everything" is wrong
there, and `pace` matters twice as much as it otherwise would. This is why the
weights should be **learned from revealed preferences**, not hard-coded.

---

## Why this reframes the whole challenge

Put the three answers together:

1. Soft fields are **static** → asking is a one-time, lossless purchase. No
   staleness problem, no re-ask, no exploration/decoration tradeoff.
2. Soft fields are **mostly unobserved** and you need **both** sides of a pair → the
   baseline sees complete fit for only 11% of pairs, or 4% in cold start.
3. Only **4 of 7** soft fields matter → asking about the other 3 is pure waste.

And the supplied `greedy` baseline **spends 100% of its clarification budget on
`constraints` bundles and essentially never buys soft fields.** It is therefore
ranking pairs using, in most cases, less than one observed field. That is the
opening.

The whole design reduces to:

> **Spend 3 units to make a person *eligible*. Spend the leftover 1-unit budget to
> make them *understandable*. Never spend on the 3 dead fields. Rank by fit
> marginalised over what is still unknown.**

---

## One caution, so the ceiling stays honest

Because soft fields are static and asking is lossless, it is tempting to assume
buying all of them solves the problem. It does not. Measured lift from sorting pairs
by fit is only about **1.4×**, because:

- per-person hidden traits (responsiveness, personal bias) are seen at most **1–2
  times** per episode, so they cannot be learned — knowing them would only lift the
  ratio to 1.48×;
- the legal-pair set is tiny (~103 pairs total), so there is little room to be
  selective;
- ~98.5% of every introduction dies for reasons outside the policy's control,
  including a 3-day answer window that discards 64% of otherwise-perfect outcomes.

The `greedy` baseline is already at **81–98% of the achievable ceiling** (and
`sparse` is essentially saturated at 98%). This is a **1.3–1.6× problem, not a 5×
one**. See [`README.md`](README.md) §2 for the full accounting.

We also tested the obvious "just optimise the 4 weights globally" idea — see
[`research/RESEARCH_NOTE.md`](research/RESEARCH_NOTE.md) §5.1. It fails for a
structural reason, not a compute one: the entire range a global optimiser could
exploit (0.132 in primary score) is *smaller than the noise on a single run*
(0.356), so its selection step would be optimising noise. Fitting those 4
parameters by maximum likelihood on revealed responses is the right tool.