# Research Notes — Sequential Matching Hackathon

Working research folder for the Vouchsafe Sequential Matching Hackathon (release 1.0.0).
Documents what the problem asks and what solution approaches are viable, with emphasis
on **online learning** methods.

## Contents

| File | What's inside |
|---|---|
| [RESEARCH_NOTE.md](RESEARCH_NOTE.md) | **Prior empirical deep-dive (round-1 draft)** with measured results from `analysis/` scripts: recovered generative law, feasibility cascade, funnel decomposition, ceiling analysis, candidate-policy trial, literature positioning, honest gain assessment (+ two open verification items). |
| [01_problem_breakdown.md](01_problem_breakdown.md) | What the hackathon actually asks: frozen mechanics, scoring formula, interface protocol, hard constraints, feedback semantics, baseline reference points, and what each scenario family stresses. |
| [02_solution_space.md](02_solution_space.md) | Solution families ranked by fit (learned scorer + max-weight matching; per-member bandit; VOI clarification; deferred acceptance; full RL — rejected; LLM — rejected) and the prior-art anchors to cite. Cross-checked against RESEARCH_NOTE.md's measurements. |
| [03_online_learning_design.md](03_online_learning_design.md) | The concrete online learning design: what to learn, leak-free update rules, Bayesian pair scorer, member-level Thompson bandit, VOI clarification policy, waiting threshold, and the 1 MiB memory budget math. Reconciled with measured evidence. |
| [04_architecture_and_plan.md](04_architecture_and_plan.md) | Module layout, pretraining asset shape, phased execution plan (day-by-day), required ablations, honest-evaluation traps to avoid, risks + mitigations, and a report outline. Adjusted to the measured headroom. |
| [05_explore_exploit_notes.md](05_explore_exploit_notes.md) | The taught course material (GA/DE → surrogates/BO → bandits → VOI & drift) distilled and mapped onto our policy: the greedy lock-in numbers that justify Thompson sampling, decision-relevant VOI for the ask budget, drift remedies (discounting, restart-exploration triggers, coverage-based wrong-belief detection), and which course methods belong to cheaper-evaluation columns (cited, not used). + additions to the prior-art map. |
| [06_hard_rules_soft_goals.md](06_hard_rules_soft_goals.md) | The second deck (physics wire/magnet → pond ecosystem → constrained learning) distilled: the bead-and-magnet as the eligibility-vs-score split, the pond quiz's no-double-counting discipline for feedback updates, the interaction mask as hard-eligibility structure, hard-budget scarcity lessons (weights re-price, never create supply), the Θ/ψ/computed parameter-hygiene table for the report, and identifiability as the formal reason member latents stay priors. + additions to the prior-art map. |

## One-paragraph recommendation

**Primary architecture:** an online **logistic weight learner over the four soft fields**
(`relationship_goal, relationship_pace, lifestyle, conversations`) — the only pair-value
signal in the generator, and the lever that handles `shift` without knowing the variant —
with hard-bundle clarification first (supply-limited, so clear everything askable early,
zone-clustered in `sparse`), leftover budget on soft fields, marginalisation over
unobserved fields, **general-graph** matching allocation (greedy + local search; the graph
is not bipartite), and **drift-aware scheduling** (front-load value before day 35).
Per-member Beta posteriors are kept only as cheap pool-prior shrinkage — the measured data
rate (1–2 observations/member) cannot support them as a headline claim.

**Measured honest headroom (RESEARCH_NOTE.md §11):** candidate policy 0.417 vs greedy
0.403 primary score (within noise), with clearer intermediate gains (+10%/+17% mutual
acceptances). Total achievable multiplier ≈ 1.3–1.6×, concentrated in `sparse` and
`cold_start`. The ceiling analysis is itself the strongest deliverable.

**Non-goals (documented, not implemented):** full RL (Learn2Match's friction-loss result
predicts its failure here), external LLM scoring, offline IPS on the propensity-free
shipped snapshots, and black-box weight search (measured: noise dominates the effect).

## How this maps to the submission checklist

Everything in this folder is pre-writing. The actual submission needs, per
`docs/SUBMISSION.md`: runnable policy behind `policy.py` JSON protocol, committed
`Dockerfile`, pinned deps, an ablation, baseline comparisons on identical seeds and all
six variants, a technical report covering what `examples/REPORT_GUIDE.md` items 1–8 list,
and machine-readable results JSON.
