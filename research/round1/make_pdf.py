"""Build R1_SUBMISSION.pdf from stdlib only (no TeX toolchain here).

Content mirrors R1_SUBMISSION.tex. All lines are pre-wrapped (<=80 chars)
so no font metrics are needed. Output: PDF 1.4, A4, Helvetica/Helvetica-Bold.
Usage: python3 make_pdf.py  (writes R1_SUBMISSION.pdf next to itself)
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE_W, PAGE_H = 595.27, 841.89
ML, MR, MT, MB = 62, 62, 58, 62
TW = PAGE_W - ML - MR
BODY, LEAD = 10.0, 13.2
MAX_LINES = int((PAGE_H - MT - MB) // LEAD)

# ---------------------------------------------------------------- content
TITLE = [
    ("t1", "Sequential Matching under Reciprocal Uncertainty:"),
    ("t1", "Ceilings, Acquisition, and Exact Matching"),
    ("t2", "Round 1 Research Submission -- Vouchsafe Sequential Matching Hackathon"),
    ("t3", "Team mohar-xe  |  Kit release 1.0.0  |  9 October 2026  |  commit b1e1b46"),
]

BLOCKS = []
B = BLOCKS.append


def para(*lines):
    B(("para", list(lines)))


def bul(*items):
    B(("bul", list(items)))


def sec(title):
    B(("sec", title))


def table(headers, rows, widths, aligns):
    B(("table", (headers, rows, widths, aligns)))


sec("Abstract")
para(
    "We study the Sequential Matching Hackathon: twice daily for 60 days",
    "(+40 follow-up), a policy spends a 12-unit ask budget and proposes",
    "disjoint feasible pairs to maximise mutual second-meeting intentions",
    "(MSMI) per 100 arrived members, averaged over six scenario families.",
    "Our contribution is measurement-first. Only 1.24% of pairs are",
    "hard-feasible (0.24% sparse); one third of members can never be matched;",
    "98.5% of every introduction is lost to mechanisms no policy controls,",
    "above all a 3-day answer window discarding ~64% of successes. The only",
    "reliable pair signal is agreement on 4 of 7 soft fields. The reachable",
    "edge set caps introductions and greedy already realises 89-99% of the",
    "corrected ceiling -- so this is an acquisition problem, not an RL",
    "problem (honest headroom ~1.3-1.6x). Our stdlib-only policy climbs a",
    "paired ladder 0.333 -> 0.514 (directional; headline underpowered ~20x).",
    "Fixed-prior beats adaptive everywhere, as noise theory predicts.",
)

sec("1  Problem interpretation")
para(
    "Each of 60 days: (i) clarification asks costing <= 12 units (hard bundle",
    "= 3, one named soft field = 1; declined is permanent), then (ii) a batch",
    "of disjoint feasible pairs (empty = wait). Pairs must pass reciprocal",
    "hard checks both ways (age, gender, zones, smoking, children, structure,",
    "schedule), be unintroduced, with all hard fields known. Feedback matures",
    "with delay; 40 follow-up days resolve outcomes. MSMI: both yes -> date",
    "within 30d -> both second-yes within 3d of the date. Score: mean over 6",
    "families x 20 seeds. Tiebreakers: coverage, mutual acceptances, lower",
    "ask cost, lower inference time. JSON <= 1 MiB, 10 s/call, 2 cores,",
    "offline; memory self-carried. Soft fields are static (0/200 changed);",
    "only goal/pace/lifestyle/conversations enter the scorer (0.70 / 0.40 /",
    "0.25 / 0.20; shifted: 0.25 / 0.80 / -0.25 / 0.50). The other three soft",
    "fields are dead weight: never ask about them.",
)

sec("2  Hypothesis")
para(
    "A reciprocal pair scorer learned online from delayed/censored feedback,",
    "with global max-weight allocation and VOI-ranked clarification, raises",
    "MSMI/100 over greedy via (i) learned weights vs uniform soft-count,",
    "(ii) global allocation vs greedy fill, (iii) soft acquisition vs none,",
    "(iv) drift-aware sequencing. Full RL is rejected (Learn2Match friction",
    "result: it fails exactly when volume is saturated and the gap is",
    "friction). LLM scoring is rejected (offline CPU; template conversations).",
)

sec("3  Reciprocal feasibility: introductions are capped")
para(
    "First-failing-rule on truth: who_to_meet 57%, age 26%, geography",
    "12% (19% sparse); rest ~4% combined. P(matchable) = 0.667 / 0.656 /",
    "0.600 -- one third can never be matched. |E| is a hard ceiling (no",
    "repeats). A sparse contradiction (|E| = 20.9 vs 22.6 realised) was a",
    "seed-set mismatch: same seeds give |E| = 22.8 vs 22.6 (99% saturated),",
    "assigned subset of reachable with 0 violations. Clarification is",
    "supply-limited: 720 units vs ~80 askable members, exhausted by ~day 20.",
)
table(
    ["Family", "Match/200", "|E|", "MaxMatch", "Greedy"],
    [
        ["development*", "133.4", "103.1", "34.5", "~92 (89%)"],
        ["sparse", "131.2", "20.9-22.8", "14.7", "~99% (*)"],
        ["cold_start", "120.0", "90.5", "31.8", "77.6 (86%)"],
    ],
    [150, 90, 90, 70, 71], ["l", "r", "r", "r", "r"],
)
para("* +delayed/shift/drift (structurally identical). (*) same-seed comparison.")

sec("4  Probability analysis (ranking only)")
para(
    "Per pair: one shared shock s ~ N(0, 0.45^2); per direction accept ~",
    "Bern(sigma(-0.25 + bias + fit + s + drift)), respond ~ Bern(rate); date",
    "w.p. 0.78; second yes sigma(0.15 + sbias + s + 0.4[goal match]).",
    "Directions correlate (rho ~ 0.43): integrate over s (error < 5e-6),",
    "never multiply marginals. Replay validates ranking (top decile 0.041",
    "pred / 0.033 emp vs bottom 0.002 / 0.000). Funnel (greedy, dev.): 91.6",
    "-> 52.0 respond (57%) -> 12.6 mutual (24%, controllable) -> 10.3 dates",
    "-> 3.7 second-yes -> 1.3 in-window -> ~1.0 MSMI. The 3-day window alone",
    "destroys ~64%. Corrected joint (one E over s of all four sigmoids, both",
    "response stages): sum ~1.07-1.15 vs ~1.0 realised -- greedy at 87-93%",
    "of ceiling. Top/bottom-half lift 1.38x (fit) -> 1.48x (+oracle rate).",
    "DE weight search ruled out: range 0.132 < noise 0.356. Common random",
    "numbers cut SE 1.7x and are mandatory.",
)

sec("5  Proposed policy (implemented, stdlib-only)")
bul(
    "(a) Acquisition, matchability-gated: v1 wasted 183/384 soft asks",
    "(47.7%) on unmatchable members -- now gated. Zone-clustered hard",
    "clearance first; leftover on 4 live fields in tiers T0 (feasible pair)",
    "-> T1 (cleared) -> T2 (uncleared), by |w_k|. Adaptive bar.",
    "(b) Value model: observed: +w_k if equal else -w_k/2; unknown:",
    "E = w_k*(1.5/|S_k| - 0.5) under uniform prior (exact; kills the",
    "None-equals-None artefact).",
    "(c) Weights: single-pass streaming logistic on matured responses",
    "(v1 re-fed events ~55x -- fixed); eta = 0.25/sqrt(1+n), anchor 0.10",
    "to dev prior, clip 1.0. Synthetic check: pace 0.83 vs true 0.80.",
    "(d) Allocation: exact general-graph MWM (branch-and-bound, warm start):",
    "268/268 fuzz-optimal vs brute force, < 1 ms/day. Graph is",
    "non-bipartite (~47% same-gender acceptance).",
)

sec("6  Missing and delayed data")
para(
    "Declined != not-asked (permanent, never bought). Missing responses are",
    "censored, never imputed; only matured outcomes update; pending queue by",
    "expected observation day. Assignment-day snapshots (no leakage; static",
    "fields remove staleness). No IPS (propensity null; pool splits only;",
    "same-seed simulator comparisons). Memory ~10 KB. Thompson sampling over",
    "pool priors; rank-only decisions; posterior-predictive coverage check as",
    "wrong-belief detector.",
)

sec("7  Evaluation")
para(
    "Baselines (identical seeds, all families): greedy, no-clarification,",
    "random-feasible. Ablations: no soft acquisition; fixed vs online weights;",
    "greedy-fill vs MWM; no drift scheduling; bandit off; theta sweep.",
    "Measured ladder (seeds 101-106, paired):",
)
table(
    ["Configuration", "MSMI/100", "vs greedy", "Mutual"],
    [
        ["Organiser greedy", "0.333", "--", "10.14"],
        ["Candidate v1", "0.347", "+4.2%", "10.56"],
        ["VOI ask + greedy match", "0.389", "+16.7%", "10.50"],
        ["Full (VOI+MWM+adaptive)", "0.444", "+33.3%", "10.81"],
        ["Full (VOI+MWM+fixed)", "0.514", "+54.2%", "11.20"],
    ],
    [221, 80, 80, 90], ["l", "r", "r", "r"],
)
para(
    "Honest reading: min detectable ~0.175 at n = 36, so only the fixed",
    "headline reaches it; per-variant cells (n = 6) are illustrative. Ladder",
    "is monotone, mutual +10%, and fixed beats adaptive even in shift (0.500",
    "vs 0.417) -- as noise theory predicts (~50 trials cannot carry 4",
    "weights). Cost note: we spend ~409 vs ~198 ask units (tiebreaker loss;",
    "Round 2: cost-capped ablation). Harness smoke test (seed 101, dev.):",
    "greedy msmi = 1 -> team msmi = 3, both valid, 0.15 s/call.",
)

sec("8  Failure cases and limitations")
bul(
    "Withheld answers (~1/3 unmatchable, kept in denominators); saturated",
    "sparse supply; 8-day-busy coupling (hence global matching);",
    "right-censoring (day-55 intros resolve ~day 77, unseen by policy).",
    "All outcomes synthetic -- no evidence about real relationships. Kit",
    "vendored verbatim (pin a8e26b3, MIT + synthetic-data licence); our",
    "work in research/; branch merged with history kept, policy moved out",
    "of the kit subtree. Reproduce: unittest (22 OK) + verify_data.py, then",
    "analysis/{prob_model,validate_model,why_cap,ceiling,value_spread,",
    "candidate,verify_team_policy,team_policy_eval}.py.",
)

sec("References")
para(
    "[1] Zong et al. Learn to Match. arXiv:2606.06744. [2] Li, Wang & Kong,",
    "Bandit Learning in Matching Markets survey, IJCAI 2025 (Liu-Mania-",
    "Jordan; Basu; Athanasopoulos et al. UAI 2026). [3] Udwani, non-adaptive",
    "greedy 1/2-competitive, arXiv:2403.18059. [4] Saar-Tsechansky et al.,",
    "Active Feature-Value Acquisition, Mgmt Sci 2009; Ma et al., NeurIPS",
    "2023. [5] Thompson (1933); Russo et al. (2018); Agarwal & Goyal (2013);",
    "Garivier & Moulines (2011); Joulani et al. (2013); Lakkaraju et al.",
    "(2017); Das & Kamenica (2005); Howard (1966). [6] Hayashi et al., OPE",
    "for matching markets, RecSys 2025. [7] Segal-Halevi (2011); Drummond &",
    "Boutilier, IJCAI 2013; Blum et al., OR 68(2) 2020.",
)

# ---------------------------------------------------------------- builder
def esc(s):
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build():
    pages = [[]]  # list of pages; each page is a list of ops
    lines = [0]

    def need(n):
        if lines[0] + n > MAX_LINES:
            pages.append([])
            lines[0] = 0

    def emit(font, size, x, text, dy=None):
        pages[-1].append((font, size, x, text))
        lines[0] += dy if dy else 1

    y_of = lambda i: PAGE_H - MT - i * LEAD
    _ = y_of

    # title block
    for kind, t in TITLE:
        need(2)
        if kind == "t1":
            emit("HB", 15, None, t)
        elif kind == "t2":
            emit("HB", 11, None, t)
        else:
            emit("H", 9, None, t)
    need(1)
    pages[-1].append(("RULE",))
    lines[0] += 1

    for blk in BLOCKS:
        if blk[0] == "sec":
            need(3)
            pages[-1].append(("SPACE",))
            lines[0] += 1
            emit("HB", 12, ML, blk[1])
        elif blk[0] == "para":
            need(len(blk[1]) + 1)
            for ln in blk[1]:
                emit("H", BODY, ML, ln)
            pages[-1].append(("SPACE",))
            lines[0] += 1
        elif blk[0] == "bul":
            need(len(blk[1]) + 1)
            for ln in blk[1]:
                pre = "- " if not ln.startswith("(") else "  "
                emit("H", BODY, ML, pre + ln)
            pages[-1].append(("SPACE",))
            lines[0] += 1
        elif blk[0] == "table":
            headers, rows, widths, aligns = blk[1]
            need(len(rows) + 3)
            xs = []
            x = ML + (TW - sum(widths)) / 2
            for w in widths:
                xs.append(x)
                x += w
            for h, xx in zip(headers, xs):
                emit("HB", 9, xx, h, dy=0)
            lines[0] += 1
            pages[-1].append(("TRULE", (xs[0], xs[0] + sum(widths))))
            lines[0] += 1
            for r in rows:
                for cell, xx, w, a in zip(r, xs, widths, aligns):
                    emit("H", 9, (xx, w, a), cell, dy=0)
                lines[0] += 1
            pages[-1].append(("SPACE",))
            lines[0] += 1
    return pages


def render(pages, out):
    objs = []
    # 1 catalog, 2 pages, fonts...
    n_pages = len(pages)
    # object numbering: 1 catalog, 2 pages, 3..3+n-1 page objs,
    # then contents, then 2 fonts
    f1 = 3 + 2 * n_pages  # Helvetica
    f2 = f1 + 1  # Helvetica-Bold
    kids = " ".join("%d 0 R" % (3 + i) for i in range(n_pages))
    objs.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objs.append("2 0 obj\n<< /Type /Pages /Kids [%s] /Count %d >>\nendobj\n" % (kids, n_pages))
    for i, ops in enumerate(pages):
        cnum = 3 + n_pages + i
        objs.append(
            "%d 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.2f %.2f] "
            "/Resources << /Font << /F1 %d 0 R /F2 %d 0 R >> >> /Contents %d 0 R >>\nendobj\n"
            % (3 + i, PAGE_W, PAGE_H, f1, f2, cnum)
        )
    for i, ops in enumerate(pages):
        parts = []
        # footer
        parts.append(
            "BT /F1 8 Tf %.2f %.2f Td (%s) Tj ET"
            % (ML, MB - 22, esc("Round 1 submission -- sequential-matching-hackathon @ b1e1b46  |  p. %d/%d" % (i + 1, n_pages)))
        )
        ln = 0
        for op in ops:
            if op[0] == "SPACE":
                ln += 1
                continue
            if op[0] == "RULE":
                y = PAGE_H - MT - ln * LEAD + 4
                parts.append("%.2f %.2f m %.2f %.2f l S" % (ML, y, ML + TW, y))
                ln += 1
                continue
            if op[0] == "TRULE":
                y = PAGE_H - MT - ln * LEAD + 4
                parts.append("%.2f %.2f m %.2f %.2f l S" % (op[1][0], y, op[1][1], y))
                ln += 1
                continue
            font, size, x, text = op
            y = PAGE_H - MT - ln * LEAD
            if x is None:  # centered
                approx = len(text) * size * (0.60 if font == "HB" else 0.55)
                x = ML + max(0, (TW - approx) / 2)
            elif isinstance(x, tuple):  # table cell: (x, colw, align)
                xx, colw, al = x
                if al == "r":
                    est = len(text) * size * 0.556
                    x = xx + max(0, colw - est)
                else:
                    x = xx
            parts.append(
                "BT /%s %.1f Tf %.2f %.2f Td (%s) Tj ET" % ("F2" if font == "HB" else "F1", size, x, y, esc(text))
            )
            ln += 1
        stream = "\n".join(parts)
        objs.append(
            "%d 0 obj\n<< /Length %d >>\nstream\n%s\nendstream\nendobj\n"
            % (3 + n_pages + i, len(stream.encode("latin-1")), stream)
        )
    objs.append("%d 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>\nendobj\n" % f1)
    objs.append("%d 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>\nendobj\n" % f2)

    pdf = ["%PDF-1.4\n"]
    offsets = {}
    pos = len(pdf[0].encode("latin-1"))
    for o in objs:
        # object number = first token
        num = int(o.split(" ")[0])
        offsets[num] = pos
        pos += len(o.encode("latin-1"))
    xref_pos = pos
    n = max(offsets) + 1
    xref = ["xref\n0 %d\n" % n, "0000000000 65535 f \n"]
    for k in range(1, n):
        xref.append("%010d 00000 n \n" % offsets[k])
    trailer = "trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (n, xref_pos)
    data = pdf[0] + "".join(objs) + "".join(xref) + trailer
    out.write_bytes(data.encode("latin-1"))


if __name__ == "__main__":
    pages = build()
    render(pages, HERE / "R1_SUBMISSION.pdf")
    print("pages: %d -> %s" % (len(pages), HERE / "R1_SUBMISSION.pdf"))
