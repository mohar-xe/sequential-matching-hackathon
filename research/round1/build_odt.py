"""Build R1_SUBMISSION.odt (stdlib only) for LibreOffice -> PDF conversion.

Two-column A4 paper look: Liberation Serif body (Times metrics), justified +
hyphenated, real tables, styled formula runs, page-number footer.
Usage: python3 build_odt.py   (writes R1_SUBMISSION.odt next to itself)
"""
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
HASH = "b1e1b46"
BODY_FONT = "Liberation Serif"
HEAD_FONT = "Liberation Sans"
MONO_FONT = "Liberation Mono"

# ---------------------------------------------------------------- helpers
B, I, BI, MO = "B", "I", "BI", "MO"  # span kinds


def run(text, kind=None):
    return (text, kind)


def esc(t):
    return escape(t, {'"': "&quot;"})


def span_xml(text, kind):
    t = esc(text)
    if kind is None:
        return t
    return '<text:span text:style-name="T_%s">%s</text:span>' % (kind, t)


def para(style, runs, attrs=""):
    inner = "".join(span_xml(t, k) for t, k in runs)
    return "<text:p text:style-name=\"%s\"%s>%s</text:p>" % (style, attrs, inner)


def P(style, text):
    return para(style, [run(text)])


# ---------------------------------------------------------------- content
doc = []

doc.append(para("P_Title", [run("Sequential Matching under Reciprocal Uncertainty:")]))
doc.append(para("P_Title2", [run("Ceilings, Acquisition, and Exact Matching")]))
doc.append(para("P_Sub", [run("Round 1 Research Submission \u2014 Vouchsafe Sequential Matching Hackathon")]))
doc.append(para("P_Auth", [run("Team mohar-xe \u2014 Kit release 1.0.0 \u2014 9 October 2026 \u2014 built from " + HASH)]))
doc.append(para("P_Rule", [run("")]))
doc.append(para("P_AbHead", [run("Abstract")]))
doc.append(P("P_Abstract",
    "We study the Sequential Matching Hackathon: twice daily for 60 days (+40 follow-up), a policy "
    "spends a 12-unit ask budget and proposes disjoint feasible pairs to maximise mutual second-meeting "
    "intentions (MSMI) per 100 arrived members, averaged over six scenario families. Our contribution is "
    "measurement-first. Only 1.24% of pairs are hard-feasible (0.24% sparse); one third of members can "
    "never be matched; 98.5% of every introduction is lost to mechanisms no policy controls, above all a "
    "3-day answer window discarding about 64% of successes. The only reliable pair signal is agreement on "
    "4 of 7 soft fields. The reachable edge set caps introductions, and greedy already realises 89\u201399% "
    "of the corrected ceiling \u2014 so this is an acquisition problem, not an RL problem (honest headroom "
    "about 1.3\u20131.6\u00d7). Our stdlib-only policy (matchability-gated VOI acquisition, exact general-graph "
    "maximum-weight matching, single-pass anchored online weights) climbs a paired ladder 0.333 \u2192 0.514 "
    "(directional; the headline is underpowered about 20\u00d7). Fixed-prior beats adaptive everywhere, as "
    "noise theory predicts. All numbers come from 13 stdlib-only scripts against the public simulator; "
    "every person and outcome is synthetic."))

doc.append(para("P_H1", [run("1  Problem interpretation")]))
doc.append(P("P_Body",
    "Each of 60 decision days the policy (i) issues clarification asks costing at most 12 units "
    "(hard-constraint bundle = 3, one named soft field = 1; declined is permanent), then (ii) proposes a "
    "batch of pairwise-disjoint feasible pairs (empty = wait). Pairs must pass reciprocal hard checks both "
    "ways (age, gender, zones, smoking, children, structure, schedule), be unintroduced, with all hard fields "
    "known. Feedback matures with delay; 40 follow-up days resolve outcomes. MSMI needs both yes \u2192 date "
    "within 30 days \u2192 both second-yes within 3 days of the date. Score: mean over six families \u00d7 20 "
    "private seeds. Withheld: seeds, arrivals, hidden preferences, reply rates. Limits: JSON \u2264 1 MiB, "
    "10 s/call, 2 cores, offline; memory self-carried."))
doc.append(P("P_Body",
    "Three facts reframe the challenge: (a) soft values are static (0 of 200 changed over 60 days) \u2014 "
    "asking is a lossless one-time purchase; drift shifts the scoring formula (\u22120.5 logit from day 35), "
    "not preferences. (b) Only 11.4% of pairs (3.8% cold start) show complete fit before asking, vs 56% after "
    "buying the four fields. (c) Only goal, pace, lifestyle and conversations enter the scorer "
    "(0.70/0.40/0.25/0.20; shifted: 0.25/0.80/\u22120.25/0.50); the other three soft fields are dead weight "
    "(force-changing them leaves MSMI bit-identical) and must never be bought."))

doc.append(para("P_H1", [run("2  Hypothesis")]))
doc.append(P("P_Body",
    "A reciprocal pair scorer learned online from delayed/censored feedback, with global maximum-weight "
    "allocation and VOI-ranked clarification, raises MSMI/100 over greedy via (i) learned weights vs uniform "
    "soft-count, (ii) global allocation vs greedy fill, (iii) soft acquisition vs none, (iv) drift-aware "
    "sequencing. Full RL is rejected: Learn2Match\u2019s friction-loss result predicts failure exactly when "
    "volume is saturated and the gap is friction. LLM scoring is rejected (offline CPU; template conversations)."))

doc.append(para("P_H1", [run("3  Reciprocal feasibility: introductions are capped")]))
doc.append(P("P_Body",
    "First-failing-rule attribution on truth: who-to-meet 57%, age 26%, geography 12% (19% sparse); the rest "
    "about 4% combined. P(matchable) = 0.667/0.656/0.600 \u2014 one third can never be matched. |E| is a hard "
    "ceiling (no repeats). An apparent sparse contradiction (|E| = 20.9 vs 22.6 realised) was a seed-set "
    "mismatch: identical seeds give |E| = 22.8 vs 22.6 (99% saturated), assigned a subset of reachable with "
    "0 violations. Clarification is supply-limited: 720 units vs about 80 askable members, exhausted by day "
    "20. Sparse is saturated, not an opportunity; headroom is cold-start volume plus shift/drift ordering."))


def table(headers, rows, widths, header_bold=True):
    cols = "".join(
        '<table:table-column table:style-name="C%d" table:number-columns-repeated="1"/>' % i
        for i in range(len(widths)))
    col_styles = "".join(
        '<style:style style:name="C%d" style:family="table-column">'
        '<style:table-column-properties style:column-width="%.2fcm"/></style:style>'
        % (i, w) for i, w in enumerate(widths))
    doc.append(("__colstyles__", col_styles))
    h = "".join('<table:covered-table-cell/>' if c is None else
                '<table:table-cell table:style-name="TC_H"><text:p text:style-name="P_TCH">%s</text:p>'
                '</table:table-cell>' % esc(c) for c in headers)
    out = ['<table:table table:name="T%d" table:style-name="TBL">%s'
           '<table:table-header-rows>'
           '<table:table-row table:style-name="RH"><table:table-cells>%s</table:table-cells>'
           '</table:table-row></table:table-header-rows>' % (len([d for d in doc if isinstance(d, tuple)]), cols, h)]
    for ri, r in enumerate(rows):
        last = (ri == len(rows) - 1)
        cells = "".join(
            '<table:table-cell table:style-name="%s"><text:p text:style-name="P_TC">%s</text:p>'
            '</table:table-cell>' % ("TC_B" if last else "TC", esc(c)) for c in r)
        out.append('<table:table-row table:style-name="RR"><table:table-cells>%s</table:table-cells>'
                   '</table:table-row>' % cells)
    out.append('</table:table>')
    doc.append(("__table__", "".join(out)))


table(["Family", "Match", "|E|", "Max", "Greedy"],
      [["dev*", "133.4", "103.1", "34.5", "~92 (89%)"],
       ["sparse", "131.2", "20.9\u201322.8", "14.7", "~99% \u2020"],
       ["cold start", "120.0", "90.5", "31.8", "77.6 (86%)"]],
       [3.4, 1.4, 1.7, 1.4, 1.9])
doc.append(para("P_TNote", [run("\u2020 Match = matchable/200; Max = max matching. Same-seed comparison. * +delayed/shift/drift.")]))

doc.append(para("P_H1", [run("4  Probability analysis (ranking only)")]))
doc.append(para("P_Body", [
    run("Per assigned pair: one shared shock "), run("s", I),
    run(" ~ N(0, 0.45\u00b2), then per direction "), run("accept", I),
    run(" ~ Bern(\u03c3(\u22120.25 + "), run("b", I), run("a", "SUB"),
    run(" + fit + "), run("s", I), run(" + drift)), "), run("respond", I),
    run(" ~ Bern("), run("r", I), run("a", "SUB"), run("); date w.p. 0.78; second yes \u03c3(0.15 + "),
    run("sb", I), run("a", "SUB"), run(" + "), run("s", I),
    run(" + 0.4[goal match]). Directions correlate (\u03c1 \u2248 0.43 via fit + shared): never multiply "
          "marginals; integrate over "), run("s", I),
    run(" (41-node rule, error < 5\u00d710"), run("\u207b\u2076", None),
    run("). Replay validates ranking (top decile 0.041 pred / 0.033 emp vs bottom 0.002 / 0.000).")]))
doc.append(P("P_Body",
    "Funnel (greedy, development): 91.6 intros \u2192 52.0 respond (57%) \u2192 12.6 mutual (24%, the "
    "controllable stage) \u2192 10.3 dates \u2192 3.7 second-yes \u2192 1.3 in-window \u2192 about 1.0 MSMI. "
    "The 3-day window ((3/5)\u00b2) alone destroys about 64%. A draft \u201c0.56 vs 1.0\u201d under-prediction "
    "was a mislabelled single-batch sum; the corrected joint,"))
doc.append(para("P_Formula", [
    run("P", I), run("MSMI", "SUB"), run(" = "), run("r", I), run("1", "SUB"), run("r", I), run("2", "SUB"),
    run(" (0.78)(0.36) "), run("E", I), run("s", "SUB"), run(" ["),
    run("A", I), run("a", "SUB"), run("A", I), run("b", "SUB"),
    run("S", I), run("a", "SUB"), run("S", I), run("b", "SUB"), run("]")]))
doc.append(P("P_Body",
    "with one expectation over the shared shock of all four sigmoids (acceptance \u00d7 second-yes, both "
    "directions) and both response stages, gives a sum of about 1.07\u20131.15 vs about 1.0 realised \u2014 "
    "greedy at 87\u201393% of ceiling. Rankings are identical under either form. Top/bottom-half lift is "
    "1.38\u00d7 (fit) \u2192 1.48\u00d7 (+oracle rate), so member latents (1\u20132 obs/member) are not worth "
    "learning. Black-box weight search is ruled out by measurement: exploitable range 0.132 < single-run "
    "noise 0.356. Common random numbers cut SE 1.7\u00d7 and are mandatory."))

doc.append(para("P_H1", [run("5  Proposed policy (implemented, stdlib-only)")]))
for tag, runs in [
    ("(a)", [run("Acquisition, matchability-gated. ", B),
             run("v1 spent 183 of 384 soft asks (47.7%) on permanently unmatchable members \u2014 now gated "
                 "out. Zone-clustered hard clearance first; leftover on the 4 live fields in tiers T0 "
                 "(feasible pair) \u2192 T1 (cleared) \u2192 T2 (uncleared), by magnitude ") ,
             run("w", I), run("k", "SUB"),
             run(", with partner-observed preference. Adaptive bar.")]),
    ("(b)", [run("Value model. ", B),
             run("Observed both sides: "), run("+w", I), run("k", "SUB"),
             run(" if equal else "), run("\u2212w", I), run("k", "SUB"),
             run("/2; unknown:")]),
]:
    doc.append(para("P_Bull", [run(tag + "  ")] + runs))
doc.append(para("P_Formula", [
    run("E", I), run("[contrib", None), run("k", "SUB"), run("] = "), run("w", I), run("k", "SUB"),
    run(" (1.5/N", None), run("k", "SUB"), run(" \u2212 0.5),  ", None),
    run("N", I), run("k", "SUB"), run(" options (exact here; kills the None-equals-None artefact).", None)]))
for tag, runs in [
    ("(c)", [run("Weights. ", B),
             run("Single-pass streaming logistic on matured directional responses (v1 re-fed cumulative "
                 "feedback, updating day-5 events about 55\u00d7 \u2014 fixed via seen-event tracking); "),
             run("\u03b7 = 0.25/\u221a(1+n)", I), run(", anchor "),
             run("\u03bb = 0.10", I), run(" to the development prior, clip 1.0. Synthetic check: pace 0.83 "
                 "vs true 0.80.")]),
    ("(d)", [run("Allocation. ", B),
             run("Exact general-graph MWM (branch-and-bound over components, greedy warm start; the graph is "
                 "non-bipartite, about 47% same-gender acceptance): 268/268 fuzz-optimal vs brute force, "
                 "< 1 ms/day. A \u22120.10 post-day-35 offset is a declared fixed parameter, not a drift "
                 "detector (Round 2: trigger from the observed acceptance drop).")]),
]:
    doc.append(para("P_Bull", [run(tag + "  ")] + runs))

doc.append(para("P_H1", [run("6  Missing and delayed data")]))
doc.append(P("P_Body",
    "Declined != not-asked (permanent, never bought). Missing responses are censored, never imputed; only "
    "matured outcomes update; pending queue keyed on expected observation day. Assignment-day feature "
    "snapshots (no leakage; static fields remove staleness). No IPS claims (propensity null; pool splits "
    "only; same-seed simulator comparisons). Memory about 10 KB. Thompson sampling over pool priors; "
    "rank-only decisions; posterior-predictive coverage check as the wrong-belief detector."))

doc.append(para("P_H1", [run("7  Evaluation")]))
doc.append(P("P_Body",
    "Baselines (identical seeds, all families): greedy, no-clarification, random-feasible. Ablations: no "
    "soft acquisition; fixed vs online weights; greedy-fill vs MWM; no drift scheduling; bandit off; "
    "threshold sweep. Measured ladder (seeds 101\u2013106, paired):"))
table(["Configuration", "MSMI", "vs", "Mut"],
      [["Organiser greedy", "0.333", "\u2014", "10.14"],
       ["Candidate v1", "0.347", "+4.2%", "10.56"],
       ["VOI + greedy", "0.389", "+16.7%", "10.50"],
       ["Full (adaptive)", "0.444", "+33.3%", "10.81"],
       ["Full (fixed)", "0.514", "+54.2%", "11.20"]],
      [4.6, 1.9, 1.9, 1.9])
doc.append(P("P_Body",
    "Honest reading: minimum detectable effect about 0.175 at n = 36, so only the fixed headline reaches it "
    "and per-variant cells (n = 6) are illustrative. The ladder is monotone, mutual acceptances rise 10%, "
    "and fixed beats adaptive even in shift (0.500 vs 0.417) \u2014 exactly as noise theory predicts (about "
    "50 trials cannot carry 4 weights). Cost note: we spend about 409 vs about 198 ask units (tiebreaker "
    "loss; Round 2: cost-capped ablation). Harness smoke test (official evaluate.py, seed 101, development): "
    "greedy msmi = 1 \u2192 team msmi = 3, both valid, 0.15 s/call."))

doc.append(para("P_H1", [run("8  Failure cases and limitations")]))
doc.append(para("P_Bull", [run("\u2022  "),
    run("Withheld answers (about 1/3 unmatchable, kept in denominators); saturated sparse supply; 8-day-busy "
        "schedule coupling (hence global matching); right-censoring (day-55 intros resolve about day 77, "
        "unseen by the policy).")]))
doc.append(para("P_Bull", [run("\u2022  "),
    run("All outcomes synthetic \u2014 no evidence about real relationships. Kit vendored verbatim (pin "
        "a8e26b3, MIT + synthetic-data licence); our work is in research/; the team branch merged with "
        "history kept and policy files moved out of the kit subtree. Reproduce: unittest (22 OK) + "
        "verify_data.py, then the analysis scripts (prob_model, validate_model, why_cap, ceiling, "
        "value_spread, candidate, verify_team_policy, team_policy_eval).")]))
doc.append(para("P_H1", [run("References")]))
for ref in [
    "[1]  Zong et al. Learn to Match. arXiv:2606.06744.",
    "[2]  Li, Wang & Kong, bandit learning in matching markets survey, IJCAI 2025 (Liu-Mania-Jordan; Basu; "
    "Athanasopoulos et al. UAI 2026).",
    "[3]  Udwani, non-adaptive greedy 1/2-competitive, arXiv:2403.18059.",
    "[4]  Saar-Tsechansky et al., active feature-value acquisition, Mgmt Sci 2009; Ma et al., NeurIPS 2023.",
    "[5]  Thompson (1933); Russo et al. (2018); Agarwal & Goyal (2013); Garivier & Moulines (2011); Joulani "
    "et al. (2013); Lakkaraju et al. (2017); Das & Kamenica (2005); Howard (1966).",
    "[6]  Hayashi et al., off-policy evaluation for matching markets, RecSys 2025.",
    "[7]  Segal-Halevi (2011); Drummond & Boutilier, IJCAI 2013; Blum et al., OR 68(2) 2020.",
]:
    doc.append(para("P_Ref", [run(ref)]))

doc.append(("__section__", "appendix"))
doc.append(para("P_H1", [run("Appendix: reproducing this")]))
for line in ["cd The-Sequential-Matching-Problem",
             "python3 -m unittest -v && python3 verify_data.py",
             "cd ../research/analysis",
             "python3 verify_team_policy.py",
             "python3 team_policy_eval.py --seeds 101,102,103,104,105,106"]:
    doc.append(para("P_Code", [run(line, MO)]))

# ---------------------------------------------------------------- styles (see CONTENT_TMPL / STYLES_XML below)


CONTENT_TMPL = """<?xml version="1.0" encoding="UTF-8"?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg:1.0" office:version="1.2">
<office:scripts/><office:font-face-decls>
<style:font-face style:name="Liberation Serif" svg:font-family="Liberation Serif"/>
<style:font-face style:name="Liberation Sans" svg:font-family="Liberation Sans"/>
<style:font-face style:name="Liberation Mono" svg:font-family="Liberation Mono"/>
<style:font-face style:name="DejaVu Serif" svg:font-family="DejaVu Serif"/>
</office:font-face-decls>
<office:automatic-styles>
<style:style style:name="P_Title" style:family="paragraph" style:parent-style-name="Standard" style:master-page-name="MP"><style:paragraph-properties fo:text-align="center" fo:margin-top="0.1cm" fo:margin-bottom="0.05cm"/><style:text-properties fo:font-family="Liberation Serif" fo:font-size="20pt" fo:font-weight="bold"/></style:style>
<style:style style:name="P_Title2" style:family="paragraph" style:parent-style-name="Standard"><style:paragraph-properties fo:text-align="center" fo:margin-bottom="0.15cm"/><style:text-properties fo:font-family="Liberation Serif" fo:font-size="14pt" fo:font-weight="bold"/></style:style>
<style:style style:name="P_Sub" style:family="paragraph" style:parent-style-name="Standard"><style:paragraph-properties fo:text-align="center" fo:margin-bottom="0.1cm"/><style:text-properties fo:font-family="Liberation Serif" fo:font-size="10.5pt" fo:font-weight="bold"/></style:style>
<style:style style:name="P_Auth" style:family="paragraph" style:parent-style-name="Standard"><style:paragraph-properties fo:text-align="center" fo:margin-bottom="0.1cm"/><style:text-properties fo:font-family="Liberation Serif" fo:font-size="9pt"/></style:style>
<style:style style:name="P_Rule" style:family="paragraph" style:parent-style-name="Standard"><style:paragraph-properties fo:margin-top="0.15cm" fo:margin-bottom="0.25cm" fo:border-bottom="0.75pt solid #000000" fo:padding-bottom="0.05cm"/><style:text-properties fo:font-size="2pt"/></style:style>
<style:style style:name="P_AbHead" style:family="paragraph" style:parent-style-name="Standard"><style:paragraph-properties fo:text-align="center" fo:margin-bottom="0.1cm"/><style:text-properties fo:font-family="Liberation Serif" fo:font-size="10.5pt" fo:font-weight="bold"/></style:style>
<style:style style:name="P_Abstract" style:family="paragraph" style:parent-style-name="Standard"><style:paragraph-properties fo:text-align="justify" fo:margin-left="1.1cm" fo:margin-right="1.1cm" fo:margin-bottom="0.25cm" fo:hyphenate="true" fo:language="en" fo:country="US"/><style:text-properties fo:font-family="Liberation Serif" fo:font-size="9.5pt"/></style:style>
<style:style style:name="P_H1" style:family="paragraph" style:parent-style-name="Standard"><style:paragraph-properties fo:margin-top="0.35cm" fo:margin-bottom="0.12cm" fo:keep-with-next="always"/><style:text-properties fo:font-family="Liberation Sans" fo:font-size="12pt" fo:font-weight="bold" fo:color="#1a1a1a"/></style:style>
<style:style style:name="P_Body" style:family="paragraph" style:parent-style-name="Standard"><style:paragraph-properties fo:text-align="justify" fo:margin-bottom="0.12cm" fo:hyphenate="true" fo:language="en" fo:country="US" fo:orphans="2" fo:widows="2"/><style:text-properties fo:font-family="Liberation Serif" fo:font-size="10pt"/></style:style>
<style:style style:name="P_Bull" style:family="paragraph" style:parent-style-name="Standard"><style:paragraph-properties fo:text-align="justify" fo:margin-left="0.65cm" fo:text-indent="-0.45cm" fo:margin-bottom="0.1cm" fo:hyphenate="true" fo:language="en" fo:country="US" fo:orphans="2" fo:widows="2"/><style:text-properties fo:font-family="Liberation Serif" fo:font-size="10pt"/></style:style>
<style:style style:name="P_Formula" style:family="paragraph" style:parent-style-name="Standard"><style:paragraph-properties fo:text-align="center" fo:margin-top="0.12cm" fo:margin-bottom="0.12cm"/><style:text-properties fo:font-family="Liberation Serif" fo:font-size="10pt"/></style:style>
<style:style style:name="P_TNote" style:family="paragraph" style:parent-style-name="Standard"><style:paragraph-properties fo:margin-bottom="0.12cm"/><style:text-properties fo:font-family="Liberation Serif" fo:font-size="8.5pt"/></style:style>
<style:style style:name="P_Ref" style:family="paragraph" style:parent-style-name="Standard"><style:paragraph-properties fo:margin-left="0.7cm" fo:text-indent="-0.5cm" fo:margin-bottom="0.08cm"/><style:text-properties fo:font-family="Liberation Serif" fo:font-size="9pt"/></style:style>
<style:style style:name="P_Code" style:family="paragraph" style:parent-style-name="Standard"><style:paragraph-properties fo:margin-bottom="0cm" fo:hyphenate="false" fo:language="en" fo:country="US"/><style:text-properties fo:font-family="Liberation Mono" fo:font-size="8.5pt"/></style:style>
<style:style style:name="TBL" style:family="table"><style:table-properties table:align="center" style:width="8.6cm" fo:margin-top="0.1cm" fo:margin-bottom="0.1cm"/></style:style>
<style:style style:name="TC" style:family="table-cell"><style:table-cell-properties fo:padding="0.06cm" fo:border="none"/></style:style>
<style:style style:name="TC_H" style:family="table-cell"><style:table-cell-properties fo:padding="0.06cm" fo:border-top="0.75pt solid #000000" fo:border-bottom="0.75pt solid #000000"/></style:style>
<style:style style:name="TC_B" style:family="table-cell"><style:table-cell-properties fo:padding="0.06cm" fo:border-bottom="0.75pt solid #000000"/></style:style>
<style:style style:name="P_TC" style:family="paragraph" style:parent-style-name="Standard"><style:paragraph-properties fo:margin-bottom="0cm"/><style:text-properties fo:font-family="Liberation Serif" fo:font-size="9pt"/></style:style>
<style:style style:name="P_TCH" style:family="paragraph" style:parent-style-name="Standard"><style:paragraph-properties fo:margin-bottom="0cm"/><style:text-properties fo:font-family="Liberation Serif" fo:font-size="9pt" fo:font-weight="bold"/></style:style>
<style:style style:name="T_B" style:family="text"><style:text-properties fo:font-weight="bold"/></style:style>
<style:style style:name="T_I" style:family="text"><style:text-properties fo:font-style="italic"/></style:style>
<style:style style:name="T_BI" style:family="text"><style:text-properties fo:font-weight="bold" fo:font-style="italic"/></style:style>
<style:style style:name="T_MO" style:family="text"><style:text-properties fo:font-family="Liberation Mono"/></style:style>
<style:style style:name="T_SUB" style:family="text"><style:text-properties style:text-position="sub 58%"/></style:style>
</office:automatic-styles>
<office:body><office:text>
BODY
</office:text></office:body>
</office:document-content>
"""

STYLES_XML = """<?xml version="1.0" encoding="UTF-8"?>
<office:document-styles xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0" xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg:1.0" office:version="1.2">
<office:font-face-decls>
<style:font-face style:name="Liberation Serif" svg:font-family="Liberation Serif"/>
<style:font-face style:name="Liberation Sans" svg:font-family="Liberation Sans"/>
<style:font-face style:name="Liberation Mono" svg:font-family="Liberation Mono"/>
</office:font-face-decls>
<office:styles>
<style:style style:name="Standard" style:family="paragraph" style:class="text"/>
<style:default-style style:family="paragraph"><style:paragraph-properties fo:hyphenation-ladder-count="no-limit"/><style:text-properties fo:font-family="Liberation Serif" fo:font-size="10pt" fo:language="en" fo:country="US"/></style:default-style>
<style:style style:name="P_Foot" style:family="paragraph" style:parent-style-name="Standard"><style:paragraph-properties><style:tab-stops><style:tab-stop style:position="16.6cm" style:type="right"/></style:tab-stops></style:paragraph-properties><style:text-properties fo:font-family="Liberation Serif" fo:font-size="8pt"/></style:style>
</office:styles>
<office:automatic-styles>
<style:page-layout style:name="PL">
<style:page-layout-properties page-width="21cm" page-height="29.7cm" print-orientation="portrait" style:print-page-layout="true" fo:margin-top="1.7cm" fo:margin-bottom="2.0cm" fo:margin-left="2.2cm" fo:margin-right="2.2cm" style:num-format="1">
</style:page-layout-properties>
<style:footer-style><style:header-footer-properties fo:min-height="0.8cm" fo:margin-top="0.3cm"/></style:footer-style>
</style:page-layout>
</office:automatic-styles>
<office:master-styles>
<style:master-page style:name="MP" style:page-layout-name="PL"><style:footer>
<text:p text:style-name="P_Foot">Round 1 submission \u2014 built from HASHLEFT <text:tab/>Page <text:page-number text:select-page="current">1</text:page-number></text:p>
</style:footer></style:master-page>
</office:master-styles>
</office:document-styles>
"""

META_XML = """<?xml version="1.0" encoding="UTF-8"?>
<office:document-meta xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:meta="urn:oasis:names:tc:opendocument:xmlns:meta:1.0" xmlns:dc="http://purl.org/dc/elements/1.1/" office:version="1.2">
<office:meta><dc:title>Sequential Matching under Reciprocal Uncertainty: Ceilings, Acquisition, and Exact Matching</dc:title><dc:creator>mohar-xe</dc:creator><meta:creation-date>2026-10-09T00:00:00</meta:creation-date></office:meta>
</office:document-meta>
"""

MANIFEST_XML = """<?xml version="1.0" encoding="UTF-8"?>
<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0">
<manifest:file-entry manifest:full-path="/" manifest:media-type="application/vnd.oasis.opendocument.text"/>
<manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>
<manifest:file-entry manifest:full-path="styles.xml" manifest:media-type="text/xml"/>
<manifest:file-entry manifest:full-path="meta.xml" manifest:media-type="text/xml"/>
</manifest:manifest>
"""


def build():
    body, colstyles = [], []
    for item in doc:
        if isinstance(item, tuple):
            kind, payload = item
            if kind == "__colstyles__":
                colstyles.append(payload)
            elif kind == "__table__":
                body.append(payload)
            # __section__ markers intentionally ignored: single-column layout
            continue
        body.append(item)
    content = CONTENT_TMPL.replace("COLSTYLES", "".join(colstyles)).replace("BODY", "".join(body))
    styles = STYLES_XML.replace("HASHLEFT", HASH)
    out = HERE / "R1_SUBMISSION.odt"
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("mimetype", "application/vnd.oasis.opendocument.text", compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/manifest.xml", MANIFEST_XML, compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("content.xml", content, compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("styles.xml", styles, compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("meta.xml", META_XML, compress_type=zipfile.ZIP_DEFLATED)
    print("wrote", out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    build()
