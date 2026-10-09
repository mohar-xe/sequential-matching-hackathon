"""Recovered generative law of kit.py, plus exact oracle quantities.

Everything here is derived by reading kit.py. Nothing in this file is a
production claim -- it is reverse-engineered synthetic ground truth used to
measure how much headroom the hackathon metric actually has.
"""
import math
import sys

import _bootstrap  # noqa: F401  (sets sys.path portably)
from kit import Simulator, generate, eligibility, HARD, SOFT, OPTIONS  # noqa: E402

# ---------------------------------------------------------------- quadrature
_NORMAL = __import__('statistics').NormalDist()
_NPTS = 41
_LO, _HI = -4.5, 4.5
_GRID = []
_W_MID = (_HI - _LO) / _NPTS
for _i in range(_NPTS):
    _c = _LO + (_HI - _LO) * (_i + 0.5) / _NPTS
    _GRID.append((_c, _W_MID * math.exp(-_c * _c / 2.0) / math.sqrt(2 * math.pi)))


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def fit_term(ta, tb, variant='development'):
    """kit.py Simulator._prob, reproduced exactly. ta/tb are 'truth' dicts."""
    weights = {'relationship_goal': .7, 'relationship_pace': .4,
               'lifestyle': .25, 'conversations': .2}
    if variant == 'shift':
        weights = {'relationship_goal': .25, 'relationship_pace': .8,
                   'lifestyle': -.25, 'conversations': .5}
    return sum(w * (1 if ta[k] == tb[k] else -.5) for k, w in weights.items())


def drift(day, variant):
    return -.5 if variant == 'drift' and day >= 35 else 0.0


def e_shared_prod(za, zb, sig=0.45):
    """E_{shared~N(0,sig)} [ sigmoid(za+shared) * sigmoid(zb+shared) ].

    This is the correlation that makes reciprocal acceptance non-independent.
    Midpoint rule over +-4.5 sigma with 41 nodes: error ~1e-4.
    """
    tot = 0.0
    for c, phi in _GRID:
        s = sig * c
        tot += phi * sigmoid(za + s) * sigmoid(zb + s)
    return tot


def p_msmi_truth(ma, mb, day=0, variant='development', full=True):
    """Oracle P(Mutual Second-Meeting Intention) for one assigned pair.

    full=True uses every latent; full=False assumes only `fields` are known,
    which is what an informed-but-legal policy can actually compute.
    """
    ta, tb = ma['truth'], mb['truth']
    src_a = ta if full else ma['fields']
    src_b = tb if full else mb['fields']
    fit = fit_term(src_a, src_b, variant)
    dr = drift(day, variant)

    bias_a = ma['bias'] if full else 0.0
    bias_b = mb['bias'] if full else 0.0
    sb_a = ma['second_bias'] if full else 0.0
    sb_b = mb['second_bias'] if full else 0.0

    p_acc = e_shared_prod(-.25 + bias_a + fit + dr, -.25 + bias_b + fit + dr)

    ga = src_a.get('relationship_goal')
    gb = src_b.get('relationship_goal')
    gterm = .4 * (1 if (ga is not None and gb is not None and ga == gb) else 0)
    p_sec = e_shared_prod(.15 + sb_a + gterm, .15 + sb_b + gterm)

    r = ma['response_rate'] * mb['response_rate']
    return r * p_acc * 0.78 * p_sec * 0.36


def p_msmi_joint(ma, mb, day=0, variant='development'):
    """Corrected oracle P(MSMI): joint expectation over the shared shock.

    p_msmi_truth above factorizes as E[acc_a*acc_b] * E[sec_a*sec_b] and
    charges only ONE response-rate product, but kit.py draws ONE `shared`
    per pair driving BOTH stages, and requires a response in BOTH stages
    (introduction_response AND second_meeting_intention each need
    Bernoulli(response_rate)). The correct form is a single joint
    expectation E_shared[acc_a*acc_b*sec_a*sec_b] times r1*r2*0.78*0.36,
    where r1 = r2 = ra*rb. The factorized form over-predicts absolute
    levels by ~1.4x (1/0.585 second-response miss, partly offset by a
    ~1.2-1.3x shared-reuse boost); ranking is unaffected. Validated:
    sum over reachable edges, seeds 41-42 development, old 1.55-1.61 vs
    joint 1.07-1.15 (ratio ~0.70), against greedy realised ~1.0.
    """
    ta, tb = ma['truth'], mb['truth']
    src_a, src_b = ta, tb
    fit = fit_term(src_a, src_b, variant)
    dr = drift(day, variant)
    bias_a, bias_b = ma['bias'], mb['bias']
    sb_a, sb_b = ma['second_bias'], mb['second_bias']
    ga, gb = src_a.get('relationship_goal'), src_b.get('relationship_goal')
    gterm = .4 * (1 if (ga is not None and gb is not None and ga == gb) else 0)
    r1 = ma['response_rate'] * mb['response_rate']
    tot = 0.0
    for c, phi in _GRID:
        s = 0.45 * c
        tot += (phi * sigmoid(-.25 + bias_a + fit + dr + s)
                * sigmoid(-.25 + bias_b + fit + dr + s)
                * sigmoid(.15 + sb_a + gterm + s)
                * sigmoid(.15 + sb_b + gterm + s))
    return r1 * r1 * 0.78 * 0.36 * tot


def p_both_respond(ma, mb):
    return ma['response_rate'] * mb['response_rate']


# ------------------------------------------------------------ matching
def greedy_match(edges, members):
    """Max-weight matching by descending-weight greedy (baseline order)."""
    order = sorted(edges, key=lambda e: -e[0])
    used, out = set(), []
    for w, pair in order:
        if used.isdisjoint(pair):
            out.append((w, pair))
            used.update(pair)
    return out


def local_search_match(edges, members, sweeps=8):
    """Greedy seed + repeated single-pair replacement / augmentation."""
    cur = dict(greedy_match(edges, members))
    adj = {}
    for w, pair in edges:
        adj.setdefault(pair[0], []).append((w, pair))
        adj.setdefault(pair[1], []).append((w, pair))
    idx = {m: i for i, m in enumerate(members)}
    best = sum(w for w, _ in cur.values())
    for _ in range(sweeps):
        improved = False
        # drop one matched pair, greedily re-fill its two endpoints
        for mpair in list(cur):
            cur.pop(mpair)
            gained = 0.0
            taken = {i for p in cur for i in p}
            cands = []
            for endpoint in mpair:
                for w, p in adj.get(endpoint, ()):
                    if all(i not in taken for i in p):
                        cands.append((w, p))
            cands.sort(key=lambda e: -e[0])
            used, added = set(), []
            for w, p in cands:
                if used.isdisjoint(p):
                    used.update(p)
                    added.append((w, p))
            gained = sum(w for w, _ in added)
            for a in added:
                cur[a[1]] = a
            tot = sum(w for w, _ in cur.values())
            if tot > best + 1e-12:
                best = tot
                improved = True
            else:
                for a in added:
                    cur.pop(a[1], None)
                cur[mpair] = next(x for x in [(0.0, mpair)] if True)
                cur[mpair] = (next((w for w, p in edges if p == mpair), 0.0), mpair)
        if not improved:
            break
    return list(cur.values())


def feasibility_edges(members, past, use_truth=False, complete=True):
    """Feasible non-repeat pairs, optionally letting the oracle see truth."""
    out = []
    n = len(members)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = members[i], members[j]
            if (a['member_id'], b['member_id']) in past:
                continue
            if use_truth:
                ok = _truth_feasible(a, b)
            else:
                ok = eligibility(a, b)['status'] == 'feasible'
            if ok:
                out.append((a, b))
    return out


def _truth_feasible(a, b):
    """eligibility() evaluated against ground truth instead of observations."""
    fa, fb = a['truth'], b['truth']
    for x, y, fx, fy in ((a, b, fa, fb), (b, a, fb, fa)):
        if y['gender'] not in fx['who_to_meet']:
            return False
        if y['age'] < fx['age_min'] or y['age'] > fx['age_max']:
            return False
        if y['zone'] not in fx['acceptable_zones']:
            return False
        if fx['partner_smoking'] == 'no_smoking' and fy['smoking'] in ('yes', 'occasionally'):
            return False
        if fx['partner_children'] == 'no_children' and fy['has_children'] is True:
            return False
    if fa['relationship_structure'] != fb['relationship_structure']:
        return False
    if {fa['wants_children'], fb['wants_children']} == {'yes', 'no'}:
        return False
    if not (set(fa['schedule']) & set(fb['schedule'])):
        return False
    return True


# ---------------------------------------------------------- instrumentation
def decompose(sim):
    """Where does an episode lose MSMI? Follow kit.metrics() exactly."""
    events = sim.receive_feedback()
    s = {'assignments': len(sim.introductions), 'both_responded': 0,
         'mutual_acceptance': 0, 'date': 0, 'sec_both_yes': 0,
         'sec_not_both_yes': 0, 'msmi': 0, 'lost_to_window': 0,
         'lost_to_late_date': 0, 'sec_missing': 0}
    for intro in sim.introductions:
        es = [e for e in events if e['introduction_id'] == intro['introduction_id']]
        rs = [e for e in es if e['event'] == 'introduction_response']
        if len(rs) == 2 and all(e['value'] is not None for e in rs):
            s['both_responded'] += 1
        if len(rs) == 2 and all(e['value'] == 'yes' for e in rs):
            s['mutual_acceptance'] += 1
        ds = [e for e in es if e['event'] == 'date_happened' and e['value'] is True]
        if ds:
            s['date'] += 1
            if ds[0]['occurred_day'] - intro['assigned_day'] <= 30:
                ss = [e for e in es if e['event'] == 'second_meeting_intention']
                if len(ss) == 2 and all(e['value'] == 'yes' for e in ss):
                    s['sec_both_yes'] += 1
                    if all(e['occurred_day'] - ds[0]['occurred_day'] <= 3 for e in ss):
                        s['msmi'] += 1
                    else:
                        s['lost_to_window'] += 1
                else:
                    s['sec_not_both_yes'] += 1
            else:
                s['lost_to_late_date'] += 1
        s['sec_missing'] += sum(e.get('missing_reason') == 'no_response' for e in es)
    return s


def run_episode(world, ask_fn, match_fn, days=60, follow=40, log=False):
    sim = Simulator(world)
    for d in range(days):
        sim.resolve_asks(ask_fn(sim.observe(), d))
        sim.advance(match_fn(sim.observe(), d))
    arrived = sum(m['arrived_day'] <= 59 for m in sim.members.values())
    for _ in range(follow):
        sim.advance([])
    s = decompose(sim)
    s['arrived'] = arrived
    s['msmi_per_100'] = 100 * s['msmi'] / max(1, arrived)
    s['coverage'] = sum(1 for i in sim.introductions for _ in (0, 1))
    served = {x for i in sim.introductions for x in (i['user_a'], i['user_b'])}
    s['coverage'] = len(served) / max(1, arrived)
    s['assign_per_100'] = 100 * s['assignments'] / max(1, arrived)
    s['ask_cost'] = sim.asks_total
    return s, sim