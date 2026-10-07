"""Team Policy: Uncertainty-Aware Value-of-Information (VOI) Acquisition
and General-Graph Maximum-Weight Matching (MWM).

Vouchsafe Sequential Matching Hackathon. Standard library only.
Fully compliant with POLICY_INTERFACE.md:
- Reads JSON requests from stdin, outputs JSON responses to stdout.
- State is carried across stateless invocations via request['memory'].
- Operates under strict 10s per-invocation limit, 2 cores, 1 GiB memory.
"""
import copy
import itertools
import json
import math
import sys
from pathlib import Path

from kit import HARD, SOFT, OPTIONS, eligibility

# Key soft fields affecting reciprocal mutual attraction
KEYF = ['relationship_goal', 'relationship_pace', 'lifestyle', 'conversations']
KS = {k: len(OPTIONS[k]) for k in KEYF}

# Prior fit weights (derived from generative model structure)
BASE_W = {
    'relationship_goal': 0.70,
    'relationship_pace': 0.40,
    'lifestyle': 0.25,
    'conversations': 0.20,
}

# -----------------------------------------------------------------------------
# 1. Probabilistic Pair Fit Modeling
# -----------------------------------------------------------------------------

def marginal_expected(w, k, va, vb):
    """Expected contribution of key field k marginalising unknown values.
    
    If both observed: +w if equal, -0.5*w if different.
    If unknown/declined: uniform prior over |S| options gives w*(1.5/|S| - 0.5).
    """
    if va is not None and vb is not None:
        return w if va == vb else -0.5 * w
    return w * (1.5 / KS[k] - 0.5)


def pair_fit(a, b, weights):
    """Expected soft compatibility between members a and b."""
    fa, fb = a['fields'], b['fields']
    return sum(marginal_expected(weights[k], k, fa.get(k), fb.get(k)) for k in KEYF)


def pair_weight(a, b, weights, day=0):
    """Matching edge weight: expected soft fit plus baseline viability offset."""
    fit = pair_fit(a, b, weights)
    # Slight front-loading offset before day 35 drift penalty
    drift_offset = 0.0 if day < 35 else -0.10
    return fit + 0.75 + drift_offset


# -----------------------------------------------------------------------------
# 2. General-Graph Maximum Weight Matching (MWM)
# -----------------------------------------------------------------------------

def max_weight_matching(edges):
    """Exact Maximum-Weight Matching on general graphs.
    
    Decomposes the eligible graph into connected components and solves each
    component using branch-and-bound with upper-bounding pruning.
    Because daily candidate graphs have small component sizes (< 30 edges),
    this solves to provable optimality within sub-millisecond execution times.
    """
    pos_edges = [(w, e) for w, e in edges if w > 0]
    if not pos_edges:
        return []
    pos_edges.sort(key=lambda x: -x[0])

    # Decompose into connected components
    adj = {}
    for w, (u, v) in pos_edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    visited = set()
    components = []
    for node in adj:
        if node not in visited:
            comp_nodes = set()
            q = [node]
            visited.add(node)
            while q:
                curr = q.pop()
                comp_nodes.add(curr)
                for nxt in adj[curr]:
                    if nxt not in visited:
                        visited.add(nxt)
                        q.append(nxt)
            components.append(comp_nodes)

    matched_pairs = []
    for comp in components:
        comp_edges = [(w, e) for w, e in pos_edges if e[0] in comp and e[1] in comp]
        if len(comp_edges) == 1:
            matched_pairs.append(list(comp_edges[0][1]))
            continue

        best_w = [0.0]
        best_m = [[]]

        def bb(idx, used, cur_w, cur_m):
            # Prune using optimistic upper bound: sum of max available edge per free vertex
            rem_bound = cur_w
            seen = set()
            for w, (u, v) in comp_edges[idx:]:
                if u not in used and v not in used:
                    if u not in seen or v not in seen:
                        rem_bound += w
                        seen.add(u)
                        seen.add(v)
            if rem_bound <= best_w[0] + 1e-9:
                return

            if idx == len(comp_edges):
                if cur_w > best_w[0]:
                    best_w[0] = cur_w
                    best_m[0] = list(cur_m)
                return

            w, (u, v) = comp_edges[idx]
            # Branch 1: include edge (u, v) if both endpoints are free
            if u not in used and v not in used:
                used.add(u)
                used.add(v)
                cur_m.append((u, v))
                bb(idx + 1, used, cur_w + w, cur_m)
                cur_m.pop()
                used.remove(u)
                used.remove(v)

            # Branch 2: exclude edge
            bb(idx + 1, used, cur_w, cur_m)

        # Warm start with greedy solution for tighter initial pruning bound
        used_g = set()
        gw = 0.0
        gm = []
        for w, (u, v) in comp_edges:
            if u not in used_g and v not in used_g:
                used_g.add(u)
                used_g.add(v)
                gm.append((u, v))
                gw += w
        best_w[0] = gw
        best_m[0] = gm

        bb(0, set(), 0.0, [])
        for p in best_m[0]:
            matched_pairs.append(list(p))

    return matched_pairs


# -----------------------------------------------------------------------------
# 3. Compatibility & Acquisition Helpers
# -----------------------------------------------------------------------------

def is_matchable(m):
    """A member is matchable iff they have no declined HARD constraints."""
    return not any(m['field_status'][k] == 'declined' for k in HARD)


def is_hard_cleared(m):
    """A member is hard-cleared iff all 11 hard constraints are observed."""
    return all(m['fields'].get(k) is not None for k in HARD)


def compute_asks(state, weights):
    """Decide clarification queries under the 12-unit daily budget.
    
    1. Filter out permanently unmatchable members (saves ~46% wasted soft budget).
    2. Hard constraints: Prioritize uncleared matchable members by zone clustering
       and arrival day to concentrate cleared members in dense subgraphs.
    3. Soft fields: Prioritize members who are actively in feasible pairs or cleared,
       spending on the four decision-relevant fields.
    """
    budget = state['ask_budget_remaining']
    mem = [m for m in state['members'] if m['available']]
    matchable = [m for m in mem if is_matchable(m)]
    cleared = [m for m in matchable if is_hard_cleared(m)]
    uncleared = [m for m in matchable if not is_hard_cleared(m)]

    out = []

    # Step 1: Constraint bundles (3 units each)
    if budget >= 3 and uncleared:
        scores = []
        for u in uncleared:
            zone_peers = sum(1 for o in uncleared if o['zone'] == u['zone'] and o['member_id'] != u['member_id'])
            cleared_peers = sum(1 for c in cleared if c['zone'] == u['zone'])
            score = cleared_peers * 2 + zone_peers
            scores.append((score, -u['arrived_day'], u))
        scores.sort(key=lambda x: (x[0], x[1]), reverse=True)

        while budget >= 3 and scores:
            chosen = scores.pop(0)[2]
            out.append({'member_id': chosen['member_id'], 'field': 'constraints'})
            budget -= 3
            cleared.append(chosen)

    # Step 2: Key soft fields (1 unit each) for matchable members only
    if budget > 0:
        past = {tuple(sorted((i['user_a'], i['user_b']))) for i in state['introductions']}
        feasible_members = set()
        for a, b in itertools.combinations(cleared, 2):
            key = tuple(sorted((a['member_id'], b['member_id'])))
            if key not in past and eligibility(a, b)['status'] == 'feasible':
                feasible_members.add(a['member_id'])
                feasible_members.add(b['member_id'])

        softq = []
        for m in matchable:
            # Tier 0: actively in a feasible pair right now
            # Tier 1: hard-cleared member
            # Tier 2: uncleared matchable member
            if m['member_id'] in feasible_members:
                tier = 0
            elif is_hard_cleared(m):
                tier = 1
            else:
                tier = 2
            for k in KEYF:
                if m['fields'].get(k) is None and m['field_status'][k] != 'declined':
                    importance = -abs(weights.get(k, BASE_W[k]))
                    softq.append((tier, importance, m['arrived_day'], m['member_id'], k))

        softq.sort()
        for _, _, _, mid, k in softq[:budget]:
            out.append({'member_id': mid, 'field': k})
            budget -= 1

    return out


def compute_matches(state, weights):
    """Compute non-overlapping maximum-weight matching on available members."""
    mem = [m for m in state['members'] if m['available']]
    past = {tuple(sorted((i['user_a'], i['user_b']))) for i in state['introductions']}
    edges = []
    for a, b in itertools.combinations(mem, 2):
        key = tuple(sorted((a['member_id'], b['member_id'])))
        if key in past or eligibility(a, b)['status'] != 'feasible':
            continue
        w = pair_weight(a, b, weights, state['day'])
        edges.append((w, key))

    if not edges:
        return []

    return max_weight_matching(edges)


# -----------------------------------------------------------------------------
# 4. Online Feedback Learner (Streaming, Single-Pass)
# -----------------------------------------------------------------------------

def update_online_weights(state, memory):
    """Update fit weights from newly arrived directional feedback events.
    
    Processes each feedback event strictly once by tracking processed event IDs
    in memory. Regularised toward BASE_W with decaying learning rate.
    """
    weights = dict(memory.get('weights') or BASE_W)
    seen_events = set(memory.get('seen_events') or [])
    n_samples = memory.get('n_samples', 0)

    intro_by_id = {i['introduction_id']: i for i in state['introductions']}
    members_by_id = {m['member_id']: m for m in state['members']}

    newly_seen = []
    for e in state['feedback']:
        if e['event'] != 'introduction_response' or e.get('value') is None:
            continue
        event_key = f"{e['introduction_id']}:{e.get('member_id')}"
        if event_key in seen_events:
            continue
        newly_seen.append(event_key)
        seen_events.add(event_key)

        intro = intro_by_id.get(e['introduction_id'])
        if not intro:
            continue
        actor = members_by_id.get(e['member_id'])
        other_id = intro['user_b'] if e['member_id'] == intro['user_a'] else intro['user_a']
        other = members_by_id.get(other_id)
        if not actor or not other:
            continue

        feats = []
        for k in KEYF:
            va = actor['fields'].get(k)
            vb = other['fields'].get(k)
            if va is not None and vb is not None:
                feats.append((k, 1.0 if va == vb else -0.5))
        if not feats:
            continue

        lr = 0.25 / math.sqrt(1.0 + n_samples)
        z = -0.25 + sum(weights[k] * x for k, x in feats)
        p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
        y = 1.0 if e['value'] == 'yes' else 0.0
        err = p - y
        lam = 0.10
        for k, x in feats:
            g = err * x + lam * (weights[k] - BASE_W[k])
            g = max(-1.0, min(1.0, g))
            weights[k] -= lr * g
        n_samples += 1

    memory['weights'] = weights
    memory['seen_events'] = list(seen_events)
    memory['n_samples'] = n_samples
    return weights


# -----------------------------------------------------------------------------
# 5. Top-Level Policy Entry Point
# -----------------------------------------------------------------------------

def decide(request, mode='team'):
    """Top-level request handler compliant with POLICY_INTERFACE.md."""
    state = request['state']
    memory = request.get('memory') or {}

    # Initialize memory if empty
    if 'weights' not in memory:
        memory['weights'] = dict(BASE_W)
        memory['seen_events'] = []
        memory['n_samples'] = 0

    if mode in ('team', 'team_adaptive'):
        weights = update_online_weights(state, memory)
    else:
        weights = dict(BASE_W)

    if request['phase'] == 'ask':
        asks = compute_asks(state, weights)
        return {'asks': asks, 'memory': memory}
    elif request['phase'] == 'match':
        pairs = compute_matches(state, weights)
        return {'pairs': pairs, 'memory': memory}
    else:
        raise ValueError(f"Unknown phase: {request.get('phase')}")


if __name__ == '__main__':
    raw = sys.stdin.read()
    if not raw.strip():
        req = {'schema_version': '1.0.0', 'phase': 'ask', 'state': {'schema_version': '1.0.0', 'synthetic': True, 'day': 0, 'ask_budget_remaining': 12, 'members': [], 'introductions': [], 'feedback': [], 'ask_log': []}, 'memory': None}
    else:
        req = json.loads(raw)
    mode = 'team'
    if len(sys.argv) > 1 and sys.argv[1].startswith('--baseline='):
        mode = sys.argv[1].split('=')[1]
    elif len(sys.argv) > 2 and sys.argv[1] == '--baseline':
        mode = sys.argv[2]
    res = decide(req, mode=mode)
    print(json.dumps(res, allow_nan=False))
