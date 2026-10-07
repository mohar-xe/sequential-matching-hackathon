"""Unit tests for team_policy.py."""
import json
import unittest

from kit import generate, Simulator, eligibility, HARD
import team_policy


class TestTeamPolicy(unittest.TestCase):
    def test_mwm_on_odd_cycle(self):
        """Test general graph matching (triangle / non-bipartite graph)."""
        # Triangle with edges: (A, B)=1.0, (B, C)=0.9, (A, C)=0.8
        edges = [
            (1.0, ('A', 'B')),
            (0.9, ('B', 'C')),
            (0.8, ('A', 'C')),
        ]
        matched = team_policy.max_weight_matching(edges)
        self.assertEqual(len(matched), 1)
        self.assertEqual(sorted(matched[0]), ['A', 'B'])

    def test_mwm_beats_greedy(self):
        """Classic matching vs greedy case: A-B=0.9, C-D=0.05, A-D=0.65, B-C=0.65.
        Greedy picks A-B (0.9) -> sum = 0.95.
        MWM picks A-D + B-C -> sum = 1.30."""
        edges = [
            (0.90, ('A', 'B')),
            (0.65, ('A', 'D')),
            (0.65, ('B', 'C')),
            (0.05, ('C', 'D')),
        ]
        matched = team_policy.max_weight_matching(edges)
        self.assertEqual(len(matched), 2)
        pairs_sorted = sorted([sorted(p) for p in matched])
        self.assertEqual(pairs_sorted, [['A', 'D'], ['B', 'C']])

    def test_ask_budget_and_validity(self):
        """Asks must never exceed 12 budget units, and never query unmatchable members."""
        world = generate(101, 200, 'evaluation', 'development')
        sim = Simulator(world)
        st = sim.observe()
        asks = team_policy.compute_asks(st, team_policy.BASE_W)

        # Cost check
        cost = sum(3 if a['field'] == 'constraints' else 1 for a in asks)
        self.assertLessEqual(cost, 12)

        # No duplicate asks
        seen = set()
        for a in asks:
            pair = (a['member_id'], a['field'])
            self.assertNotIn(pair, seen)
            seen.add(pair)

        # Members must be available and matchable
        mem_by_id = {m['member_id']: m for m in st['members']}
        for a in asks:
            m = mem_by_id[a['member_id']]
            self.assertTrue(m['available'])
            self.assertFalse(any(m['field_status'][k] == 'declined' for k in HARD))

    def test_match_eligibility(self):
        """Matched pairs must all be feasible, available, and disjoint."""
        world = generate(101, 200, 'evaluation', 'development')
        sim = Simulator(world)
        # Advance a few days with asks
        for d in range(5):
            st = sim.observe()
            sim.resolve_asks(team_policy.compute_asks(st, team_policy.BASE_W))
            st = sim.observe()
            pairs = team_policy.compute_matches(st, team_policy.BASE_W)
            sim.advance(pairs)

            used = set()
            mem_by_id = {m['member_id']: m for m in st['members']}
            for u, v in pairs:
                self.assertNotIn(u, used)
                self.assertNotIn(v, used)
                used.add(u)
                used.add(v)
                self.assertTrue(mem_by_id[u]['available'])
                self.assertTrue(mem_by_id[v]['available'])
                self.assertEqual(eligibility(mem_by_id[u], mem_by_id[v])['status'], 'feasible')

    def test_decide_json_protocol(self):
        """Test full decide() protocol with JSON serialization."""
        req_ask = {
            'schema_version': '1.0.0',
            'phase': 'ask',
            'state': {
                'schema_version': '1.0.0',
                'synthetic': True,
                'day': 0,
                'ask_budget_remaining': 12,
                'members': [],
                'introductions': [],
                'feedback': [],
                'ask_log': []
            },
            'memory': None
        }
        res_ask = team_policy.decide(req_ask, mode='team')
        self.assertIn('asks', res_ask)
        self.assertIn('memory', res_ask)
        self.assertIsInstance(res_ask['asks'], list)

        req_match = {
            'schema_version': '1.0.0',
            'phase': 'match',
            'state': req_ask['state'],
            'memory': res_ask['memory']
        }
        res_match = team_policy.decide(req_match, mode='team')
        self.assertIn('pairs', res_match)
        self.assertIn('memory', res_match)
        self.assertIsInstance(res_match['pairs'], list)

        # JSON serialize without nan/inf
        raw = json.dumps(res_match, allow_nan=False)
        self.assertTrue(len(raw) > 0)


if __name__ == '__main__':
    unittest.main()
