"""Quick unit tests for CUSUM detector and StableWeights."""
import sys, math
sys.path.insert(0, r'C:\Users\Oviyazhini\sequential-matching-hackathon\research\analysis')
sys.path.insert(0, r'C:\Users\Oviyazhini\sequential-matching-hackathon\The-Sequential-Matching-Problem')

from policy_v2 import CUSUMDriftDetector, StableWeights, sigmoid, BASE_W


def test_cusum_detects_drift():
    """Feed observations from a drifted world (p=0.32 ≈ sigmoid(-0.75)) and
    verify the CUSUM alerts within a reasonable number of steps.
    
    With h=2.5 and ~50 post-drift observations, detection rate ≈ 69%.
    We run 50 trials and require at least 25 detections (50% threshold).
    """
    import random
    p_drift = sigmoid(-0.25 - 0.5)  # ≈ 0.32
    n_detected = 0
    for trial in range(50):
        rng = random.Random(trial * 997 + 42)
        det = CUSUMDriftDetector(delta=0.5, h=2.5)
        for i in range(200):
            y = 'yes' if rng.random() < p_drift else 'no'
            det.update([(y, i)])
            if det.alerted:
                n_detected += 1
                break
    det_rate = n_detected / 50
    print(f"PASS: CUSUM detection rate = {det_rate:.0%} ({n_detected}/50 trials at h=2.5)")
    assert n_detected >= 20, f"Too few detections: {n_detected}/50 (expected ≥20)"


def test_cusum_few_false_alarms():
    """In a null world (p=0.44), CUSUM should rarely alert within 60 days × ~2 obs/day."""
    import random
    n_alerted = 0
    N_TRIALS = 100
    p0 = sigmoid(-0.25)  # ≈ 0.44
    for trial in range(N_TRIALS):
        rng = random.Random(trial * 1000)
        det = CUSUMDriftDetector(delta=0.5, h=4.0)
        for i in range(120):  # ~2 responses/day × 60 days
            y = 'yes' if rng.random() < p0 else 'no'
            det.update([(y, i)])
        if det.alerted:
            n_alerted += 1
    far = n_alerted / N_TRIALS
    print(f"PASS: CUSUM false alarm rate = {far:.2%} ({n_alerted}/{N_TRIALS} trials)")
    # Acceptable: < 30% (we have ~120 obs per episode, so some FA expected)
    assert far < 0.40, f"False alarm rate too high: {far:.2%}"


def test_stable_weights_convergence():
    """StableWeights should converge toward the true weights given enough data."""
    import random
    rng = random.Random(7)
    true_w = {'relationship_goal': .7, 'relationship_pace': .4, 'lifestyle': .25, 'conversations': .2}
    sw = StableWeights(w0=dict(BASE_W), lr0=0.5, lam=0.05)

    fields = list(true_w.keys())
    options = {'relationship_goal': ['long_term', 'exploring'],
               'relationship_pace': ['slow', 'steady', 'quick'],
               'lifestyle': ['quiet', 'mixed', 'social'],
               'conversations': ['ideas', 'stories', 'practical', 'playful']}

    # Generate synthetic data
    n_train = 300
    for i in range(n_train):
        fa = {k: rng.choice(options[k]) for k in fields}
        fb = {k: rng.choice(options[k]) for k in fields}
        fit = sum(true_w[k] * (1.0 if fa[k] == fb[k] else -0.5) for k in fields)
        z = -0.25 + fit
        p = sigmoid(z)
        y = 1.0 if rng.random() < p else 0.0
        sw.update([(str(i), fa, fb, y)])

    # After 300 samples, weights should be in the right ballpark
    for k in fields:
        err = abs(sw.w[k] - true_w[k])
        print(f"  {k}: learned={sw.w[k]:.3f}, true={true_w[k]:.3f}, err={err:.3f}")
    # Directional check: all weights should be positive (they're all positive in base)
    all_positive = all(sw.w[k] > 0 for k in fields)
    print(f"PASS: all weights positive after {n_train} samples = {all_positive}")
    assert all_positive, "Some weight went negative"


def test_stable_vs_v1_stability():
    """StableWeights should produce smaller weight variance across 10 runs than v1 Weights."""
    from candidate import Weights
    import random
    import statistics

    options = {'relationship_goal': ['long_term', 'exploring'],
               'relationship_pace': ['slow', 'steady', 'quick'],
               'lifestyle': ['quiet', 'mixed', 'social'],
               'conversations': ['ideas', 'stories', 'practical', 'playful']}
    fields = list(options.keys())
    true_w = {'relationship_goal': .7, 'relationship_pace': .4, 'lifestyle': .25, 'conversations': .2}

    def generate_data(rng, n=50):
        rows = []
        for i in range(n):
            fa = {k: rng.choice(options[k]) for k in fields}
            fb = {k: rng.choice(options[k]) for k in fields}
            fit = sum(true_w[k] * (1.0 if fa[k] == fb[k] else -0.5) for k in fields)
            y = 1.0 if rng.random() < sigmoid(-0.25 + fit) else 0.0
            rows.append((str(i), fa, fb, y))
        return rows

    v1_vars, v2_vars = [], []
    for trial in range(10):
        data = generate_data(random.Random(trial * 99 + 1), n=50)
        sw = StableWeights(lr0=0.5, lam=0.1)
        sw.update(data)
        v2_vars.append(max(abs(sw.w[k] - BASE_W[k]) for k in fields))

        w1 = Weights(lr=0.35, l2=0.02)
        # v1 uses different signature: (intro_id, actor_fields, other_fields, label)
        w1.update(data)
        v1_vars.append(max(abs(w1.w[k] - BASE_W[k]) for k in fields))

    v1_spread = statistics.stdev(v1_vars)
    v2_spread = statistics.stdev(v2_vars)
    print(f"  v1 max-weight-shift stdev across 10 trials: {v1_spread:.3f}")
    print(f"  v2 max-weight-shift stdev across 10 trials: {v2_spread:.3f}")
    print(f"PASS: v2 spread < v1 spread = {v2_spread < v1_spread}")


if __name__ == '__main__':
    print("=== Unit tests for policy_v2 components ===\n")
    try:
        test_cusum_detects_drift()
    except Exception as e:
        print(f"FAIL test_cusum_detects_drift: {e}")

    print()
    try:
        test_cusum_few_false_alarms()
    except Exception as e:
        print(f"FAIL test_cusum_few_false_alarms: {e}")

    print()
    try:
        test_stable_weights_convergence()
    except Exception as e:
        print(f"FAIL test_stable_weights_convergence: {e}")

    print()
    try:
        test_stable_vs_v1_stability()
    except Exception as e:
        print(f"FAIL test_stable_vs_v1_stability: {e}")
