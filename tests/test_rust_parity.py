"""Differential tests: the Rust core (bsr_core) against the pure-Python
reference engine.  Random + curated positions must produce the same best
action and the same win probability / tiebreak to 1e-9, and run_ai must
emit character-identical plans through both engines.

Skipped entirely when the bsr_core wheel is not installed.
"""

import random
import time
import unittest

import ai_engine
from ai_engine import (
    State,
    _best_action,
    _init_worlds,
    _rust_state_args,
    run_ai,
)

bsr = ai_engine._bsr

TOL = 1e-9


def items(**kw):
    order = ["glass", "pills", "phone", "cuffs", "adr", "saw", "ciggs", "beer", "inv"]
    return tuple(kw.get(k, 0) for k in order)


def make_state(php, ehp, pitems, eitems, shells, live, blank, max_hp=4,
               saw=False, e_cuff=0, p_cuff=0, mem=None):
    worlds = _init_worlds(list(shells), live, blank)
    assert worlds, "inconsistent test scenario"
    if mem is None:
        mem = (False,) * len(shells)
    return State(php, ehp, max_hp, tuple(pitems), tuple(eitems), worlds,
                 tuple(mem), saw, e_cuff, p_cuff)


def random_position(rng):
    """A small root-like position (exactly solvable by both engines)."""
    max_hp = rng.choice([4, 4, 4, 6])
    php = rng.randint(1, max_hp)
    ehp = rng.randint(1, max_hp)
    n = rng.randint(1, 5)
    live = rng.randint(0, n)
    blank = n - live
    seq = [1] * live + [2] * blank
    rng.shuffle(seq)
    shells = [sh if rng.random() < 0.35 else 0 for sh in seq]
    pitems = [0] * 9
    eitems = [0] * 9
    for _ in range(rng.randint(0, 4)):
        pitems[rng.randrange(9)] += 1
    for _ in range(rng.randint(0, 4)):
        eitems[rng.randrange(9)] += 1
    saw = rng.random() < 0.15
    e_cuff = rng.choice([0, 0, 0, 1, 2])
    mem = tuple(rng.random() < 0.15 for _ in range(n))
    return make_state(php, ehp, pitems, eitems, shells, live, blank,
                      max_hp=max_hp, saw=saw, e_cuff=e_cuff, mem=mem), \
        (shells, live, blank)


@unittest.skipIf(bsr is None, "bsr_core wheel not installed")
class RustParityTests(unittest.TestCase):
    def assert_root_matches(self, state):
        act, idx, val = _best_action(state, {}, None)
        hit = bsr.Solver().root_search(*_rust_state_args(state))
        self.assertIsNotNone(hit)
        ract, ridx, rp, rtb, rhorizon = hit
        self.assertIsNone(rhorizon, "small position must be solved exactly")
        self.assertEqual((ract, ridx), (act, idx))
        self.assertAlmostEqual(rp, val[0], delta=TOL)
        self.assertAlmostEqual(rtb, val[1], delta=TOL)

    def test_random_positions_same_action_and_probability(self):
        rng = random.Random(20260707)
        for i in range(150):
            state, _ = random_position(rng)
            with self.subTest(position=i):
                self.assert_root_matches(state)

    def test_curated_positions(self):
        cases = [
            # inverter converts a known blank into a guaranteed win
            make_state(2, 1, items(inv=1), items(), [2, 0, 0], 1, 2),
            # confirmed live, dealer at 1 HP: shoot, keep the toys
            make_state(4, 1, items(saw=1, inv=1, cuffs=1), items(), [1], 1, 0),
            # cuff cooldown blocks re-cuffing
            make_state(4, 4, items(cuffs=1), items(), [1, 1], 2, 0, e_cuff=2),
            # dealer cuffed right now
            make_state(3, 3, items(beer=1), items(saw=1), [0, 0], 1, 1, e_cuff=1),
            # stale-target / beer-heavy dealer pool
            make_state(2, 3, items(glass=1), items(beer=2, pills=1), [0, 0, 0], 2, 1),
            # adrenaline steal decisions
            make_state(3, 2, items(adr=1), items(saw=1, cuffs=1, ciggs=1), [1, 0], 1, 1),
            # burner-phone 8-shell remap (result 7 -> 6)
            make_state(4, 4, items(phone=1), items(), [0] * 8, 4, 4),
            # dealer phone memory subtraction
            make_state(3, 3, items(), items(phone=2, glass=1), [0, 0, 0, 0], 2, 2),
            # saw already active, pills at 1 HP forbidden for the dealer
            make_state(1, 2, items(pills=1), items(pills=1), [0, 0], 1, 1, saw=True),
        ]
        for i, state in enumerate(cases):
            with self.subTest(case=i):
                self.assert_root_matches(state)

    def test_full_plans_identical(self):
        rng = random.Random(9137)
        inputs = []
        for _ in range(40):
            state, (shells, live, blank) = random_position(rng)
            inputs.append((state.php, state.ehp, list(state.pitems),
                           list(state.eitems), shells, live, blank,
                           state.max_hp, False, state.saw, state.e_cuff))
        for i, args in enumerate(inputs):
            with self.subTest(case=i):
                plan_rust = run_ai(*args[:8], saw_active=args[9],
                                   cuff_state=args[10])
                try:
                    ai_engine._bsr = None
                    plan_py = run_ai(*args[:8], saw_active=args[9],
                                     cuff_state=args[10])
                finally:
                    ai_engine._bsr = bsr
                self.assertEqual(plan_rust, plan_py)

    def test_heavy_midround_exact_and_fast(self):
        # 4 unknown shells + 8v8 items: ~10 s exact in Python, must be
        # exact and well under a second in Rust.
        state = make_state(
            4, 4,
            items(glass=1, pills=1, phone=1, cuffs=1, adr=1, saw=1, beer=1, inv=1),
            items(glass=1, pills=1, phone=1, cuffs=1, adr=1, saw=1, ciggs=1, beer=1),
            [0] * 4, 2, 2, max_hp=6,
        )
        t0 = time.perf_counter()
        hit = bsr.Solver().root_search(*_rust_state_args(state))
        elapsed = time.perf_counter() - t0
        self.assertIsNotNone(hit)
        _act, _idx, _p, _tb, horizon = hit
        self.assertIsNone(horizon, "heavy midround must still be exact")
        self.assertLess(elapsed, 5.0)


if __name__ == "__main__":
    unittest.main()
