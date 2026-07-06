import time
import unittest

from ai_engine import (
    ADRENALINE,
    BEER,
    BLANK,
    CIGGS,
    CUFF_COOLDOWN,
    CUFF_NONE,
    CUFFS,
    INVERTER,
    LIVE,
    PHONE,
    PILLS,
    SAW,
    State,
    _apply_player,
    _best_action,
    _dealer_eligible,
    _dealer_figures_out,
    _init_worlds,
    _player_actions,
    _probs,
    _weighted_flip,
    run_ai,
)


def items(**kw):
    order = ["glass", "pills", "phone", "cuffs", "adr", "saw", "ciggs", "beer", "inv"]
    return tuple(kw.get(k, 0) for k in order)


def make_state(php, ehp, pitems, eitems, shells, live, blank, max_hp=4,
               saw=False, e_cuff=CUFF_NONE, p_cuff=CUFF_NONE):
    worlds = _init_worlds(list(shells), live, blank)
    assert worlds, "inconsistent test scenario"
    return State(php, ehp, max_hp, tuple(pitems), tuple(eitems), worlds,
                 (False,) * len(shells), saw, e_cuff, p_cuff)


class BeliefTests(unittest.TestCase):
    def test_future_shell_knowledge_constrains_current_probability(self):
        worlds = _init_worlds([0, 1, 0], 1, 2)
        self.assertEqual(worlds, ((2, 1, 2),))
        s = make_state(3, 3, items(), items(), [0, 1, 0], 1, 2)
        self.assertEqual(_probs(s), (0.0, 1.0))

    def test_inverter_flips_exact_belief(self):
        s = make_state(3, 3, items(inv=1), items(), [0, 0, 0], 2, 1)
        outcomes = _apply_player(s, "item", INVERTER)
        self.assertEqual(len(outcomes), 1)
        _, ns, phase = outcomes[0]
        self.assertEqual(phase, "player")
        p_live, p_blank = _probs(ns)
        self.assertAlmostEqual(p_live, 1 / 3)
        self.assertAlmostEqual(p_blank, 2 / 3)

    def test_phone_creates_information_branches(self):
        s = make_state(1, 3, items(phone=1), items(), [0, 0, 0], 1, 2)
        outcomes = _apply_player(s, "item", PHONE)
        self.assertAlmostEqual(sum(p for p, _, _ in outcomes), 1.0)
        self.assertTrue(all(phase == "player" for _, _, phase in outcomes))
        self.assertTrue(any(len(ns.worlds) < len(s.worlds) for _, ns, _ in outcomes))

    def test_full_load_phone_never_reveals_last_shell(self):
        # BurnerPhone.gd remaps result 7 to 6 on a full 8-shell load: the
        # last shell can never be revealed, position 7 is twice as likely.
        s = make_state(4, 4, items(phone=1), items(), [0] * 8, 4, 4)
        outcomes = _apply_player(s, "item", PHONE)
        for _, ns, _ in outcomes:
            first = ns.worlds[0][7]
            self.assertFalse(
                len(ns.worlds) > 1 and all(w[7] == first for w in ns.worlds),
                "last shell of an 8-load must stay unknown",
            )


class RulesTests(unittest.TestCase):
    def test_no_double_handcuffs(self):
        # Real rule: the dealer cannot be cuffed twice in a row — after the
        # skipped turn re-cuffing stays blocked until he plays a real turn.
        plan = run_ai(4, 3, list(items(cuffs=2)), list(items()),
                      [1, 1, 1], 3, 0, 4)
        cuff_steps = [t for t in plan if t.startswith("Use HANDCUFFS")]
        self.assertLessEqual(len(cuff_steps), 1)

    def test_cuffs_blocked_on_cooldown(self):
        s = make_state(4, 4, items(cuffs=1), items(), [1, 1], 2, 0,
                       e_cuff=CUFF_COOLDOWN)
        acts = _player_actions(s)
        self.assertNotIn(("item", CUFFS), acts)

    def test_beer_on_last_shell_triggers_reload(self):
        s = make_state(4, 4, items(beer=1), items(), [1], 1, 0)
        outcomes = _apply_player(s, "item", BEER)
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0][2], "reload")

    def test_self_shot_pruned_on_confirmed_live(self):
        s = make_state(4, 4, items(), items(), [1, 0], 2, 0)
        self.assertNotIn(("shoot_self", -1), _player_actions(s))

    def test_pills_can_kill_player(self):
        s = make_state(1, 4, items(pills=1), items(), [0, 0], 1, 1)
        outcomes = _apply_player(s, "item", PILLS)
        phases = sorted(phase for _, _, phase in outcomes)
        self.assertEqual(phases, ["loss", "player"])


class GuaranteedWinTests(unittest.TestCase):
    def test_inverter_win_recognized_as_guaranteed(self):
        # Dealer at 1 HP, current shell KNOWN BLANK, inverter in hand:
        # the engine must see a proven 100% win, never a gamble.
        s = make_state(2, 1, items(inv=1), items(), [2, 0, 0], 1, 2)
        _act, _idx, (p, _tb) = _best_action(s, {}, None)
        self.assertAlmostEqual(p, 1.0)
        plan = run_ai(2, 1, list(items(inv=1)), list(items()),
                      [2, 0, 0], 1, 2, 4)
        self.assertTrue(any("INVERTER" in t for t in plan))

    def test_guaranteed_win_prefers_saving_items(self):
        # Confirmed live shell, dealer at 1 HP: just shoot — burning the
        # saw/inverter first wins too but wastes items.
        plan = run_ai(4, 1, list(items(saw=1, inv=1, cuffs=1)), list(items()),
                      [1], 1, 0, 4)
        self.assertEqual(len(plan), 1)
        self.assertTrue(plan[0].startswith("Shoot DEALER — confirmed LIVE"))


class DealerPolicyTests(unittest.TestCase):
    def test_weighted_coin_flip_matches_endless_mode(self):
        # DealerIntelligence.CoinFlip(): more lives -> shoot player (1),
        # more blanks -> shoot self (0), equal -> true 50/50.
        self.assertEqual(_weighted_flip(((1, 1, 2),)), [(1.0, 1)])
        self.assertEqual(_weighted_flip(((2, 2, 1),)), [(1.0, 0)])
        self.assertEqual(_weighted_flip(((1, 2),)), [(0.5, 0), (0.5, 1)])

    def test_figure_out_shell_inference(self):
        # Knows via phone memory of the current shell.
        self.assertTrue(_dealer_figures_out((1, 2), (True, False)))
        # Knows when only one type remains.
        self.assertTrue(_dealer_figures_out((1, 1, 1), (False,) * 3))
        # Knows by subtracting phone-known shells: (live, blank, live) with
        # the blank at index 1 known -> the unknown rest is all live.
        self.assertTrue(_dealer_figures_out((1, 2, 1), (False, True, False)))
        # Genuinely unknown.
        self.assertFalse(_dealer_figures_out((1, 2, 2), (False,) * 3))

    def test_dealer_medicine_rules(self):
        # DealerIntelligence.gd:165-169: expired medicine requires hp below
        # max, no cigarettes anywhere in his pool, not already used this
        # turn, and NEVER at exactly 1 HP.
        def eligible_pills(ehp, eitems_, used_med=False):
            s = make_state(4, ehp, items(), eitems_, [0, 0, 0], 1, 2)
            elig = _dealer_eligible(s, s.eitems, False, 0, used_med)
            return any(idx == PILLS for idx, _ in elig)

        self.assertFalse(eligible_pills(1, items(pills=1)))
        self.assertTrue(eligible_pills(2, items(pills=1)))
        self.assertTrue(eligible_pills(3, items(pills=1)))
        self.assertFalse(eligible_pills(4, items(pills=1)))          # full HP
        self.assertFalse(eligible_pills(2, items(pills=1, ciggs=1)))  # cigs first
        self.assertFalse(eligible_pills(2, items(pills=1), used_med=True))

    def test_dealer_saw_only_on_known_live(self):
        s = make_state(4, 4, items(), items(saw=1), [0, 0, 0], 1, 2)
        self.assertFalse(_dealer_eligible(s, s.eitems, False, 0, False))
        self.assertTrue(
            any(idx == SAW for idx, _ in
                _dealer_eligible(s, s.eitems, True, LIVE, False))
        )
        self.assertFalse(_dealer_eligible(s, s.eitems, True, BLANK, False))


class ApiTests(unittest.TestCase):
    def test_input_validation(self):
        self.assertEqual(run_ai(0, 3, [0] * 9, [0] * 9, [0], 1, 0)[0],
                         "You are dead — game over")
        self.assertEqual(run_ai(3, 0, [0] * 9, [0] * 9, [0], 1, 0)[0],
                         "Dealer HP is 0 — round should already be over")
        self.assertEqual(run_ai(3, 3, [0] * 9, [0] * 9, [], 0, 0)[0],
                         "No shells loaded — update shell count first")
        self.assertEqual(run_ai(3, 3, [0] * 9, [0] * 9, [1, 2], 2, 1)[0],
                         "Shell knowledge is inconsistent with LIVE / BLANK counts")

    def test_plan_starts_with_an_action(self):
        # Win-chance display was removed on purpose — the first task line
        # must be an actionable step, not a header.
        plan = run_ai(4, 4, [0] * 9, [0] * 9, [0, 0, 0], 1, 2, 4)
        self.assertTrue(plan[0].startswith(("Shoot", "Use")))


class PerformanceTests(unittest.TestCase):
    def test_heavy_midround_position_is_fast_and_exact(self):
        # The old engine needed 210 seconds here (4 unknown shells, 8v8
        # items) and could not be interrupted.
        t0 = time.perf_counter()
        plan = run_ai(
            4, 4,
            list(items(glass=1, pills=1, phone=1, cuffs=1, adr=1, saw=1,
                       beer=1, inv=1)),
            list(items(glass=1, pills=1, phone=1, cuffs=1, adr=1, saw=1,
                       ciggs=1, beer=1)),
            [0] * 4, 2, 2, 6,
        )
        elapsed = time.perf_counter() - t0
        self.assertTrue(plan)
        self.assertLess(elapsed, 30.0)

    def test_worst_case_full_load_uses_bounded_fallback(self):
        # Fresh 8-shell load with both inventories full: too large for an
        # exact solve, must finish quickly via the honest depth-limited
        # fallback instead of eating minutes and gigabytes.
        t0 = time.perf_counter()
        plan = run_ai(
            4, 4,
            list(items(glass=1, pills=1, phone=1, cuffs=1, adr=1, saw=1,
                       beer=1, inv=1)),
            list(items(glass=1, pills=1, phone=1, cuffs=1, adr=1, saw=1,
                       ciggs=1, beer=1)),
            [0] * 8, 4, 4, 6,
        )
        elapsed = time.perf_counter() - t0
        self.assertTrue(plan)
        self.assertLess(elapsed, 90.0)


if __name__ == "__main__":
    unittest.main()
