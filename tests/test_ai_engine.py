import unittest

from ai_engine import (
    ADRENALINE,
    INVERTER,
    PHONE,
    State,
    _init_worlds,
    _probs,
    _use_item,
    run_ai,
)


class AiEngineTests(unittest.TestCase):
    def test_future_shell_knowledge_constrains_current_probability(self):
        worlds = _init_worlds([0, 1, 0], 1, 2)
        state = State(3, 3, 4, [0] * 9, [0] * 9, worlds)
        self.assertEqual(worlds, ((2, 1, 2),))
        self.assertEqual(_probs(state), (0.0, 1.0))

    def test_phone_creates_information_branches(self):
        worlds = _init_worlds([0, 0, 0], 1, 2)
        state = State(1, 3, 4, [0, 0, 1, 0, 0, 0, 0, 0, 0], [0] * 9, worlds)
        outcomes = _use_item(state, PHONE, True)

        self.assertEqual(len(outcomes), 4)
        self.assertAlmostEqual(sum(prob for prob, _ in outcomes), 1.0)
        self.assertTrue(any(len(ns.worlds) == 1 for _, ns in outcomes))
        self.assertTrue(any(len(ns.worlds) == 2 for _, ns in outcomes))

    def test_unknown_inverter_updates_exact_belief(self):
        worlds = _init_worlds([0, 0, 0], 2, 1)
        state = State(3, 3, 4, [0, 0, 0, 0, 0, 0, 0, 0, 1], [0] * 9, worlds)
        outcomes = _use_item(state, INVERTER, True)

        self.assertEqual(len(outcomes), 1)
        _, next_state = outcomes[0]
        p_live, p_blank = _probs(next_state)
        self.assertAlmostEqual(p_live, 1 / 3)
        self.assertAlmostEqual(p_blank, 2 / 3)

    def test_run_ai_can_choose_phone_when_information_is_best(self):
        result = run_ai(
            1, 3,
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0] * 9,
            [0, 0, 0], 2, 1, 4,
        )
        self.assertIn("Use PHONE", result[0])
        self.assertIn("Mark the revealed future shell", result[1])

    def test_adrenaline_text_names_the_stolen_item(self):
        result = run_ai(
            1, 3,
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0], 1, 2, 4,
        )
        self.assertIn("Use ADRENALINE", result[0])
        self.assertIn("steal dealer's CIGARETTES", result[0])

    def test_two_shell_control_scenario_does_not_default_to_shoot_dealer(self):
        result = run_ai(
            4, 4,
            [0, 0, 1, 1, 0, 2, 0, 1, 0],
            [0, 0, 0, 1, 2, 0, 1, 0, 1],
            [0, 0], 1, 1, 4,
        )
        self.assertNotIn("Shoot DEALER", result[0])
        self.assertTrue(
            result[0].startswith("Use PHONE")
            or result[0].startswith("Use HANDCUFFS")
            or result[0].startswith("Use HACKSAW")
            or result[0].startswith("Use BEER")
        )


if __name__ == "__main__":
    unittest.main()
