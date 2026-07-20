"""Deterministic tests for the cadence-aware FX execution policy."""

from datetime import datetime, timedelta
import unittest

import pandas as pd

from fx_timing import evaluate_cadence_windows, execution_quality, plan_execution
from signals import build_signals


def probs(p_now: float) -> dict:
    return {
        "exchange_now": p_now,
        "split": 0.15,
        "wait": max(0.0, 0.85 - p_now),
        "composite": 0.0,
        "agreement": 0.7,
        "regime": 0.0,
    }


class ExecutionPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 7, 20, 12, 0, 0)

    def plan(
        self,
        p_now: float,
        elapsed_days: float | None,
        opportunity_score: float | None = None,
    ):
        anchor = (
            None
            if elapsed_days is None
            else self.now - timedelta(days=elapsed_days)
        )
        return plan_execution(
            probs(p_now),
            anchor,
            now=self.now,
            floor=0.25,
            ceiling=0.50,
            cadence_days=4,
            opportunity_score=opportunity_score,
        )

    def test_first_run_opens_an_initial_fill(self) -> None:
        plan = self.plan(0.31, None)
        self.assertTrue(plan.action_now)
        self.assertEqual(plan.trigger, "initial_fill")
        self.assertEqual(plan.decision, "exchange_now")

    def test_fresh_fill_preserves_optionality(self) -> None:
        plan = self.plan(0.31, 0.5)
        self.assertFalse(plan.action_now)
        self.assertEqual(plan.trigger, "cadence_building")
        self.assertAlmostEqual(plan.days_to_due, 3.5)

    def test_even_an_exceptional_signal_waits_until_day_three(self) -> None:
        plan = self.plan(0.90, 2.9)
        self.assertFalse(plan.action_now)
        self.assertEqual(plan.trigger, "cadence_building")

    def test_day_three_opportunity_uses_declining_hurdle(self) -> None:
        plan = self.plan(0.31, 3.0, opportunity_score=0.50)
        self.assertTrue(plan.action_now)
        self.assertEqual(plan.trigger, "opportunity_window")
        self.assertLess(plan.opportunity_threshold, 0.50)

    def test_day_three_can_still_wait_for_a_better_tick(self) -> None:
        plan = self.plan(0.31, 3.0, opportunity_score=0.30)
        self.assertFalse(plan.action_now)
        self.assertEqual(plan.trigger, "window_open")
        self.assertAlmostEqual(plan.days_to_due, 1.0)

    def test_day_four_never_waits_even_with_a_weak_signal(self) -> None:
        plan = self.plan(0.05, 4.0)
        self.assertTrue(plan.action_now)
        self.assertEqual(plan.trigger, "cadence_due")
        self.assertEqual(plan.days_to_due, 0.0)

    def test_tranche_size_stays_inside_configured_bounds(self) -> None:
        low = self.plan(0.05, 4.0)
        high = self.plan(0.95, 4.0)
        self.assertEqual(low.size_frac, 0.25)
        self.assertEqual(high.size_frac, 0.50)

    def test_hurdle_declines_monotonically_toward_due_date(self) -> None:
        hurdles = [self.plan(0.1, day).opportunity_threshold for day in (0, 1, 2, 3, 4)]
        self.assertEqual(hurdles, sorted(hurdles, reverse=True))
        self.assertAlmostEqual(hurdles[-1], 0.20)

    def test_execution_quality_rewards_a_better_local_usd_rate(self) -> None:
        index = pd.date_range("2026-06-01", periods=20, freq="B")
        rising = pd.Series([5.00 + i * 0.01 for i in range(20)], index=index)
        falling = rising.iloc[::-1].reset_index(drop=True)
        falling.index = index
        directional = probs(0.31)
        self.assertGreater(
            execution_quality(directional, rising),
            execution_quality(directional, falling),
        )


class WalkForwardSignalTests(unittest.TestCase):
    def test_empty_optional_history_is_skipped(self) -> None:
        index = pd.date_range("2022-01-01", periods=80, freq="B")
        base = pd.Series([5 + i * 0.001 for i in range(80)], index=index)
        data = {
            "usdbrl": base,
            "dxy": base * 20,
            "brent": base * 15,
            "vale": base * 3,
            "vix": base * 4,
            "ibov": base * 20_000,
            "six_l": pd.Series(dtype=float),
            "cot_eur": pd.Series(dtype=float),
        }
        signals, _ = build_signals(data)
        names = {signal.name for signal in signals}
        self.assertNotIn("BRL Futures (6L)", names)
        self.assertNotIn("CFTC EUR Position", names)

    def test_cadence_evaluation_is_strictly_walk_forward(self) -> None:
        index = pd.date_range("2022-01-03", periods=180, freq="B")
        base = pd.Series([5 + (i % 17) * 0.004 for i in range(180)], index=index)
        data = {
            "usdbrl": base,
            "dxy": base * 20,
            "brent": base * 15,
            "vale": base * 3,
            "vix": base * 4,
            "ibov": base * 20_000,
        }
        windows = evaluate_cadence_windows(data, None, cadence_days=4)
        self.assertGreater(len(windows), 10)
        self.assertTrue(
            all(
                window.selected_day in {window.day_three, window.day_four}
                for window in windows
            )
        )


if __name__ == "__main__":
    unittest.main()
