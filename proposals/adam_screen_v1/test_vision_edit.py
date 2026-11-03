"""Tiny numerical/cache-contract tests; no torch/model/provider invocation."""
import math
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from test_merged_patch import FakeTorch, Tensor
from vision_edit import _rebuild, checked_gain, invalidate_after_edit, VisualMergerEditor


class VisionEditTests(unittest.TestCase):
    def test_gain_sequence_rebuilds_original_and_restore_preserves_signed_zero(self):
        original = Tensor([[1.0, -2.0], [-0.0, 0.125]])
        before = [struct.pack("!d", x) for row in original.rows for x in row]
        for gain in (0.98, 1.02, 0.98):
            edited = _rebuild(original, gain, FakeTorch())
            self.assertEqual(edited.rows, [[gain, -2 * gain], [-0.0, .125 * gain]])
            self.assertEqual([struct.pack("!d", x) for row in original.rows for x in row], before)
        restored = _rebuild(original, 1, FakeTorch())
        self.assertIsNot(restored, original)
        self.assertEqual([struct.pack("!d", x) for row in restored.rows for x in row], before)

    def test_no_arbitrary_or_nonfinite_calibration(self):
        for value in (True, False, "1", float("nan"), float("inf"), 0, .5, .980001, 2):
            with self.subTest(value=value), self.assertRaises(ValueError):
                checked_gain(value)
        with self.assertRaises(ValueError):
            _rebuild(Tensor([[math.inf]]), .98, FakeTorch())

    def test_refuses_partial_or_untrained_model_before_storage(self):
        for model in (SimpleNamespace(training=True), SimpleNamespace(training=False),
                      SimpleNamespace(training=False, is_quantized=True)):
            with self.assertRaises(ValueError):
                VisualMergerEditor(model, torch_module=FakeTorch())

    def test_cache_barrier_order(self):
        calls = []
        class Engine:
            def __init__(self): self.llm_engine = self
            def has_unfinished_requests(self): calls.append("pending"); return False
            def reset_prefix_cache(self): calls.append("prefix"); return True
            def reset_mm_cache(self): calls.append("processor")
            def reset_encoder_cache(self): calls.append("encoder")
        receipt = invalidate_after_edit(Engine())
        self.assertEqual(calls, ["pending", "prefix", "processor", "encoder"])
        self.assertTrue(receipt["encoder_reset_returned"])

    def test_pending_or_failed_prefix_never_reaches_later_cache_calls(self):
        for pending, reset in ((True, True), (False, False), (False, None)):
            calls = []
            engine = SimpleNamespace()
            engine.llm_engine = SimpleNamespace(has_unfinished_requests=lambda: pending,
                reset_encoder_cache=lambda: calls.append("encoder"))
            engine.reset_prefix_cache = lambda: reset
            engine.reset_mm_cache = lambda: calls.append("processor")
            with self.assertRaises((ValueError, RuntimeError)):
                invalidate_after_edit(engine)
            self.assertEqual(calls, [])

    def test_encoder_failure_is_not_returned_as_success(self):
        def fail(): raise RuntimeError("cache reset failed")
        engine = SimpleNamespace(reset_prefix_cache=lambda: True, reset_mm_cache=lambda: None,
            llm_engine=SimpleNamespace(has_unfinished_requests=lambda: False, reset_encoder_cache=fail))
        with self.assertRaisesRegex(RuntimeError, "cache reset failed"):
            invalidate_after_edit(engine)


if __name__ == "__main__":
    unittest.main()
