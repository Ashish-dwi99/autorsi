"""Tiny CPU indexing and metadata tests; no models, vLLM, or provider calls."""
import json
import pathlib
import sys
import unittest

try:
    import numpy as np
except ImportError:
    np = None

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from weight_space_lab.vllm018_compat import (
    admitted_weight_names, normalize_config,
    split_expert_tensor, unpack_checkpoint,
)


class ExpertLayoutTest(unittest.TestCase):
    @unittest.skipIf(np is None, "NumPy is optional for tiny view tests")
    def test_gate_up_order_and_shared_storage(self):
        packed = np.arange(2 * 6 * 4).reshape(2, 6, 4)
        parts = list(split_expert_tensor(packed, "gate_up_proj", experts=2,
                                         hidden=4, intermediate=3))
        self.assertEqual([(e, n) for e, n, _ in parts],
                         [(0, "gate_proj"), (0, "up_proj"),
                          (1, "gate_proj"), (1, "up_proj")])
        for expert in range(2):
            np.testing.assert_array_equal(parts[2 * expert][2], packed[expert, :3])
            np.testing.assert_array_equal(parts[2 * expert + 1][2], packed[expert, 3:])
        for _, _, part in parts:
            self.assertTrue(np.shares_memory(packed, part))

    @unittest.skipIf(np is None, "NumPy is optional for tiny view tests")
    def test_down_order_without_transpose(self):
        packed = np.arange(2 * 4 * 3).reshape(2, 4, 3)
        parts = list(split_expert_tensor(packed, "down_proj", experts=2,
                                         hidden=4, intermediate=3))
        for expert, name, part in parts:
            self.assertEqual(name, "down_proj")
            np.testing.assert_array_equal(part, packed[expert])
            self.assertTrue(np.shares_memory(packed, part))

    @unittest.skipIf(np is None, "NumPy is optional for tiny view tests")
    def test_wrong_layout_rejected(self):
        with self.assertRaises(ValueError):
            list(split_expert_tensor(np.zeros((2, 4, 6)), "gate_up_proj",
                                     experts=2, hidden=4, intermediate=3))

    def test_unpinned_metadata_rejected(self):
        with self.assertRaises(ValueError):
            normalize_config(b"{}")
        with self.assertRaises(ValueError):
            admitted_weight_names(b"{}")


_PREFLIGHT = pathlib.Path(__file__).resolve().parents[1] / "evidence/private/preflight.json"


@unittest.skipUnless(_PREFLIGHT.is_file(), "Private saved metadata is not in this checkout")
class SavedMetadataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = json.loads(_PREFLIGHT.read_text())

    def test_saved_config_normalizes_to_original_hf4_semantics(self):
        # The metadata's original byte serialization is reproducible; the
        # production function still checks the exact historical SHA256.
        merged = self.saved["merged_config"]["json"]
        raw = (json.dumps(merged, indent=2) + "\n").encode()
        normalized = normalize_config(raw)
        original = self.saved["snapshots"][0]["config.json"]["json"]
        self.assertEqual(normalized["text_config"]["num_experts"], 128)
        self.assertIs(normalized["text_config"]["tie_word_embeddings"], False)
        # The explicit false equals the HF4 text-config default. Historical
        # producer version is not silently rewritten to the base version.
        normalized["text_config"].pop("tie_word_embeddings")
        self.assertIsNone(normalized["text_config"].pop("sliding_window"))
        normalized["transformers_version"] = original["transformers_version"]
        self.assertEqual(normalized, original)
        self.assertIn("num_local_experts", merged["text_config"])

    def test_incomplete_checkpoint_cannot_publish_complete(self):
        raw = (json.dumps(self.saved["merged_index"]["json"], indent=2) + "\n").encode()
        names = admitted_weight_names(raw)
        receipt = {}
        with self.assertRaises(ValueError):
            list(unpack_checkpoint([], names, receipt))
        self.assertIs(receipt["complete"], False)

    def test_duplicate_source_key_is_rejected(self):
        raw = (json.dumps(self.saved["merged_index"]["json"], indent=2) + "\n").encode()
        names = admitted_weight_names(raw)
        receipt = {}
        with self.assertRaises(ValueError):
            list(unpack_checkpoint([("lm_head.weight", object()),
                                    ("lm_head.weight", object())], names, receipt))
        self.assertIs(receipt["complete"], False)


if __name__ == "__main__":
    unittest.main()
