"""Pure tests: no torch/model imports, GPU use, browser, or provider calls."""

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "src" / "weight_space_lab" / "patch.py"
spec = importlib.util.spec_from_file_location("weight_space_patch_under_test", SOURCE)
patch = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = patch
spec.loader.exec_module(patch)


class FakeTensor:
    def __init__(self, rows, *, floating=True, dtype="float32", device="cpu"):
        self.rows = [list(row) for row in rows]
        self.floating = floating
        self.dtype = dtype
        self.device = device
        self.shape = (len(rows), len(rows[0]) if rows else 0)
        self.ndim = 2

    def is_floating_point(self):
        return self.floating

    def detach(self):
        return self

    def clone(self):
        return FakeTensor(self.rows, floating=self.floating, dtype=self.dtype, device=self.device)

    def mul(self, value):
        return FakeTensor([[x * value for x in row] for row in self.rows], dtype=self.dtype, device=self.device)


class Truth:
    def __init__(self, value):
        self.value = value

    def all(self):
        return self

    def item(self):
        return self.value


class FakeTorch:
    @staticmethod
    def is_tensor(value):
        return isinstance(value, FakeTensor)

    @staticmethod
    def isfinite(value):
        return Truth(all(math.isfinite(x) for row in value.rows for x in row))


def config():
    return {
        "base_model_name_or_path": patch.BASE_MODEL,
        "peft_type": "LORA", "r": 16, "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "use_dora": False, "use_rslora": False, "bias": "none",
    }


def tensors():
    result = {}
    for target in ("q_proj", "k_proj", "v_proj", "o_proj"):
        prefix = f"base_model.model.model.language_model.layers.0.self_attn.{target}"
        result[prefix + ".lora_A.weight"] = FakeTensor([[float(i + 1), -float(i)] for i in range(16)])
        result[prefix + ".lora_B.weight"] = FakeTensor([[float(i - 3) for i in range(16)] for _ in range(3)])
    return result


def multiply(b, a):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*a.rows)] for row in b.rows]


class PatchTests(unittest.TestCase):
    def test_import_does_not_import_torch(self):
        code = (
            "import importlib.util,sys; "
            f"s=importlib.util.spec_from_file_location('isolated_patch', {str(SOURCE)!r}); "
            "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m); "
            "assert 'torch' not in sys.modules"
        )
        subprocess.run([sys.executable, "-I", "-c", code], check=True, capture_output=True)

    def test_B_only_scaling_changes_delta_linearly_and_preserves_source(self):
        source = tensors()
        before = {key: copy.deepcopy(value.rows) for key, value in source.items()}
        for factor in (0.0, 0.5, 0.75, 1.0):
            output, metadata = patch.scale_lora_b(source, config(), factor, torch_module=FakeTorch)
            for pair in patch.discover_lora_pairs(source):
                old_delta = multiply(source[pair.b_key], source[pair.a_key])
                new_delta = multiply(output[pair.b_key], output[pair.a_key])
                self.assertEqual(new_delta, [[factor * x for x in row] for row in old_delta])
                self.assertEqual(output[pair.a_key].rows, before[pair.a_key])
                self.assertIsNot(output[pair.a_key], source[pair.a_key])
                self.assertIsNot(output[pair.b_key], source[pair.b_key])
            self.assertEqual(metadata["pair_count"], 4)
            self.assertEqual(metadata["baseline"], factor == 1.0)
            self.assertEqual(metadata["existing_merged_checkpoint_equivalence"], "NOT_ESTABLISHED")
        self.assertEqual({key: value.rows for key, value in source.items()}, before)

    def test_baseline_has_no_tensor_arithmetic(self):
        original = FakeTensor.mul
        def forbidden(*args):
            raise AssertionError("baseline must only clone")
        FakeTensor.mul = forbidden
        try:
            patch.scale_lora_b(tensors(), config(), 1.0, torch_module=FakeTorch)
        finally:
            FakeTensor.mul = original

    def test_rejects_wrong_base_and_nonstandard_scaling(self):
        edits = (
            {"base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct"},
            {"r": True}, {"r": 8}, {"lora_alpha": 16}, {"use_dora": True},
            {"use_rslora": True}, {"use_qalora": True}, {"lora_bias": True},
            {"fan_in_fan_out": True}, {"bias": "all"},
            {"rank_pattern": {"x": 8}}, {"alpha_pattern": {"x": 16}},
            {"modules_to_save": ["lm_head"]}, {"quantization_config": {"bits": 4}},
            {"load_in_4bit": True}, {"load_in_8bit": True},
            {"loftq_config": {"loftq_bits": 4}},
            {"target_modules": ["q_proj", "q_proj", "v_proj", "o_proj"]},
        )
        for change in edits:
            with self.subTest(change=change), self.assertRaises(patch.PatchValidationError):
                patch.validate_adapter_config(config() | change)
        for factor in (-0.5, 0.1, 1.5, True, "0.5", float("nan"), float("inf")):
            with self.subTest(factor=factor), self.assertRaises(patch.PatchValidationError):
                patch.validate_scale(factor)

    def test_no_key_can_be_silently_dropped(self):
        source = tensors()
        key = next(iter(source))
        malformed = (
            {k: v for k, v in source.items() if k != key},
            source | {"base_model.model.lm_head.weight": FakeTensor([[1.0]])},
            {k.replace(".lora_A.weight", ".lora_A.default.weight"): v for k, v in source.items()},
            {k: v for k, v in source.items() if ".o_proj." not in k},
        )
        for mapping in malformed:
            with self.assertRaises(patch.PatchValidationError):
                patch.discover_lora_pairs(mapping)

    def test_bad_matrices_refuse_before_source_mutation(self):
        for replacement in (
            FakeTensor([[1.0, 2.0]]),
            FakeTensor([[float("nan"), 2.0] for _ in range(16)]),
            FakeTensor([[float("inf"), 2.0] for _ in range(16)]),
            FakeTensor([[1, 2] for _ in range(16)], floating=False),
            FakeTensor([[1.0, 2.0] for _ in range(16)], dtype="float16"),
            FakeTensor([[1.0, 2.0] for _ in range(16)], dtype="float8_e4m3fn"),
            FakeTensor([[1.0, 2.0] for _ in range(16)], device="cuda"),
        ):
            source = tensors()
            key = next(k for k in source if ".lora_A." in k)
            source[key] = replacement
            with self.assertRaises(patch.PatchValidationError):
                patch.scale_lora_b(source, config(), 0.5, torch_module=FakeTorch)
            self.assertIs(source[key], replacement)

    def test_provenance_hashes_actual_separate_artifacts_and_marks_equivalence_unknown(self):
        _, metadata = patch.scale_lora_b(tensors(), config(), 0.5, torch_module=FakeTorch)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, cfg = root / "source", root / "patched", root / "adapter_config.json"
            source.write_bytes(b"original serialized adapter")
            output.write_bytes(b"separate patched adapter")
            cfg.write_text(json.dumps(config()))
            record = patch.build_patch_provenance(source, output, cfg, metadata, existing_merged_reference="bu-merged/bu-30b-ft")
            self.assertEqual(record["source_adapter"]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(record["patched_adapter"]["sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(record["existing_merged_checkpoint_equivalence"], "NOT_ESTABLISHED")
            self.assertIsNone(record["base_model"]["revision"])
            with self.assertRaises(patch.PatchValidationError):
                patch.build_patch_provenance(source, source, cfg, metadata)
            hardlink = root / "alias"
            hardlink.hardlink_to(source)
            with self.assertRaises(patch.PatchValidationError):
                patch.build_patch_provenance(source, hardlink, cfg, metadata)
            with self.assertRaises(patch.PatchValidationError):
                patch.file_sha256(source, max_bytes=2)
            cfg.write_text(json.dumps(config() | {"lora_dropout": 0.1}))
            with self.assertRaises(patch.PatchValidationError):
                patch.build_patch_provenance(source, output, cfg, metadata)


if __name__ == "__main__":
    unittest.main()
