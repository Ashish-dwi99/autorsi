"""Pure matrix tests: no torch import, model load, GPU, browser or provider."""

from contextlib import nullcontext
import copy
import math
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from weight_space_lab.merged_patch import MergedWeightEditor, merged_weight_key
from weight_space_lab.patch import BASE_MODEL, PatchValidationError


class Scalar:
    def __init__(self, value):
        self.value = value

    def all(self):
        return self

    def item(self):
        return self.value


class Device:
    def __init__(self, name):
        self.name = str(name)
        self.type = self.name.split(":")[0]

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return str(self) == str(other)


class Tensor:
    layout = "strided"
    ndim = 2
    is_quantized = False

    def __init__(self, rows, dtype="bfloat16", device="cpu"):
        self.rows = [list(row) for row in rows]
        self.dtype = dtype
        self.device = Device(device)
        self.shape = (len(rows), len(rows[0]))

    def is_floating_point(self):
        return self.dtype in {"bfloat16", "float32", "float64"}

    def data_ptr(self):
        return id(self)

    def detach(self):
        return self

    def clone(self):
        return Tensor(self.rows, self.dtype, self.device)

    def to(self, *, dtype=None, device=None):
        return Tensor(self.rows, dtype or self.dtype, device or self.device)

    def copy_(self, other):
        self.rows = copy.deepcopy(other.rows)
        return self

    def mul_(self, value):
        self.rows = [[x * value for x in row] for row in self.rows]
        return self

    def add_(self, other):
        self.rows = [[x + y for x, y in zip(a, b)] for a, b in zip(self.rows, other.rows)]
        return self

    def __sub__(self, other):
        return Tensor([[x - y for x, y in zip(a, b)] for a, b in zip(self.rows, other.rows)], self.dtype, self.device)

    def __ne__(self, other):
        return Tensor([[x != y for x, y in zip(a, b)] for a, b in zip(self.rows, other.rows)])

    def abs(self):
        return Tensor([[abs(x) for x in row] for row in self.rows])

    def square(self):
        return Tensor([[x * x for x in row] for row in self.rows])

    def max(self):
        return Scalar(max(x for row in self.rows for x in row))

    def numel(self):
        return self.shape[0] * self.shape[1]


class FakeTorch:
    float32 = "float32"
    float64 = "float64"

    def __init__(self):
        self.precision = "medium"
        self.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)))
        self.matmul_calls = 0
        self.fail_after = None

    is_tensor = staticmethod(lambda x: isinstance(x, Tensor))
    isfinite = staticmethod(lambda x: Scalar(all(math.isfinite(v) for row in x.rows for v in row)))
    equal = staticmethod(lambda a, b: a.rows == b.rows)
    count_nonzero = staticmethod(lambda x: Scalar(sum(v != 0 for row in x.rows for v in row)))
    sum = staticmethod(lambda x, dtype=None: Scalar(sum(v for row in x.rows for v in row)))
    no_grad = staticmethod(nullcontext)

    def get_float32_matmul_precision(self):
        return self.precision

    def set_float32_matmul_precision(self, precision):
        self.precision = precision

    def autocast(self, *, device_type, enabled):
        assert device_type == "cuda" and enabled is False
        return nullcontext()

    def matmul(self, b, a):
        assert self.precision == "highest" and self.backends.cuda.matmul.allow_tf32 is False
        assert a.dtype == b.dtype == "float32" and a.device.type == b.device.type == "cuda"
        self.matmul_calls += 1
        if self.fail_after is not None and self.matmul_calls >= self.fail_after:
            raise RuntimeError("injected incomplete edit")
        return Tensor([[sum(x * y for x, y in zip(row, column)) for column in zip(*a.rows)] for row in b.rows], "float32", "cuda:0")


def inputs():
    adapter, weights = {}, {}
    for layer in range(48):
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            module = f"base_model.model.model.language_model.layers.{layer}.self_attn.{projection}"
            adapter[module + ".lora_A.weight"] = Tensor([[1.0, -1.0] for _ in range(16)])
            adapter[module + ".lora_B.weight"] = Tensor([[0.25] * 16, [0.5] * 16])
            weights[merged_weight_key(module)] = Tensor([[10.0, -0.0], [20.0, 30.0]], device="cuda:0")
    model = SimpleNamespace(training=False, is_quantized=False, config=SimpleNamespace(), named_parameters=lambda: weights.items())
    config = {"base_model_name_or_path": BASE_MODEL, "peft_type": "LORA", "r": 16,
              "lora_alpha": 32, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}
    return model, adapter, config, weights


def bytes_of(tensor):
    return b"".join(struct.pack("!d", x) for row in tensor.rows for x in row)


class MergedPatchTests(unittest.TestCase):
    def test_no_torch_import(self):
        src = str(Path(__file__).resolve().parents[1] / "src")
        subprocess.run([sys.executable, "-I", "-c", f"import sys;sys.path.insert(0,{src!r});import weight_space_lab.merged_patch;assert 'torch' not in sys.modules"], check=True, capture_output=True)

    def test_edit_is_relative_to_original_and_restore_preserves_bytes(self):
        model, adapter, config, weights = inputs()
        original = {key: bytes_of(tensor) for key, tensor in weights.items()}
        adapter_before = {key: bytes_of(tensor) for key, tensor in adapter.items()}
        torch = FakeTorch()
        editor = MergedWeightEditor(model, adapter, config, torch_module=torch)
        for factor in (0.5, 0.75, 0.0, 0.75):
            result = editor.apply(factor)
            expected = [[10.0 + (factor - 1) * 8, (factor - 1) * -8],
                        [20.0 + (factor - 1) * 16, 30.0 + (factor - 1) * -16]]
            for weight in weights.values():
                self.assertEqual(weight.rows, expected)
            self.assertEqual(result["pair_count"], 192)
            self.assertEqual(result["totals"]["changed_elements"], 768)
            self.assertAlmostEqual(result["totals"]["max_abs_delta"], abs(factor - 1) * 16)
            self.assertAlmostEqual(result["totals"]["l2_delta"], math.sqrt(192 * (factor - 1) ** 2 * 640))
            self.assertEqual(result["zero_scale_is_original_base"], "NOT_ESTABLISHED")
            self.assertEqual(torch.precision, "medium")
            self.assertIs(torch.backends.cuda.matmul.allow_tf32, True)
        count = torch.matmul_calls
        result = editor.apply(1)
        self.assertEqual(torch.matmul_calls, count)
        self.assertTrue(result["baseline_copy_equality_verified"])
        self.assertEqual(result["totals"]["changed_elements"], 0)
        self.assertEqual({key: bytes_of(tensor) for key, tensor in weights.items()}, original)
        self.assertEqual({key: bytes_of(tensor) for key, tensor in adapter.items()}, adapter_before)

    def test_backups_do_not_alias_caller_adapter_or_weights(self):
        model, adapter, config, weights = inputs()
        editor = MergedWeightEditor(model, adapter, config, torch_module=FakeTorch())
        for tensor in adapter.values():
            tensor.rows[0][0] = 999
        for tensor in weights.values():
            tensor.rows[0][0] = 999
        editor.apply(0.5)
        self.assertTrue(all(tensor.rows == [[6.0, 4.0], [12.0, 38.0]] for tensor in weights.values()))

    def test_validation_precedes_any_weight_mutation(self):
        for defect in ("missing_layer", "extra_layer", "wrong_mapping", "wrong_shape", "nonfinite", "wrong_dtype", "wrong_device", "quantized", "training"):
            with self.subTest(defect=defect):
                model, adapter, config, weights = inputs()
                if defect == "missing_layer":
                    adapter = {k: v for k, v in adapter.items() if ".layers.47." not in k}
                elif defect == "extra_layer":
                    adapter = {k.replace(".layers.47.", ".layers.48."): v for k, v in adapter.items()}
                elif defect == "wrong_mapping":
                    weights.pop(next(iter(weights)))
                elif defect == "wrong_shape":
                    weights[next(iter(weights))] = Tensor([[1.0]], device="cuda:0")
                elif defect == "nonfinite":
                    next(iter(adapter.values())).rows[0][0] = float("inf")
                elif defect == "wrong_dtype":
                    next(iter(weights.values())).dtype = "float32"
                elif defect == "wrong_device":
                    next(iter(weights.values())).device = Device("cpu")
                elif defect == "quantized":
                    model.is_quantized = True
                elif defect == "training":
                    model.training = True
                before = {key: bytes_of(tensor) for key, tensor in weights.items()}
                with self.assertRaises(PatchValidationError):
                    MergedWeightEditor(model, adapter, config, torch_module=FakeTorch())
                self.assertEqual({key: bytes_of(tensor) for key, tensor in weights.items()}, before)

    def test_failed_edit_blocks_next_variant_but_allows_explicit_restore(self):
        model, adapter, config, weights = inputs()
        original = {key: bytes_of(tensor) for key, tensor in weights.items()}
        torch = FakeTorch()
        editor = MergedWeightEditor(model, adapter, config, torch_module=torch)
        torch.fail_after = 2
        with self.assertRaises(RuntimeError):
            editor.apply(0.5)
        self.assertEqual(torch.precision, "medium")
        self.assertIs(torch.backends.cuda.matmul.allow_tf32, True)
        with self.assertRaises(PatchValidationError):
            editor.apply(0.75)
        editor.apply(1)
        self.assertEqual({key: bytes_of(tensor) for key, tensor in weights.items()}, original)

    def test_scale_and_reentry_refuse_before_edit(self):
        model, adapter, config, weights = inputs()
        editor = MergedWeightEditor(model, adapter, config, torch_module=FakeTorch())
        original = {key: bytes_of(tensor) for key, tensor in weights.items()}
        for scale in (True, -1, 0.3, float("nan")):
            with self.assertRaises(PatchValidationError):
                editor.apply(scale)
        editor._busy = True
        with self.assertRaises(PatchValidationError):
            editor.apply(1)
        self.assertEqual({key: bytes_of(tensor) for key, tensor in weights.items()}, original)


if __name__ == "__main__":
    unittest.main()
