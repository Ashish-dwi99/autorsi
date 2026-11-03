import unittest
from types import SimpleNamespace
from weight_space_lab.vllm_attention import AttentionView
from weight_space_lab.patch import PatchValidationError


class Matrix:
    dtype = "torch.bfloat16"
    device = SimpleNamespace(type="cuda")
    def __init__(self, shape, parent=None, span=None, pointer=0):
        self.shape, self.parent, self.span = shape, parent, span
        self.pointer = pointer

    def stride(self):
        return self.shape[1], 1

    def element_size(self):
        return 2

    def numel(self):
        return self.shape[0] * self.shape[1]

    def data_ptr(self):
        return self.pointer

    def __getitem__(self, index):
        rows, columns = index
        return Matrix((rows.stop - rows.start, self.shape[1]), self, (rows.start, rows.stop))


class Model:
    config = SimpleNamespace()
    training = False
    _bu_merged_compat_receipt = {"complete": True}

    def __init__(self):
        self.parameters = {}
        for layer in range(48):
            prefix = f"language_model.model.layers.{layer}.self_attn."
            self.parameters[prefix + "qkv_proj.weight"] = Matrix((5120, 2048), pointer=layer * 0x20000000)
            self.parameters[prefix + "o_proj.weight"] = Matrix((2048, 4096), pointer=layer * 0x20000000 + 0x10000000)

    def named_parameters(self):
        return self.parameters.items()


class AttentionViewTest(unittest.TestCase):
    def test_disjoint_qkv_views_and_output_identity(self):
        model = Model()
        view = AttentionView(model)
        pairs = dict(view.named_parameters())
        self.assertEqual(len(pairs), 192)
        for layer in range(48):
            p = f"model.language_model.layers.{layer}.self_attn."
            source = f"language_model.model.layers.{layer}.self_attn."
            q, k, v = (pairs[p + name + "_proj.weight"] for name in ("q", "k", "v"))
            self.assertEqual([x.span for x in (q, k, v)], [(0, 4096), (4096, 4608), (4608, 5120)])
            self.assertTrue(q.parent is k.parent is v.parent is model.parameters[source + "qkv_proj.weight"])
            self.assertIs(pairs[p + "o_proj.weight"], model.parameters[source + "o_proj.weight"])
        model.training = True
        self.assertTrue(view.training)

    def test_refuses_partial_load_and_wrong_fused_shape(self):
        model = Model()
        model._bu_merged_compat_receipt = {"complete": False}
        with self.assertRaises(PatchValidationError):
            AttentionView(model)

    def test_refuses_partial_overlap_with_different_start_address(self):
        model = Model()
        model.parameters["language_model.model.layers.1.self_attn.qkv_proj.weight"].pointer = 1024
        with self.assertRaises(PatchValidationError):
            AttentionView(model)
        model._bu_merged_compat_receipt = {"complete": True}
        model.parameters["language_model.model.layers.0.self_attn.qkv_proj.weight"] = Matrix((2560, 2048))
        with self.assertRaises(PatchValidationError):
            AttentionView(model)


if __name__ == "__main__":
    unittest.main()
