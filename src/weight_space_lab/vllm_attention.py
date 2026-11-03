"""Expose the pinned TP=1 vLLM attention matrices to the existing edit recipe.

The QKV slices are disjoint views of a fused parameter. No weights are copied or
changed by this adapter. The patcher retains originals and performs all edits.
"""
from .patch import PatchValidationError


class AttentionView:
    def __init__(self, model):
        self.model = model
        self.config = model.config
        self.is_quantized = getattr(model, "is_quantized", False)
        receipt = getattr(model, "_bu_merged_compat_receipt", {})
        if not receipt.get("complete"):
            raise PatchValidationError("complete merged-checkpoint load receipt required")
        parameters = dict(model.named_parameters())
        self.views = {}
        storage_ranges = []
        for layer in range(48):
            source = f"language_model.model.layers.{layer}.self_attn."
            target = f"model.language_model.layers.{layer}.self_attn."
            fused = parameters.get(source + "qkv_proj.weight")
            output = parameters.get(source + "o_proj.weight")
            if fused is None or output is None:
                raise PatchValidationError("missing pinned vLLM attention parameters")
            if tuple(fused.shape) != (5120, 2048) or tuple(output.shape) != (2048, 4096):
                raise PatchValidationError("only full unquantized TP=1 BU attention shapes are admitted")
            for matrix, stride in ((fused, (2048, 1)), (output, (4096, 1))):
                if (tuple(matrix.stride()) != stride or str(matrix.dtype) != "torch.bfloat16"
                        or matrix.device.type != "cuda" or matrix.element_size() != 2):
                    raise PatchValidationError("contiguous BF16 CUDA attention storage required")
                start = matrix.data_ptr()
                end = start + matrix.numel() * 2
                if any(str(matrix.device) == device and start < other_end and other_start < end
                       for device, other_start, other_end in storage_ranges):
                    raise PatchValidationError("full attention parameters overlap in storage")
                storage_ranges.append((str(matrix.device), start, end))
            self.views[target + "q_proj.weight"] = fused[0:4096, :]
            self.views[target + "k_proj.weight"] = fused[4096:4608, :]
            self.views[target + "v_proj.weight"] = fused[4608:5120, :]
            self.views[target + "o_proj.weight"] = output

    @property
    def training(self):
        return getattr(self.model, "training", True)

    def named_parameters(self):
        return self.views.items()
