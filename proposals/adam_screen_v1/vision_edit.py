"""Exploratory visual-merger calibration, NOT a learned capability transfer.

Only the two final main-merger parameters of the pinned unquantized TP=1
Qwen3-VL runtime are touched. DeepStack, ViT blocks, experts and routers remain
unchanged. Every edit starts from retained originals. No imports or mutation of
a model occur on module import. The caller must serialize edits and inference.
"""
import math

TARGETS = {
    "visual.merger.linear_fc2.weight": (2048, 4608),
    "visual.merger.linear_fc2.bias": (2048,),
}
GAINS = (0.98, 1.0, 1.02)


def checked_gain(value):
    if type(value) not in (float, int) or value not in GAINS:
        raise ValueError("only frozen main-merger gains .98, 1, 1.02 are allowed")
    return float(value)


def _rebuild(original, gain, torch):
    """Elementwise FP32 scaling then original dtype; no cumulative arithmetic."""
    gain = checked_gain(gain)
    if gain == 1.0:
        return original.clone()
    result = original.to(dtype=torch.float32).mul_(gain).to(dtype=original.dtype)
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError("nonfinite visual edit")
    return result


class VisualMergerEditor:
    def __init__(self, model, *, torch_module=None):
        if torch_module is None:
            import torch as torch_module
        torch = self.torch = torch_module
        if (getattr(model, "training", True) is not False
                or getattr(model, "is_quantized", False)
                or getattr(getattr(model, "config", None), "quantization_config", None) not in (None, {})
                or not getattr(model, "_bu_merged_compat_receipt", {}).get("complete")):
            raise ValueError("complete unquantized eval model required")
        # Include duplicate names so tied parameters cannot hide an alias.
        parameters = dict(model.named_parameters(remove_duplicate=False))
        targets = {}
        ranges = []
        for name, shape in TARGETS.items():
            tensor = parameters.get(name)
            if (tensor is None or not torch.is_tensor(tensor) or tuple(tensor.shape) != shape
                    or str(tensor.dtype) != "torch.bfloat16" or tensor.device.type != "cuda"
                    or not tensor.is_contiguous() or tensor.element_size() != 2
                    or not bool(torch.isfinite(tensor).all().item())):
                raise ValueError("unexpected visual merger storage: " + name)
            start, end = tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * 2
            for device, low, high in ranges:
                if str(tensor.device) == device and start < high and low < end:
                    raise ValueError("overlapping visual targets")
            ranges.append((str(tensor.device), start, end))
            targets[name] = tensor
        # Reject aliasing with ANY other named parameter, including DeepStack.
        # The pinned model is dense BF16. Contiguous spans are required for all
        # parameters so a strange view cannot escape a simple range check.
        for name, tensor in parameters.items():
            if name in targets:
                continue
            if not tensor.is_contiguous():
                raise ValueError("unreviewed noncontiguous model parameter")
            low = tensor.data_ptr()
            high = low + tensor.numel() * tensor.element_size()
            if any(str(tensor.device) == device and low < end and start < high
                   for device, start, end in ranges):
                raise ValueError("visual target aliases another parameter: " + name)
        self.model, self.targets = model, targets
        with torch.no_grad():
            self.originals = {name: tensor.detach().to(device="cpu").clone()
                              for name, tensor in targets.items()}
        self.state, self.busy = "original", False

    def apply(self, gain):
        gain = checked_gain(gain)
        if self.busy or self.model.training or (self.state == "partial_or_failed" and gain != 1.0):
            raise ValueError("visual edit not admitted")
        self.busy, self.state = True, "partial_or_failed"
        torch, rows = self.torch, []
        try:
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
                for name, target in self.targets.items():
                    original = self.originals[name].to(device=target.device)
                    value = _rebuild(original, gain, torch)
                    delta = value.to(dtype=torch.float32) - original.to(dtype=torch.float32)
                    changed = int(torch.count_nonzero(value != original).item())
                    maximum = float(delta.abs().max().item())
                    squared = float(torch.sum(delta.square(), dtype=torch.float64).item())
                    if not math.isfinite(maximum) or not math.isfinite(squared):
                        raise ValueError("nonfinite visual delta statistics")
                    target.copy_(value)
                    if not bool(torch.equal(target, value)):
                        raise ValueError("visual edit copy readback failed")
                    rows.append({"name": name, "shape": list(target.shape),
                                 "elements": target.numel(), "changed_elements": changed,
                                 "max_abs_delta": maximum, "l2_delta": math.sqrt(squared)})
            self.state = "original" if gain == 1.0 else "edited"
        finally:
            self.busy = False
        return {"recipe": "main_visual_merger_output_gain_v1", "gain": gain,
                "formula": "BF16(FP32(original weight or bias) * gain)",
                "targets": rows, "rebuilt_from_original": True,
                "restoration_copy_equal": gain == 1.0,
                "cache_invalidation_required_before_inference": True,
                "effect": "unevaluated calibration, not a learned visual task delta"}


def invalidate_after_edit(engine):
    """For pinned vLLM0.18, clear ALL caches after a completed serialized edit.

    Resetting the multimodal processor cache alone does not clear encoded image
    features. Failure prevents the caller from admitting any next generation.
    Apply identically to original and edited variants. No concurrent request.
    """
    llm_engine = engine.llm_engine
    if llm_engine.has_unfinished_requests():
        raise ValueError("cannot invalidate caches with unfinished requests")
    if engine.reset_prefix_cache() is not True:
        raise RuntimeError("prefix cache reset did not report success")
    engine.reset_mm_cache()
    llm_engine.reset_encoder_cache()
    return {"prefix_reset_returned_true": True, "mm_processor_reset_returned": True,
            "encoder_reset_returned": True,
            "scope": "pinned source API calls returned; not an independent cache-memory audit"}
