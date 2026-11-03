"""Source-reviewed loader bridge for the recovered BU merged checkpoint only.

This module does not load a model, create files, or alter checkpoint bytes on
import. install_single_process_loader() is an explicit, process-local hook for
vLLM 0.18.0 / Transformers 4.57.6. It is not a general checkpoint converter.
See evidence/vllm018-compatibility-research.json for the primary-source basis.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import re
from collections.abc import Iterable, Iterator
from typing import Any


CONFIG_SHA256 = "9743d159f38b383b746c54c9d286da79da71768e7997c73b37024da307ffcaa0"
INDEX_SHA256 = "1bb83a632abd9efef296e6394e35c705195f5b8d7cec394742753196b952aec0"
_EXPERT = re.compile(
    r"^(model\.language_model\.layers\.(\d+)\.mlp\.experts)\."
    r"(gate_up_proj|down_proj)$"
)
_EXPECTED_EXPERT_KEYS = {
    f"model.language_model.layers.{layer}.mlp.experts.{projection}"
    for layer in range(48) for projection in ("gate_up_proj", "down_proj")
}


def _pinned_json(raw: bytes, expected: str) -> dict:
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Checkpoint metadata differs from the admitted BU artifact")
    return json.loads(raw)


def normalize_config(raw_config: bytes) -> dict:
    """Return HF4 metadata representing the saved HF5 conditional model.

    The source version field remains historical provenance. The inner tying
    flag is explicitly false because the conditional model has an independent
    lm_head; vLLM constructs its language-model wrapper from text_config alone.
    """
    source = _pinned_json(raw_config, CONFIG_SHA256)
    result = copy.deepcopy(source)
    text = result["text_config"]
    if (source["architectures"] != ["Qwen3VLMoeForConditionalGeneration"]
            or source["tie_word_embeddings"] is not False
            or text["num_local_experts"] != 128):
        raise ValueError("Unexpected checkpoint architecture")
    text["num_experts"] = text.pop("num_local_experts")
    rope = text.pop("rope_parameters")
    if rope != {"mrope_interleaved": True, "mrope_section": [24, 20, 20],
                "rope_theta": 5000000, "rope_type": "default"}:
        raise ValueError("Unexpected rotary configuration")
    text["rope_theta"] = rope.pop("rope_theta")
    text["rope_scaling"] = rope
    text["tie_word_embeddings"] = False
    # This spelling is present in the cached original HF4 BU configuration.
    result["vision_config"]["model_type"] = "qwen3_vl_moe"
    return result


def admitted_weight_names(raw_index: bytes) -> frozenset[str]:
    index = _pinned_json(raw_index, INDEX_SHA256)
    names = frozenset(index["weight_map"])
    expert_names = {name for name in names if ".mlp.experts." in name}
    if expert_names != _EXPECTED_EXPERT_KEYS:
        raise ValueError("Expected exactly 48 pairs of packed expert tensors")
    if not {"lm_head.weight", "model.language_model.embed_tokens.weight"} <= names:
        raise ValueError("Independent output and embedding weights are required")
    return names


def split_expert_tensor(tensor: Any, projection: str, *, experts: int,
                        hidden: int, intermediate: int) -> Iterator[tuple[int, str, Any]]:
    """Return per-expert 2D views. No transpose, arithmetic, clone, or cast.

    Dimensions are explicit so tiny synthetic arrays can check the indexing
    without installing vLLM or loading a model. The production caller below
    fixes dimensions to the admitted checkpoint and requires BF16 tensors.
    """
    expected = ((experts, 2 * intermediate, hidden) if projection == "gate_up_proj"
                else (experts, hidden, intermediate))
    if projection not in ("gate_up_proj", "down_proj") or tuple(tensor.shape) != expected:
        raise ValueError(f"Unexpected {projection} shape: {tuple(tensor.shape)}")
    for expert in range(experts):
        if projection == "gate_up_proj":
            yield expert, "gate_proj", tensor[expert, :intermediate, :]
            yield expert, "up_proj", tensor[expert, intermediate:, :]
        else:
            yield expert, "down_proj", tensor[expert, :, :]


def unpack_checkpoint(weights: Iterable[tuple[str, Any]], expected_names: frozenset[str],
                      receipt: dict) -> Iterator[tuple[str, Any]]:
    """Adapt the complete raw HF iterator before vLLM's prefix mapper.

    The receipt becomes complete only after the consumer exhausts the iterator.
    A partial or failed load never admits inference. All source names, including
    nonexpert tensors, must match the pinned index exactly once.
    """
    if {n for n in expected_names if ".mlp.experts." in n} != _EXPECTED_EXPERT_KEYS:
        raise ValueError("Invalid expected checkpoint names")
    seen: set[str] = set()
    emitted_experts = 0
    receipt.update({"complete": False, "pid": os.getpid(), "source_tensors": 0,
                    "emitted_expert_tensors": 0})
    for name, tensor in weights:
        if name not in expected_names or name in seen:
            raise ValueError(f"Unexpected or duplicate checkpoint tensor: {name}")
        seen.add(name)
        match = _EXPERT.fullmatch(name)
        if match is None:
            yield name, tensor
            continue
        if str(tensor.dtype) != "torch.bfloat16":
            raise ValueError(f"Expected BF16 expert tensor: {name}")
        prefix, _, projection = match.groups()
        for expert, part, view in split_expert_tensor(
                tensor, projection, experts=128, hidden=2048, intermediate=768):
            emitted_experts += 1
            yield f"{prefix}.{expert}.{part}.weight", view
    if seen != expected_names or emitted_experts != 48 * 128 * 3:
        raise ValueError("Incomplete checkpoint iterator")
    receipt.update({"complete": True, "source_tensors": len(seen),
                    "emitted_expert_tensors": emitted_experts})


def install_single_process_loader(raw_index: bytes) -> dict:
    """Install before LLM construction, in a fresh dedicated container.

    Required runner settings: VLLM_ENABLE_V1_MULTIPROCESSING=0 set before vLLM
    import; distributed_executor_backend='uni'; TP/PP/DP=1; quantization=None.
    This hook changes only loading behavior. It does not launch inference.
    """
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise ValueError("The compatibility hook requires in-process execution")
    for package, expected in (("vllm", "0.18.0"), ("transformers", "4.57.6")):
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"Unreviewed {package} version")
    expected_names = admitted_weight_names(raw_index)
    from vllm.model_executor.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
    cls = Qwen3VLMoeForConditionalGeneration
    if getattr(cls, "_bu_merged_compat_installed", False):
        raise RuntimeError("Compatibility hook was already installed")
    original = cls.load_weights

    def load_weights(self, weights):
        text = self.config.text_config
        if (text.num_experts != 128 or text.hidden_size != 2048
                or text.moe_intermediate_size != 768 or text.num_hidden_layers != 48
                or text.tie_word_embeddings is not False):
            raise ValueError("Normalized BU configuration is required")
        from vllm.config import get_current_vllm_config
        config = get_current_vllm_config()
        parallel = config.parallel_config
        if (parallel.distributed_executor_backend != "uni"
                or parallel.tensor_parallel_size != 1
                or parallel.pipeline_parallel_size != 1
                or parallel.data_parallel_size != 1
                or config.quant_config is not None):
            raise ValueError("Only unquantized, single-process loading was reviewed")
        head = self.language_model.lm_head.weight
        embedding = self.language_model.model.embed_tokens.weight
        if head.data_ptr() == embedding.data_ptr():
            raise ValueError("Output and embedding weights were unexpectedly tied")
        receipt = {}
        self._bu_merged_compat_receipt = receipt
        loaded = original(self, unpack_checkpoint(weights, expected_names, receipt))
        if not receipt.get("complete"):
            raise RuntimeError("The loader did not consume the complete checkpoint")
        return loaded

    cls.load_weights = load_weights
    cls._bu_merged_compat_installed = True
    return {"schema": 1, "installed_pid": os.getpid(), "vllm": "0.18.0",
            "transformers": "4.57.6", "method": "packed_to_per_expert_views",
            "config_sha256": CONFIG_SHA256, "index_sha256": INDEX_SHA256,
            "expected_source_tensors": len(expected_names), "model_loaded": False}
