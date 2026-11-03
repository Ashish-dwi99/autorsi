"""The same screenshot/template encoding in CPU admission and GPU evaluation."""
import base64
import hashlib
import io
import json
from pathlib import Path


def encode_messages(processor, messages):
    from PIL import Image
    converted, images = [], []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            converted.append(message)
            continue
        parts = []
        for item in content:
            if item["type"] == "text":
                parts.append(item)
            elif item["type"] == "image_url":
                url = item["image_url"]["url"]
                if not url.startswith("data:image/png;base64,"):
                    raise ValueError("Only fixture screenshot data URLs are accepted")
                image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
                images.append(image)
                parts.append({"type": "image"})
            else:
                raise ValueError("Unknown message content")
        converted.append({"role": message["role"], "content": parts})
    text = processor.apply_chat_template(converted, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt")
    if inputs["input_ids"].shape[-1] > 8192:
        raise ValueError("Input exceeds 8192 tokens; no silent truncation")
    return inputs


def processor_manifest(base):
    names = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
             "preprocessor_config.json", "video_preprocessor_config.json",
             "added_tokens.json", "special_tokens_map.json", "merges.txt", "vocab.json")
    return {name: hashlib.sha256((Path(base) / name).read_bytes()).hexdigest() for name in names}


def fixture_manifest():
    from .eval import build_suite
    folder = Path(__file__).parent
    sources = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
               for name in ("eval/fixtures.py", "eval/runner.py", "policy_io.py")}
    tasks = {}
    for split in ("development", "acceptance"):
        for task in build_suite(split):
            data = {"goal": task.goal, "html": task.fixture_sha256, "allowed_ids": task.allowed_ids,
                    "reference_actions": task.reference_actions}
            tasks[task.task_id] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    return {"sources": sources, "tasks": tasks}
