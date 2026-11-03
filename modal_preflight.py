"""Read metadata and tensor headers from existing Modal volumes; never load a model.

One CPU invocation, no retries, original volumes mounted read-only. Model tensors
are not downloaded to the laptop. Full shard hashing is deferred and is not
implied by these metadata checks.
"""
import hashlib
import json
import pathlib
import struct
import time

import modal

app = modal.App("bu-weight-space-preflight")
image = modal.Image.debian_slim(python_version="3.11")
volumes = {
    "/base": modal.Volume.from_name("hf-cache").with_mount_options(read_only=True),
    "/adapter": modal.Volume.from_name("bu-lora").with_mount_options(read_only=True),
    "/merged": modal.Volume.from_name("bu-merged").with_mount_options(read_only=True),
}


def small_json(path, limit=2_000_000):
    raw = path.read_bytes()
    if len(raw) > limit:
        raise ValueError(f"Metadata too large: {path}")
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "json": json.loads(raw)}


def header(path):
    with path.open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        if size > 8_000_000:
            raise ValueError("Unexpected safetensors header size")
        raw = f.read(size)
    tensors = json.loads(raw)
    return {"path": str(path), "bytes": path.stat().st_size,
            "header_sha256": hashlib.sha256(raw).hexdigest(), "tensors": tensors}


@app.function(image=image, volumes=volumes, cpu=1, memory=(1024, 2048),
              timeout=180, startup_timeout=180, retries=0, max_containers=1,
              min_containers=0, buffer_containers=0, scaledown_window=2)
def inspect():
    started = time.monotonic()
    snapshots = []
    for root in (pathlib.Path("/base"), pathlib.Path("/base/hub")):
        folder = root / "models--browser-use--bu-30b-a3b-preview" / "snapshots"
        if folder.is_dir():
            for snapshot in sorted(folder.iterdir()):
                if not snapshot.is_dir():
                    continue
                record = {"path": str(snapshot), "revision": snapshot.name,
                          "files": [{"name": p.name, "bytes": p.stat().st_size,
                                     "resolved_name": p.resolve().name} for p in sorted(snapshot.iterdir()) if p.is_file()]}
                for name in ("config.json", "model.safetensors.index.json", "processor_config.json", "preprocessor_config.json"):
                    if (snapshot / name).is_file():
                        record[name] = small_json(snapshot / name)
                snapshots.append(record)
    adapter_path = pathlib.Path("/adapter/adapter")
    merged_path = pathlib.Path("/merged/bu-30b-ft")
    result = {"schema": 1, "model_executed": False, "snapshots": snapshots,
              "adapter_config": small_json(adapter_path / "adapter_config.json"),
              "adapter_header": header(adapter_path / "adapter_model.safetensors"),
              "merged_config": small_json(merged_path / "config.json"),
              "merged_index": small_json(merged_path / "model.safetensors.index.json")}
    digest = hashlib.sha256()
    with (adapter_path / "adapter_model.safetensors").open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    result["adapter_sha256"] = digest.hexdigest()
    result["elapsed_seconds"] = time.monotonic() - started
    return result


@app.local_entrypoint()
def main():
    destination = pathlib.Path(__file__).parent / "evidence/private/preflight.json"
    if destination.exists():
        raise RuntimeError("Existing preflight result: inspect it instead of rerunning")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = inspect.remote()
    with destination.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"saved": str(destination), "snapshots": [s["path"] for s in result["snapshots"]],
                      "adapter_sha256": result["adapter_sha256"], "elapsed_seconds": result["elapsed_seconds"]}))
