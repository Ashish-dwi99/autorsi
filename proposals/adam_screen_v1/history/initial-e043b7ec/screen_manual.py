"""Local screenshot/action file bridge; no endpoint, credential or provider call.

One immutable observation and command per session/sequence. The interactive
assistant must inspect the PNG and submit a typed action; this is not a replay.
Unknown assistant token/cost usage stays unknown. No claim of API-equivalent
latency is made for tool-mediated human/assistant work.
"""
import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import time

from screen_harness import MAX_PNG, canonical, write_json
from screen_protocol import Reply, parse_actions, strict_json


def read_json(path, cap=65536):
    path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(cap + 1)
        if len(raw) > cap:
            raise ValueError("bridge record exceeds bound")
        return strict_json(raw.decode(), cap)
    finally:
        os.close(fd)


class ManualPolicy:
    def __init__(self, directory, session_id, *, timeout_s=60):
        if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{6,80}", session_id):
            raise ValueError("explicit bounded session ID required")
        if type(timeout_s) not in (int, float) or not 0 < timeout_s <= 60:
            raise ValueError("manual command wait is at most60 seconds")
        self.directory = Path(directory)
        self.session_id = session_id
        self.timeout_s = timeout_s
        self.case_id = None
        self.pending = False
        self.aborted = False
        self.seen = set()
        self.directory.mkdir(exist_ok=False)
        for name in ("observations", "commands", "consumed"):
            (self.directory / name).mkdir()

    async def __call__(self, messages, _cap):
        if self.pending or self.aborted or not self.case_id:
            raise RuntimeError("manual bridge already pending, aborted or unbound")
        current = messages[-1]["content"]
        if len(current) != 2 or current[0]["type"] != "text" or current[1]["type"] != "image_url":
            raise ValueError("expected the screenshot-only message contract")
        payload = strict_json(current[0]["text"], 16384)
        seq = payload["sequence"]
        if type(seq) is not int or not 1 <= seq <= 192 or seq in self.seen:
            raise ValueError("manual observation sequence is not fresh")
        uri = current[1]["image_url"]["url"]
        if not uri.startswith("data:image/png;base64,"):
            raise ValueError("PNG screenshot required")
        png = base64.b64decode(uri.split(",", 1)[1], validate=True)
        if len(png) > MAX_PNG or not png.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("invalid bounded PNG")
        self.pending = True
        self.seen.add(seq)
        prefix = f"{seq:06d}"
        image_path = self.directory / "observations" / (prefix + ".png")
        record = {"schema": 1, "session_id": self.session_id, "case_id": self.case_id,
                  "sequence": seq, "screenshot_path": str(image_path.resolve()),
                  "screenshot_sha256": hashlib.sha256(png).hexdigest(),
                  "observation": payload, "prior_action_feedback": messages[1:-1],
                  "published_unix": time.time(), "command_wait_seconds": self.timeout_s}
        digest = hashlib.sha256(canonical(record)).hexdigest()
        record["observation_sha256"] = digest
        command_path = self.directory / "commands" / (prefix + ".json")
        consumed_path = self.directory / "consumed" / (prefix + ".json")
        try:
            with image_path.open("xb") as stream:
                stream.write(png)
            write_json(self.directory / "observations" / (prefix + ".json"), record, exclusive=True)
            write_json(self.directory / "waiting.json", {"session_id": self.session_id, "sequence": seq,
                       "observation_sha256": digest, "status": "waiting"})
            started = time.monotonic()
            while time.monotonic() - started < self.timeout_s:
                if command_path.exists():
                    command = read_json(command_path, 8192)
                    expected = {"session_id", "case_id", "sequence", "observation_sha256", "response"}
                    if not isinstance(command, dict) or set(command) != expected:
                        raise ValueError("invalid command envelope")
                    for key in ("session_id", "case_id", "sequence", "observation_sha256"):
                        if type(command[key]) is not type(record[key]) or command[key] != record[key]:
                            raise ValueError("stale or mismatched command identity")
                    encoded = canonical(command["response"]).decode()
                    parse_actions(encoded, **payload["viewport"], remaining=payload["remaining_actions"])
                    write_json(consumed_path, {"session_id": self.session_id, "sequence": seq,
                               "observation_sha256": digest,
                               "command_sha256": hashlib.sha256(canonical(command)).hexdigest()}, exclusive=True)
                    write_json(self.directory / "waiting.json", {"session_id": self.session_id,
                               "sequence": seq, "status": "consumed"})
                    return Reply(encoded, None, None)
                await asyncio.sleep(min(0.1, max(0, self.timeout_s - (time.monotonic() - started))))
            raise TimeoutError("manual action did not arrive before its deadline")
        except BaseException:
            self.aborted = True
            write_json(self.directory / "waiting.json", {"session_id": self.session_id,
                       "sequence": seq, "status": "aborted"})
            raise
        finally:
            self.pending = False


def next_observation(directory):
    directory = Path(directory)
    waiting = read_json(directory / "waiting.json")
    if waiting.get("status") != "waiting":
        return waiting
    seq = waiting["sequence"]
    if type(seq) is not int or not 1 <= seq <= 192:
        raise ValueError("invalid waiting sequence")
    record = read_json(directory / "observations" / f"{seq:06d}.json")
    if record["session_id"] != waiting["session_id"] or record["observation_sha256"] != waiting["observation_sha256"]:
        raise ValueError("waiting observation identity mismatch")
    unsigned = dict(record)
    unsigned.pop("observation_sha256")
    if hashlib.sha256(canonical(unsigned)).hexdigest() != record["observation_sha256"]:
        raise ValueError("observation content hash mismatch")
    image_path = directory / "observations" / f"{seq:06d}.png"
    if image_path.is_symlink() or image_path.stat().st_size > MAX_PNG:
        raise ValueError("invalid screenshot file")
    if hashlib.sha256(image_path.read_bytes()).hexdigest() != record["screenshot_sha256"]:
        raise ValueError("screenshot content hash mismatch")
    return record


def submit(directory, response_text):
    directory = Path(directory)
    record = next_observation(directory)
    if "observation" not in record:
        raise ValueError("no pending observation")
    payload = record["observation"]
    response = strict_json(response_text)
    parse_actions(response_text, **payload["viewport"], remaining=payload["remaining_actions"])
    command = {key: record[key] for key in ("session_id", "case_id", "sequence", "observation_sha256")}
    command["response"] = response
    path = directory / "commands" / f"{record['sequence']:06d}.json"
    temporary = path.with_suffix(".pending")
    created = False
    try:
        write_json(temporary, command, exclusive=True)
        created = True
        # Publish fully written bytes atomically without replacing an existing
        # command. No partially written final command is visible to the worker.
        os.link(temporary, path)
    finally:
        if created and temporary.exists():
            temporary.unlink()
    return {"command_path": str(path.resolve()), "sequence": record["sequence"], "status": "submitted_once"}
